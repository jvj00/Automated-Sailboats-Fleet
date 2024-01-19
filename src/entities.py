from ekf import EKF
from actuators import MotorController, Rudder, Stepper, StepperController, Wing
from logger import Logger
import numpy as np
from utils import *
from pid import PID
from sensor import GNSS, Compass, Anemometer, Speedometer, Sonar
from typing import Optional
from environment import SeabedMap

class RigidBody:
    def __init__(self, mass):
        self.mass = mass
        self.position = np.zeros(2)
        self.velocity = np.zeros(2)
        self.acceleration = np.zeros(2)

        self.angular_acceleration = 0
        self.angular_speed = 0

        self.heading = polar_to_cartesian(1, 0)

        self.friction_mu = 0.001

    def rotate(self, dt):
        _, curr_angle = cartesian_to_polar(self.heading)
        prev_angular_speed = self.angular_speed
        self.angular_speed = prev_angular_speed + (self.angular_acceleration * dt)
        curr_angle += 0.5 * self.angular_acceleration * (dt ** 2) + prev_angular_speed * dt
        self.heading = polar_to_cartesian(1, curr_angle)
    
    def translate(self, dt):
        self.position += (0.5 * self.acceleration * (dt ** 2) + self.velocity * dt)
        self.velocity += (self.acceleration * dt)
    
    # https://github.com/duncansykes/PhysicsForGames/blob/main/Physics_Project/Rigidbody.cpp
    def apply_friction(self, gravity: float, dt):
        friction_force = compute_friction_force(self.mass, gravity, self.friction_mu)
        velocity_decrease = -(friction_force * self.velocity * dt)
        # the velocity decrease must be always less than the current velocity
        if compute_magnitude(velocity_decrease) > compute_magnitude(self.velocity):
            self.velocity = np.zeros(2)
        else:
            self.velocity += velocity_decrease
    
class Wind:
    def __init__(self, density: float):
        self.density = density
        self.velocity = np.zeros(2)

class MotionMode:
    Wing = 0,
    Motor = 1

class Boat(RigidBody):
    def __init__(
            self,
            mass,
            length,
            wing: Wing,
            rudder: Rudder,
            motor_controller: MotorController,
            boat_seabed: SeabedMap,
            gnss: Optional[GNSS] = None,
            compass: Optional[Compass] = None,
            anemometer: Optional[Anemometer] = None,
            speedometer: Optional[Speedometer] = None,
            sonar: Optional[Sonar] = None,
            ekf: Optional[EKF] = None):

        super().__init__(mass)
        self.length = length
        self.drag_damping = 0.2
        self.target = None
        self.seabed = boat_seabed
        self.seabed.create_empty_seabed()
        self.motor_controller = motor_controller

        if gnss is None:
            Logger.warning('No GNSS sensor provided')
        if compass is None:
            Logger.warning('No compass sensor provided')
        if anemometer is None:
            Logger.warning('No anemometer sensor provided')
        if speedometer is None:
            Logger.warning('No speedometer sensor provided')
        if sonar is None:
            Logger.warning('No sonar sensor provided')
        self.gnss = gnss
        self.compass = compass
        self.anemometer = anemometer
        self.speedometer = speedometer
        self.sonar = sonar
        self.wing = wing
        self.rudder = rudder
        self.ekf = ekf
        self.filtered_state = None
    
    def update_filtered_state(self, true_wind_data, dt, update_gnss, update_compass):
        boat_sensors = self.speedometer, self.anemometer, self.rudder, self.wing, self.gnss, self.compass
        true_boat_data = self.velocity, self.heading, self.position
        self.filtered_state, filtered_variance = self.ekf.get_filtered_state(boat_sensors, true_boat_data, true_wind_data, dt, update_gnss, update_compass)
        return self.filtered_state, filtered_variance

    def get_state(self):
        return np.array(
            [
                *self.measure_gnss(),
                self.measure_compass()
            ]
        ).T

    def get_state_variance(self):
        state = self.get_state()
        return np.diag(
            [
                self.gnss.err_position_x.get_variance(state[0]),
                self.gnss.err_position_y.get_variance(state[1]),
                self.compass.err_angle.get_variance(state[2])
            ]
        )
    
    # compute the rotation rate
    # the rotation rate is directly proportional to the rudder angle and the boat velocity.
    # the higher is the rudder angle and the boat velocity, the higher will be the rotation rate of the boat
    # the result is scaled is using an angular damping
    # angular_speed = boat_speed / (boat_length / np.tan(rudder_angle)) = (boat_speed * np.tan(rudder_angle)) / boat_length
    def rotate(self, dt):
        turning_radius = compute_turning_radius(self.length, self.rudder.controller.get_angle())
        self.angular_speed = 0 if turning_radius == 0 else compute_magnitude(self.velocity) / turning_radius
        super().rotate(dt)
    
    def translate(self, dt):
        super().translate(dt)
    
    def move(self, dt):
        self.translate(dt)
        self.rotate(dt)
    
    # compute the acceleration that the wind produces to the boat
    # in order to avoid 
    def apply_wind(self, wind: Wind):
        wind_force = compute_wind_force(
            wind.velocity,
            wind.density,
            self.velocity,
            self.heading,
            polar_to_cartesian(1, self.wing.controller.get_angle()),
            self.wing.area,
            self.drag_damping
        )
        self.acceleration = compute_acceleration(wind_force, self.mass)

    def apply_motor(self):
        motor_force = compute_motor_thrust(self.motor_controller.power, self.velocity, self.heading)
        self.acceleration = compute_acceleration(motor_force, self.mass)
    
    def apply_forces(self, wind):
        motor_force = compute_motor_thrust(self.motor_controller.power, self.velocity, self.heading)
        wind_force = compute_wind_force(
            wind.velocity,
            wind.density,
            self.velocity,
            self.heading,
            polar_to_cartesian(1, self.wing.controller.get_angle()),
            self.wing.area,
            self.drag_damping
        )
        tot_force = motor_force + wind_force
        self.acceleration = compute_acceleration(tot_force, self.mass)


    def set_target(self, target):
        self.target = target
    
    def follow_target(self, wind: Wind, dt):
        if self.target is None:
            return
        
        if self.filtered_state is not None:
            boat_position = np.array([self.filtered_state[0], self.filtered_state[1]])
            boat_angle = self.filtered_state[2]
        else:
            boat_position = self.measure_gnss()
            boat_angle = self.measure_compass()
        
        # set the angle of rudder equal to the angle between the direction of the boat and
        # the target point
        filtered_heading = polar_to_cartesian(1, boat_angle)
        target_direction = self.target - boat_position

        angle_from_target = compute_angle_between(filtered_heading, target_direction)
        self.rudder.controller.set_target(angle_from_target)

        self.rudder.controller.move(dt)

        # use the weighted angle between the direction of the boat and the direction of the wind as setpoint
        # for the wing pid
        _, wind_angle_relative = self.measure_anemometer(wind)
        wind_direction_world = polar_to_cartesian(1, mod2pi(wind_angle_relative + boat_angle))

        # if the boat is upwind (controvento), switch to motor mode
        # in this case, in order to reduce the wing thrust as much as possible,
        # the wing must be placed parallel to the wind
        if np.pi * 0.5 <= wind_angle_relative <= np.pi * 1.5:
            wing_angle = mod2pi(wind_angle_relative + np.pi * 0.5)
            self.wing.controller.set_target(wing_angle)
            self.motor_controller.set_power(self.motor_controller.motor.max_power)
            print('Upwind')
        else:
            wind_boat_angle = compute_angle_between(filtered_heading, wind_direction_world)
            boat_w = 0.8
            wing_angle = mod2pi(-wind_boat_angle * (1 - boat_w))
            self.wing.controller.set_target(wing_angle)
            self.motor_controller.set_power(0)
            print('Downwind')

        self.wing.controller.move(dt)

        # Logger.debug(f'Rudder angle: {self.rudder.controller.get_angle()}')
        # Logger.debug(f'Angle from destination: {angle_from_target}')
    
    def measure_anemometer(self, wind):
        return self.anemometer.measure(wind.velocity, self.velocity, self.heading)
    def measure_speedometer(self):
        return self.speedometer.measure(self.velocity)
    def measure_compass(self):
        return self.compass.measure(self.heading)
    def measure_gnss(self):
        return self.gnss.measure(self.position)
    def measure_sonar(self, seabed):
        return self.sonar.measure(seabed.get_seabed_height(self.position[0], self.position[1]))
    def measure_rudder(self):
        return self.rudder.controller.measure_angle()    
    def measure_wing(self):
        return self.wing.controller.measure_angle()


class World:
    def __init__(self, gravity, wind: Wind, seabed: SeabedMap):
        self.gravity_z = gravity
        self.wind = wind
        self.seabed = seabed
    
    def update(self, boats: list[Boat], dt):
        for b in boats:
            b.apply_forces(self.wind)
            # b.apply_motor()
            b.apply_friction(self.gravity_z, dt)
            b.move(dt)     


# w = World(9.81, Wind(1.225), SeabedMap(min_x=-100, max_x=100, min_y=-100, max_y=100, resolution=5))
# w.seabed.create_seabed(min_z=20, max_z=100, max_slope=1, prob_go_up=0.2, plot=True)