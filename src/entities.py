from actuator import MotorController, Stepper, StepperController
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
    
class Wing:
    def __init__(self, area: float, controller: StepperController):
        self.area = area
        self.controller = controller

class Rudder:
    def __init__(self, controller: StepperController):
        self.controller = controller

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
            sonar: Optional[Sonar] = None):

        super().__init__(mass)
        self.length = length
        self.drag_damping = 0.2
        self.target = None
        self.seabed = boat_seabed
        self.seabed.create_empty_seabed()
        self.motor_controller = motor_controller
        self.motion_mode = MotionMode.Motor

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
        self.filtered_state = None
        
    def set_filtered_state(self, state):
        self.filtered_state = state

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
                self.gnss.err_position_x.get_sigma(state[0])**2,
                self.gnss.err_position_y.get_sigma(state[1])**2,
                self.compass.err_angle.get_sigma(state[2])**2
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
        if self.motion_mode == MotionMode.Motor:
            return
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
        if self.motion_mode == MotionMode.Wing:
            return
        motor_force = compute_motor_thrust(self.motor_controller.power, self.velocity, self.heading)
        self.acceleration = compute_acceleration(motor_force, self.mass)

    def set_target(self, target):
        self.target = target
    
    def follow_target(self, wind: Wind, dt):
        if self.target is None:
            return
        
        # set the angle of rudder equal to the angle between the direction of the boat and
        # the target point
        filtered_state = self.filtered_state if self.filtered_state is not None else self.get_state()
        # filtered_heading = polar_to_cartesian(1, filtered_state[2])
        filtered_heading = self.heading
        # target_direction = self.target - np.array([filtered_state[0], filtered_state[1]])
        target_direction = self.target - self.position

        # if the boat is heading to the target and the boat is upwind, switch to motor mode
        if np.dot(target_direction, filtered_heading) > 0 and np.dot(wind.velocity, filtered_heading) < 0:
            self.motion_mode = MotionMode.Motor
        else:
            self.motion_mode = MotionMode.Wing

        angle_from_target = compute_angle_between(filtered_heading, target_direction)
        self.rudder.controller.set_target(angle_from_target)

        self.rudder.controller.move(dt)

        if self.motion_mode == MotionMode.Motor:
            return
 
        # use the weighted angle between the direction of the boat and the direction of the wind as setpoint
        # for the wing pid
        # _, wind_angle_local = self.measure_anemometer(wind)
        # wind_angle_world = mod2pi(wind_angle_local + filtered_state[2])
        # wind_direction_world = polar_to_cartesian(1, wind_angle_world)
        wind_direction_world = normalize(wind.velocity)
        wind_boat_angle = compute_angle_between(filtered_heading, wind_direction_world)
        boat_w = 0.8
        wing_angle = mod2pi(-wind_boat_angle * (1 - boat_w))
        self.wing.controller.set_target(wing_angle)

        self.wing.controller.move(dt)
       
        # Logger.debug(f'Wind angle: {wind_angle}')
        # Logger.debug(f'Wing angle: {self.wing.controller.get_angle()}')
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
            b.follow_target(self.wind, dt)
            b.apply_wind(self.wind)
            b.apply_motor()
            b.apply_friction(self.gravity_z, dt)
            b.move(dt)     


# w = World(9.81, Wind(1.225), SeabedMap(min_x=-100, max_x=100, min_y=-100, max_y=100, resolution=5))
# w.seabed.create_seabed(min_z=20, max_z=100, max_slope=1, prob_go_up=0.2, plot=True)