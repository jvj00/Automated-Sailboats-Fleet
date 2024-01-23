from ekf import EKF
from actuators import MotorController, Rudder, Stepper, StepperController, Wing
from logger import Logger
import numpy as np
from utils import *
from pid import PID
from sensor import GNSS, UWB, Compass, Anemometer, Speedometer, Sonar
from typing import Optional
from environment import SeabedMap, SeabedBoatMap
from uuid import uuid4

class RigidBody:
    def __init__(self, mass):
        self.uuid = uuid4()
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
        curr_angle += self.angular_speed * dt
        self.heading = polar_to_cartesian(1, curr_angle)
    
    def translate(self, dt):
        self.position += (0.5 * self.acceleration * (dt ** 2) + self.velocity * dt)
    
    # https://github.com/duncansykes/PhysicsForGames/blob/main/Physics_Project/Rigidbody.cpp
    def apply_friction(self, gravity: float, dt):
        friction_force = compute_friction_force(self.mass, gravity, self.friction_mu)
        velocity_decrease = -(friction_force * self.velocity * dt)
        # the velocity decrease must be always less than the current velocity
        if compute_magnitude(velocity_decrease) > compute_magnitude(self.velocity):
            self.velocity = np.zeros(2)
        else:
            self.velocity += velocity_decrease
    
    def apply_acceleration_to_velocity(self, dt):
        self.velocity += (self.acceleration * dt)
    
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
            map: Optional[SeabedBoatMap] = None,
            wing: Optional[Wing] = None,
            rudder: Optional[Rudder] = None,
            motor_controller: Optional[MotorController] = None,
            gnss: Optional[GNSS] = None,
            compass: Optional[Compass] = None,
            anemometer: Optional[Anemometer] = None,
            speedometer_par: Optional[Speedometer] = None,
            speedometer_perp: Optional[Speedometer] = None,
            sonar: Optional[Sonar] = None,
            uwb: Optional[UWB] = None,
            ekf: Optional[EKF] = None):

        super().__init__(mass)
        self.length = length
        self.drag_damping = 0.2
        self.target = None
        self.map = map
        self.motor_controller = motor_controller
        self.trigger_motor = False

        if gnss is None:
            Logger.warning('No GNSS sensor provided')
        self.gnss = gnss

        if compass is None:
            Logger.warning('No compass sensor provided')
        self.compass = compass

        if anemometer is None:
            Logger.warning('No anemometer sensor provided')
        self.anemometer = anemometer

        if speedometer_par is None:
            Logger.warning('No parallel speedometer sensor provided')
        self.speedometer_par = speedometer_par

        if speedometer_perp is None:
            Logger.warning('No perpendicular speedometer sensor provided')
        self.speedometer_perp = speedometer_perp

        if sonar is None:
            Logger.warning('No sonar sensor provided')
        self.sonar = sonar

        if uwb is None:
            Logger.warning('No UWB sensor provided')
        self.uwb = uwb

        if wing is None:
            Logger.warning('No wing provided')
        self.wing = wing        

        if rudder is None:
            Logger.warning('No rudder provided')
        self.rudder = rudder

        if ekf is None:
            Logger.warning('No EKF provided')
        
        if map is None:
            Logger.warning('No map provided')
        self.ekf = ekf
        self.filtered_state = None
    
    def update_filtered_state(self, true_wind_data, dt, update_gnss, update_compass):
        boat_sensors = self.speedometer_par, self.speedometer_perp, self.anemometer, self.rudder, self.wing, self.motor_controller, self.gnss, self.compass
        true_boat_data = self.velocity, self.heading, self.position
        self.filtered_state, filtered_variance = self.ekf.get_filtered_state(boat_sensors, true_boat_data, true_wind_data, dt, update_gnss, update_compass)
        return self.filtered_state, filtered_variance

    def get_state(self):
        return np.array(
            [
                *self.position,
                compute_angle(self.heading)
            ]
        ).T
    
    def measure_state(self):
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
        if self.rudder is not None:
            turning_radius = compute_turning_radius(self.length, self.rudder.controller.get_angle())
            self.angular_speed = 0 if turning_radius == 0 else np.dot(self.velocity, self.heading) / turning_radius
        super().rotate(dt)
    
    def translate(self, dt):
        super().translate(dt)
    
    def move(self, dt):
        self.translate(dt)
        self.rotate(dt)
    
    def apply_forces(self, wind, dt):
        motor_force = 0
        wind_force = 0
        
        if self.motor_controller is not None:
            motor_force = compute_motor_thrust(self.motor_controller.power, self.motor_controller.motor.efficiency, self.velocity, self.heading)
        
        if self.wing is not None:
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
    
    # enable simulation data to use boat and wind data from the simulation
    # enable measured_data to use boat and wind measured data
    # set both to False to use filtered data coming from the kalman filter, if available
    def follow_target(self, wind: Wind, dt, boats: list['Boat'] = None, simulated_data = False, measured_data = False, motor_only = False, wing_only = False):
        if self.target is None:
            return

        boat_position = None
        boat_angle = None
        wind_speed = None
        wind_angle = None
        
        if simulated_data:
            boat_position = self.position
            boat_angle = compute_angle(self.heading)
            # convert wind data (absolute) to relative to the boat
            wind_speed, wind_angle = cartesian_to_polar(wind.velocity - self.velocity)
            wind_angle = mod2pi(wind_angle - compute_angle(self.heading))
        
        elif measured_data:
            if self.compass is not None and self.gnss is not None and self.anemometer is not None:
                boat_position = self.measure_gnss()
                boat_angle = self.measure_compass()
                wind_speed, wind_angle = self.measure_anemometer(wind)
            
        else:
            if self.filtered_state is not None:
                boat_position = np.array([self.filtered_state[0], self.filtered_state[1]])
                boat_angle = self.filtered_state[2]
                wind_speed, wind_angle = self.measure_anemometer(wind)
            
        if boat_position is None or boat_angle is None or wind_speed is None or wind_angle is None:
            Logger.error('Cannot follow target due to missing values')
            return

        if self.rudder is not None:
            # avoiding_collisions = False
            # if boats is not None:
            #     for b in boats:
            #         offset = 2
            #         # if the two boats are close enough, compute the angle bewteen their directions, and
            #         # move the rudder 
            #         radius = (b.length + self.length) * 0.5 + offset
            #         center = b.position
            #         start = self.position
            #         end = self.heading * 5
            #         if check_intersection(center, radius, start, end):
            #             angle = compute_angle_between(b.heading, self.heading)
            #             self.rudder.controller.set_target(angle)
            #             avoiding_collisions = True
            
            # if not avoiding_collisions:
                # set the angle of rudder equal to the angle between the direction of the boat and
                # the target point
                filtered_heading = polar_to_cartesian(1, boat_angle)
                target_direction = self.target - boat_position
                angle_from_target = mod2pi(-compute_angle_between(filtered_heading, target_direction))
                self.rudder.controller.set_target(angle_from_target)
                self.rudder.controller.move(dt)

        # if the boat is upwind (controvento), switch to motor mode
        # in this case, in order to reduce the wing thrust as much as possible,
        # the wing must be placed parallel to the wind
        if is_angle_between(wind_angle, np.pi * 1/2, np.pi * 3/2):
            self.trigger_motor = True
        elif is_angle_between(wind_angle, 0, np.pi * 1/3) or is_angle_between(wind_angle, np.pi * 5/3, np.pi * 2):
            self.trigger_motor = False

        if self.trigger_motor or motor_only:
            if self.wing is not None:
                wing_angle = self.wing.controller.get_angle()
                if np.abs(wing_angle - mod2pi(wind_angle + np.pi * 0.5)) > np.abs(wing_angle - mod2pi(wind_angle - np.pi * 0.5)):
                    wing_angle = mod2pi(wind_angle - np.pi * 0.5)
                    # print("POS")
                else:
                    wing_angle = mod2pi(wind_angle + np.pi * 0.5)
                    # print("NEG")
                self.wing.controller.set_target(wing_angle)
            if self.motor_controller is not None:
                self.motor_controller.set_power(self.motor_controller.motor.max_power)
        else:
            if self.wing is not None:
                boat_w = 0.5
                wing_angle = mod2pi(wind_angle * boat_w)
                self.wing.controller.set_target(wing_angle)
            if self.motor_controller is not None:
                self.motor_controller.set_power(0)
    
        if self.wing is not None:
            self.wing.controller.move(dt)

        # Logger.debug(f'Rudder angle: {self.rudder.controller.get_angle()}')
        # Logger.debug(f'Angle from destination: {angle_from_target}')
    
    def measure_anemometer(self, wind):
        return self.anemometer.measure(wind.velocity, self.velocity, self.heading)
    def measure_speedometer_par(self):
        return self.speedometer_par.measure(self.velocity, self.heading)
    def measure_speedometer_perp(self):
        return self.speedometer_perp.measure(self.velocity, self.heading)
    def measure_compass(self):
        return self.compass.measure(self.heading)
    def measure_gnss(self):
        return self.gnss.measure(self.position)
    def measure_sonar(self, seabed: SeabedMap, filtered_pos):
        try:
            meas = self.sonar.measure(seabed.get_seabed_height(self.position[0], self.position[1]))
            self.map.insert_measure(filtered_pos[0], filtered_pos[1], meas)
        except Exception as e:
            Logger.error(e)
            meas = 0
        return meas
    def measure_rudder(self):
        return self.rudder.controller.measure_angle()    
    def measure_wing(self):
        return self.wing.controller.measure_angle()


class World:
    def __init__(self, gravity, wind: Wind, seabed: Optional[SeabedMap] = None):
        self.gravity_z = gravity
        self.wind = wind
        self.seabed = seabed
    
    def update(self, boats: list[Boat], dt):
        for b in boats:
            b.apply_forces(self.wind, dt)
            b.move(dt)
            b.apply_acceleration_to_velocity(dt)
            b.apply_friction(self.gravity_z, dt)


# w = World(9.81, Wind(1.225), SeabedMap(min_x=-100, max_x=100, min_y=-100, max_y=100, resolution=5))
# w.seabed.create_seabed(min_z=20, max_z=100, max_slope=1, prob_go_up=0.2, plot=True)