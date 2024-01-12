from actuator import Stepper, StepperController
from logger import Logger
import numpy as np
from utils import *
from pid import PID
from sensor import GNSS, Compass, Anemometer, Speedometer, UWB
from typing import Optional

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

class Boat(RigidBody):
    def __init__(
            self,
            mass,
            length,
            wing: Wing,
            rudder: Rudder,
            gnss: Optional[GNSS] = None,
            compass: Optional[Compass] = None,
            anemometer: Optional[Anemometer] = None,
            speedometer: Optional[Speedometer] = None,
            uwb: Optional[UWB] = None):

        super().__init__(mass)
        self.length = length
        self.drag_damping = 0.2
        self.target = None

        if gnss is None:
            Logger.warning('No GNSS sensor provided')
        if compass is None:
            Logger.warning('No compass sensor provided')
        if anemometer is None:
            Logger.warning('No anemometer sensor provided')
        if speedometer is None:
            Logger.warning('No speedometer sensor provided')
        if uwb is None:
            Logger.warning('No UWB sensor provided')
        self.gnss = gnss
        self.compass = compass
        self.anemometer = anemometer
        self.speedometer = speedometer
        self.uwb = uwb
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

    def set_target(self, target):
        self.target = target
    
    def follow_target(self, wind: Wind, dt):
        if self.target is None:
            return
        # set the angle of rudder equal to the angle between the direction of the boat and
        # the target point
        filtered_state = self.filtered_state if self.filtered_state is not None else self.get_state()
        filtered_heading = polar_to_cartesian(1, filtered_state[2])
        angle_from_target = compute_angle_between(self.target, filtered_heading)
        self.rudder.controller.set_target(angle_from_target)
 
        # use the weighted angle between the direction of the boat and the direction of the wind as setpoint
        # for the wing pid
        _, wind_angle_local = self.measure_anemometer(wind)
        wind_angle_world = mod2pi(wind_angle_local + filtered_state[2])
        wind_direction_world = polar_to_cartesian(1, wind_angle_world)
        wind_boat_angle = compute_angle_between(filtered_heading, wind_direction_world)
        if wind_boat_angle > np.pi:
            wind_boat_angle -= np.pi
        boat_w = 0.5
        wing_angle = mod2pi(-wind_boat_angle * boat_w)
        self.wing.controller.set_target(wing_angle)
        # self.wing.controller.set_target(np.pi * 0.2)

        # self.rudder.controller.move(dt)
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
    def measure_uwb(self, target):
        return self.uwb.measure(self.position, target.position)
    def measure_rudder(self):
        return self.rudder.controller.measure_angle()    
    def measure_wing(self):
        return self.wing.controller.measure_angle()

class World:
    def __init__(self, gravity, wind: Wind):
        self.gravity_z = gravity
        self.wind = wind
    
    def update(self, boats: list[Boat], dt):
        for b in boats:
            b.follow_target(self.wind, dt)
            b.apply_wind(self.wind)
            b.apply_friction(self.gravity_z, dt)
            b.move(dt)
