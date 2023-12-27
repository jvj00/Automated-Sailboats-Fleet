from actuator import Stepper, StepperController
from logger import Logger
import numpy as np
from utils import *
from pid import PID
from sensor import GNSS, Compass, Anemometer, Speedometer, UWB
from typing import Optional

class Wing(StepperController):
    def __init__(self, area: float, stepper: Stepper):
        super().__init__(stepper)
        self.area = area

class Rudder(StepperController):
    def __init__(self, stepper: Stepper):
        super().__init__(stepper)

class Wind:
    def __init__(self, density: float):
        self.density = density
        self.velocity = np.zeros(2)

class Boat:
    def __init__(self, mass, wing: Wing, rudder: Rudder, rudder_pid: PID, wing_pid: PID, gnss: Optional[GNSS] = None, compass: Optional[Compass] = None, anemometer: Optional[Anemometer] = None, speedometer: Optional[Speedometer] = None, uwb: Optional[UWB] = None):
        self.mass = mass
        self.position = np.zeros(2)
        self.velocity = np.zeros(2)
        self.acceleration = np.zeros(2)
        self.heading = polar_to_cartesian(1, 0)
        self.wing = wing
        self.rudder = rudder
        self.damping = 0.005
        self.angular_damping = 0.01
        self.drag_coeff = 0.5
        self.target = None
        self.rudder_pid = rudder_pid
        self.wing_pid = wing_pid

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

    def position_matrix(self):
        return np.array([*self.position, compute_angle(self.heading)])
    
    # compute the rotation rate
    # the rotation rate is directly proportional to the rudder angle and the boat velocity.
    # the higher is the rudder angle and the boat velocity, the higher will be the rotation rate of the boat
    # the result is scaled is using an angular damping
    def rotate(self, dt):
        rotation_rate = self.rudder.get_angle() * self.angular_damping
        rotation_angle = rotation_rate * compute_magnitude(self.velocity) * dt
        current_angle = compute_angle(self.heading) + rotation_angle
        self.heading = polar_to_cartesian(1, current_angle)
    
    def translate(self, dt):
        self.velocity += (self.acceleration * dt)
        self.position += (self.velocity * dt)

    def move(self, wind, dt):
        if self.target is not None:
            self.follow_target(wind, dt)
        # self.rudder.move(dt)
        # self.wing.move(dt)
        self.rotate(dt)
        self.translate(dt)

    # compute the acceleration that the wind produces to the boat
    # in order to avoid 
    def apply_wind(self, wind: Wind):
        wind_force = compute_wind_force(wind.velocity, wind.density, self.velocity, self.heading, self.wing.get_heading(), self.wing.area, 0.5)
        self.acceleration = compute_acceleration(wind_force, self.mass)

    # https://github.com/duncansykes/PhysicsForGames/blob/main/Physics_Project/Rigidbody.cpp
    def apply_friction(self, gravity: float, dt):
        g = normalize(self.velocity) * gravity
        self.velocity -= self.velocity * self.damping * compute_magnitude(g) * dt

    def set_target(self, target):
        self.target = target
    
    def follow_target(self, wind: Wind, dt):
        # use the angle from target as setpoint for the rudder pid
        angle_from_target = compute_angle(self.target - self.position)
        self.rudder_pid.set_target(angle_from_target)
        
        # use the pid control as angle of the rudder 
        boat_angle = compute_angle(self.heading)
        control = self.rudder_pid.compute(boat_angle, dt)
        # self.rudder.set_target(control)
        self.rudder.stepper.set_angle(control)

        # use the weighted angle between the direction of the boat and the direction of the wind as setpoint
        # for the wing pid
        wind_angle = compute_angle(wind.velocity)
        boat_velocity_w = 0.7
        wind_velocity_w = 1 - boat_velocity_w
        avg_angle = (boat_velocity_w * boat_angle) + (wind_velocity_w * wind_angle)
        # self.wing.set_target(avg_angle - boat_angle)
        self.wing.stepper.set_angle(avg_angle - boat_angle)
        # Logger.debug(f'Wind angle: {wind_angle}')
        # Logger.debug(f'Boat angle: {boat_angle}')
        # Logger.debug(f'Wing angle: {self.wing.stepper.get_angle()}')
    
    def measure_anemometer(self, wind):
            return self.anemometer.measure(wind.velocity, self.velocity)
    def measure_speedometer(self):
        return self.speedometer.measure(self.velocity)
    def measure_compass(self):
        return self.compass.measure(self.heading)
    def measure_gnss(self):
        return self.gnss.measure(self.position)
    def measure_uwb(self, target):
        return self.uwb.measure(self.position, target.position)

class World:
    def __init__(self, gravity, wind: Wind, boat: Boat):
        self.gravity_z = gravity
        self.wind = wind
        self.boat = boat
    
    def update(self, dt):
        self.boat.apply_friction(self.gravity_z, dt)
        self.boat.apply_wind(self.wind)
        self.boat.move(self.wind, dt)

def compute_acceleration(force, mass):
    return force / mass

# source: ChatGPT
# F drag​ = 0.5 × CD × ρ × A × (∣Vrelative∣**2)
# compute the force generated by the wind beating to the wing
# the force is then scaled along the boat heading
def compute_wind_force(wind_velocity, wind_density, boat_velocity, boat_heading, wing_heading, wing_area, drag_coeff: float):
    v_relative = wind_velocity - boat_velocity
    v_relative_mag = compute_magnitude(v_relative)
    if v_relative_mag == 0:
        return np.zeros(2)
    f_drag_mag = 0.5 * drag_coeff * wind_density * wing_area * (v_relative_mag ** 2)
    f_wind = f_drag_mag * (v_relative / v_relative_mag)
    wing_angle_relative = compute_angle(wing_heading)
    boat_angle = compute_angle(boat_heading)
    wing_heading_absolute = polar_to_cartesian(1, boat_angle + wing_angle_relative)
    f_wind = np.dot(f_wind, wing_heading_absolute) * wing_heading_absolute
    f_wind = np.dot(f_wind, boat_heading) * boat_heading
    return f_wind