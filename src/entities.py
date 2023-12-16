from actuator import Stepper, StepperController
from logger import Logger
import numpy as np
from utils import *
from simple_pid import PID

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
    def __init__(self, mass, wing: Wing, rudder: Rudder, pid: PID):
        self.mass = mass
        self.position = np.zeros(2)
        self.velocity = np.zeros(2)
        self.acceleration = np.zeros(2)
        self.heading = polar_to_cartesian(1, 0)
        self.wing = wing
        self.rudder = rudder
        self.damping = 0.005
        self.angular_damping = 0.05
        self.target = None
        self.pid = pid

    def position_matrix(self):
        return np.array([*self.position, compute_angle(self.heading)])
    
    # compute the rotation rate
    # the rotation rate is directly proportional to the rudder angle and the boat velocity.
    # the higher is the rudder angle and the boat velocity, the higher will be the rotation rate of the boat
    # the result is scaled is using an angular damping
    def rotate(self, dt):
        rotation_rate = self.rudder.get_heading() * self.angular_damping
        self.heading += rotation_rate * compute_magnitude(self.velocity) * dt
    
    def translate(self, dt):
        self.velocity += (self.acceleration * dt)
        self.position += (self.velocity * dt)

    def move(self, dt):
        if self.target is not None:
            self.follow_target(dt)
        self.rudder.move(dt)
        self.wing.move(dt)
        self.rotate(dt)
        self.translate(dt)
        
    def apply_wind(self, wind: Wind):
        wind_force = compute_wind_force(wind, self)
        self.acceleration = compute_acceleration(wind_force, self.mass)

    # https://github.com/duncansykes/PhysicsForGames/blob/main/Physics_Project/Rigidbody.cpp
    def apply_friction(self, gravity: float, dt):
        g = normalize(self.velocity) * gravity
        self.velocity -= self.velocity * self.damping * compute_magnitude(g) * dt

    def set_target(self, target):
        self.target = target
    
    def follow_target(self, dt):
        self.pid.sample_time = dt
        distance_from_target = compute_distance(self.position, self.target)
        control = self.pid(distance_from_target)
        self.rudder.set_target(control)

class World:
    def __init__(self, gravity, wind: Wind, boat: Boat):
        self.gravity_z = gravity
        self.wind = wind
        self.boat = boat
    
    def update(self, dt):
        self.boat.apply_friction(self.gravity_z, dt)
        self.boat.apply_wind(self.wind)
        self.boat.move(dt)

def compute_acceleration(force, mass):
    return force / mass

def compute_wind_force(wind: Wind, boat: Boat):
    if compute_magnitude(wind.velocity) < 0.1:
        return np.zeros(2)
    air_mass = wind.density * boat.wing.area
    velocity = wind.velocity - boat.velocity
    wind_force = air_mass * velocity
    wind_heading = boat.wing.get_heading()
    f = np.dot(wind_force, wind_heading) * wind_heading
    return np.dot(f, boat.heading) * boat.heading
