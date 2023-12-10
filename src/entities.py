from actuator import Rudder
from logger import Logger
import numpy as np
import matplotlib.pyplot as plt
from utils import *

class Wing:
    def __init__(self, area):
        self.area = area
        # heading is perpendicular to the surface of the wing, pointing forward
        self.heading = np.array([1, 0])

class Wind:
    def __init__(self, density):
        self.density = density
        self.velocity = np.zeros(2)

class Boat:
    def __init__(self, mass, wing: Wing, rudder: Rudder):
        self.mass = mass
        self.position = np.zeros(2)
        self.velocity = np.zeros(2)
        self.acceleration = np.zeros(2)
        self.heading = np.zeros(2)
        self.wing = wing
        self.rudder = rudder
        self.damping = 0.005
    
    def position_matrix(self):
        return np.array([*self.position, compute_angle(self.heading)])
    
    def move(self, dt):
        self.rudder.move(dt)
        angle = self.rudder.get_angle()
        x_h = np.cos(angle)
        y_h = np.sin(angle)
        self.heading = np.array([x_h, y_h])
        self.acceleration = np.dot(self.acceleration, self.heading) * self.heading
        self.velocity += (self.acceleration * dt)
        self.position += (self.velocity * dt)
    
    def apply_wind(self, wind: Wind):
        wind_force = compute_wind_force(wind, self.wing)
        wind_force = np.dot(wind_force, self.heading) * self.heading
        self.acceleration = compute_acceleration(wind_force, self.mass)

    # https://github.com/duncansykes/PhysicsForGames/blob/main/Physics_Project/Rigidbody.cpp
    def apply_friction(self, gravity: float, dt):
        g = normalize(self.velocity) * gravity
        self.velocity -= self.velocity * self.damping * compute_magnitude(g) * dt

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

# air_density [kg / m^3]
# wing_area [m^2]
# wind_velocity [(m/s, m/s)]
# computes the force generated by the wind on the wing of a boat
def compute_wind_force(wind: Wind, wing: Wing):
    air_mass = wind.density * wing.area
    wind_force = air_mass * wind.velocity
    return np.dot(wind_force, wing.heading) * wing.heading
