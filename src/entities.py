from logger import Logger
import numpy as np
import matplotlib.pyplot as plt

class Wing:
    def __init__(self, heading):
        self.area = 15
        # heading is perpendicular to the surface of the wing, pointing forward
        self.heading = heading

class Boat:
    def __init__(self):
        self.mass = 50
        self.position = np.zeros(2)
        self.velocity = np.zeros(2)
        self.acceleration = np.zeros(2)
        self.heading = np.zeros(2)
        self.wing = Wing(self.heading)
    
    def to_matrix(self):
        return np.array([*self.position, compute_angle(self.heading)])

class Wind:
    def __init__(self):
        self.density = 1.293
        self.velocity = np.zeros(2)

class World:
    def __init__(self, wind: Wind, boat: Boat):
        self.gravity_z = 9.81
        self.wind = wind
        self.boat = boat
    
    def update(self, dt):
        # apply friction to the boat
        # https://github.com/duncansykes/PhysicsForGames/blob/main/Physics_Project/Rigidbody.cpp
        velocity_angle = compute_angle(self.boat.velocity)
        gravity = np.array([self.gravity_z * np.cos(velocity_angle), self.gravity_z * np.sin(velocity_angle)])
        damping = 0.01
        self.boat.velocity -= self.boat.velocity * damping * compute_magnitude(gravity) * dt

        # apply wind force to the boat
        wind_force = compute_wind_force(self.wind, self.boat)

        self.boat.acceleration = compute_acceleration(wind_force, self.boat.mass)
        self.boat.velocity += (self.boat.acceleration * dt)
        self.boat.position += (self.boat.velocity * dt)

def compute_angle(vec):
    return np.arctan2(vec[1], vec[0])

def compute_magnitude(vec):
    return np.sqrt(vec[0]*vec[0] + vec[1]*vec[1])

def compute_acceleration(force, mass):
    return force / mass

# air_density [kg / m^3]
# wing_area [m^2]
# wind_velocity [(km/h, km/h)]
# computes the force generated by the wind on the wing of a boat
# the force is max when the wind is perpendicular to the wing, and 0 when it's parallel
# so multiply the force by a gain that is computed using the sin of the angle between the
# wing and the wind
def compute_wind_force(wind: Wind, boat: Boat):
    air_mass = wind.density * boat.wing.area
    gain = np.abs(np.sin(compute_angle(wind.velocity)))
    return air_mass * wind.velocity * gain
