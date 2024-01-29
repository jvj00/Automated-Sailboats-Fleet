from uuid import uuid4
import numpy as np

from tools.utils import cartesian_to_polar, compute_friction_force, compute_magnitude, polar_to_cartesian

class RigidBody:
    def __init__(self, mass):
        self.uuid = uuid4()
        self.mass = mass
        self.position = np.zeros(2)
        self.velocity = np.zeros(2)
        self.acceleration = np.zeros(2)
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
