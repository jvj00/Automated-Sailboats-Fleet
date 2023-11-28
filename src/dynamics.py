from customprint import Logger
import numpy as np
import matplotlib.pyplot as plt

class Wind:
    def __init__(self):
        self.density = 1.293
        self.velocity = np.zeros(3)
    
    def get_angle(self):
        return np.arctan2(self.velocity[1], self.velocity[0])

class Boat:
    def __init__(self):
        self.wing_area = 1.5
        self.mass = 15.0
        self.friction = 0.2
        self.position = np.zeros(3)
        self.velocity = np.zeros(3)

class World:
    def __init__(self, wind: Wind, boat: Boat):
        self.gravity = np.array([0.0, 0.0, 9.81])
        self.wind = wind
        self.boat = boat
    
    def update(self, dt):
        boat_weight = self.gravity * self.boat.mass

        # boat/water friction setup
        friction_force_perp = self.boat.friction * boat_weight
        
        wind_force = compute_wind_force(self.wind.density, self.wind.velocity, self.boat.wing_area)
        wind_angle = self.wind.get_angle()
        friction_force_x = friction_force_perp[2] * np.cos(wind_angle)
        friction_force_y = friction_force_perp[2] * np.sin(wind_angle)
        friction_force = np.array([friction_force_x, friction_force_y, 0.0])

        acceleration = compute_acceleration(wind_force - friction_force, self.boat.mass)

        self.boat.velocity = compute_velocity(acceleration, self.boat.velocity, dt)
        self.boat.position = compute_position(acceleration, self.boat.velocity, self.boat.position, dt)


def compute_magnitude(vec):
    return np.sqrt(np.sum([c*c for c in vec]))

def compute_acceleration(force, mass):
    return force / mass

def compute_force(mass: float, acc):
    return mass * acc

def compute_velocity(acc_prev, vel_prev, dt):
    return vel_prev + acc_prev * dt

def compute_position(acc_prev, vel_prev, pos_prev, dt):
    return (0.5 * acc_prev * dt * dt) + (vel_prev * dt) + pos_prev

# air_density [kg / m^3]
# wing_area [m^2]
# wind_velocity [(km/h, km/h)]
def compute_wind_force(air_density: float, wind_velocity, wing_area: float):
    air_mass = air_density * wing_area
    return air_mass * wind_velocity

if __name__ == '__main__':
    wind = Wind()
    wind.velocity = np.array([13.0, 11.0, 0.0])
    boat = Boat()
    world = World(wind, boat)

    velocities = []
    positions = []
    times = []

    dt = 0.5
    
    for time_elapsed in np.arange(0, 100, dt):
        world.update(dt)
        velocities.append(world.boat.velocity)
        positions.append(world.boat.position)
        times.append(time_elapsed)

    y = list(map(lambda p: compute_magnitude(p), velocities))
    plt.plot(times, y)
    plt.show()
