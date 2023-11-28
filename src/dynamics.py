from customprint import Logger
import numpy as np
import matplotlib.pyplot as plt

global world
global velocities
global positions
global times

class Wind:
    def __init__(self):
        self.density = 1.293
        self.velocity = np.zeros(3)

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

def process(dt: float):
    
    boat_weight = world.gravity * world.boat.mass

    # boat/water friction setup
    friction_force_perp = world.boat.friction * boat_weight
    
    wind_force = compute_wind_force(world.wind.density, world.wind.velocity, world.boat.wing_area)
    wind_angle = np.arctan2(world.wind.velocity[1], world.wind.velocity[0])

    friction_force_x = friction_force_perp[2] * np.cos(wind_angle)
    friction_force_y = friction_force_perp[2] * np.sin(wind_angle)
    friction_force = np.array([friction_force_x, friction_force_y, 0.0])

    acceleration = compute_acceleration(wind_force - friction_force, world.boat.mass)

    world.boat.velocity = compute_velocity(acceleration, world.boat.velocity, dt)
    world.boat.position = compute_position(acceleration, world.boat.velocity, world.boat.position, dt)

    velocities.append(compute_magnitude(world.boat.velocity))
    positions.append(compute_magnitude(world.boat.position))
    times.append(time_elapsed)

    Logger.debug(f'Velocity: {world.boat.velocity}, Position: {world.boat.position}')

if __name__ == '__main__':
    wind = Wind()
    boat = Boat()
    world = World(wind, boat)

    velocities = []
    positions = []
    times = []

    dt = 0.5
    
    for time_elapsed in np.arange(0, 100, dt):
        process(dt)
            
    plt.plot(times, velocities)
    plt.show()
