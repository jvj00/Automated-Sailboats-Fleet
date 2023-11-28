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
        self.wing_area = 15
        self.mass = 100.0
        self.friction = 0.02
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

        self.boat.velocity += (acceleration * dt)
        self.boat.position += (self.boat.velocity * dt)

def compute_magnitude(vec):
    return np.sqrt(np.sum([c*c for c in vec]))

def compute_acceleration(force, mass):
    return force / mass

# air_density [kg / m^3]
# wing_area [m^2]
# wind_velocity [(km/h, km/h)]
def compute_wind_force(air_density: float, wind_velocity, wing_area: float):
    air_mass = air_density * wing_area
    return air_mass * wind_velocity

if __name__ == '__main__':
    wind = Wind()
    boat = Boat()
    world = World(wind, boat)

    boat_velocities = []
    boat_positions = []
    wind_velocities = []
    times = []

    MAX_SPEED = 5

    dt = 0.1

    for time_elapsed in np.arange(0, 20, dt):
        world.wind.velocity = np.array([MAX_SPEED * np.cos(time_elapsed * 2), 0.0, 0.0])
        # Logger.debug(world.wind.velocity)

        boat_velocities.append(world.boat.velocity.copy())
        wind_velocities.append(world.wind.velocity.copy())
        boat_positions.append(world.boat.position.copy())
        times.append(time_elapsed)

        # Logger.debug(f'Boat velocity: {world.boat.velocity}, Boat position: {world.boat.position}')
        
        world.update(dt)
        
    plt.plot(times, list(map(lambda p: p[0], boat_velocities)), color='green', label='Boat vel')
    plt.plot(times, list(map(lambda p: p[0], wind_velocities)), color='orange', label='Wind vel')
    plt.plot(times, list(map(lambda p: p[0], boat_positions)), color='blue', label='Boat pos')
    plt.legend()
    # plt.plot(times, list(map(lambda p: p[1], velocities)))
    plt.show()
