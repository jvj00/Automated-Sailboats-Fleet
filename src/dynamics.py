from customprint import Logger
import numpy as np
import matplotlib.pyplot as plt

class DynamicBody:
    def __init__(self) -> None:
        self.velocity = np.array([0.0, 0.0, 0.0])
        self.acceleration = np.array([0.0, 0.0, 0.0])

def compute_magnitude(vec):
    return np.sqrt(np.sum([c*c for c in vec]))

def compute_acceleration(force, mass):
    return force / mass

def compute_velocity(acc_prev, vel_prev, dt):
    return vel_prev + acc_prev * dt

def compute_position(acc_prev, vel_prev, pos_prev, dt):
    return (0.5 * acc_prev * dt * dt) + (vel_prev * dt) + pos_prev

# mass [kg]
# acc [m / s^2]
def compute_force(mass: float, acc):
    return mass * acc

# air_density [kg / m^3]
# wing_area [m^2]
# wind_velocity [(km/h, km/h)]
def compute_wind_force(air_density: float, wind_velocity, wing_area: float):
    air_mass = air_density * wing_area
    return air_mass * wind_velocity

if __name__ == '__main__':
    # wind setup
    # kg / m^3
    air_density = 1.293
    wind_velocity = np.array([16.0, 8.0, 0.0])
    wing_area = 1.5
    wind_force = compute_wind_force(air_density, wind_velocity, wing_area)
    wind_angle = np.arctan2(wind_velocity[1], wind_velocity[0])

    # boat setup
    gravity = np.array([0.0, 0.0, 9.81])
    mass = 15.0
    weight = gravity * mass

    # boat/water friction setup
    friction = 0.2
    friction_force_perp = friction * weight
    friction_force_x = friction_force_perp[2] * np.cos(wind_angle)
    friction_force_y = friction_force_perp[2] * np.sin(wind_angle)
    friction_force = np.array([friction_force_x, friction_force_y, 0.0])

    acceleration = compute_acceleration(wind_force - friction_force, mass)

    position = np.zeros(3)
    velocity = np.zeros(3)

    # plot setup
    positions = []
    velocities = []
    dts = []
    dt = 0.5

    for time_elapsed in np.arange(1, 100, dt):
        velocity = compute_velocity(acceleration, velocity, dt)
        position = compute_position(acceleration, velocity, position, dt)
        velocities.append(compute_magnitude(velocity))
        positions.append(compute_magnitude(position))
        dts.append(time_elapsed)
        Logger.debug(f'Velocity: {velocity}, Position: {position}')
    
    plt.plot(dts, velocities)
    # plt.plot(dts, positions)
    plt.show()