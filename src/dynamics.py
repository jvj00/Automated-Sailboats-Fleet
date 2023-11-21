from customprint import Logger
import numpy as np

class DynamicBody:
    def __init__(self, mass) -> None:
        self.mass = mass
        self.velocity = np.array([0.0, 0.0])
        self.acceleration = np.array([0.0, 0.0])
    
    def apply_force(self, force):
        self.acceleration += (force / self.mass)

# mass [kg]
# acc [m / s^2]
def compute_force(mass, acc):
    return mass * acc

# air_density [kg / m^3]
# wing_area [m^2]
# wind_velocity [(km/h, km/h)]
def compute_wind_force(wind_density: float, wind_velocity, wing_area: float):
    air_mass = wind_density * wing_area
    return air_mass * wind_velocity

if __name__ == '__main__':
    force = compute_wind_force(1.3, 1, np.array([5.0, 6.0]))
    body = DynamicBody(50)
    body.apply_force(force)
    body.apply_force(force)
    body.apply_force(force)
    Logger.debug(body.acceleration)