from customprint import Logger
import numpy as np

# gravity [m/s^2]
gravity_acc = np.array([9.81])
# dynamic friction
mu_s = 0.01

class DynamicBody:
    def __init__(self, mass) -> None:
        self.mass = mass
        self.velocity = np.array([0.0, 0.0])
        self.acceleration = np.array([0.0, 0.0])
        self.position = np.array([0.0, 0.0])
    
    def apply_force(self, force):
        self.acceleration += (force / self.mass)        
    
    def update(self, dt: float):
        if compute_magnitude(self.acceleration) == 0:
            return
        friction_force = self.get_friction_force(normalize(self.acceleration))
        body_force = self.mass * self.acceleration
        current_force = body_force - friction_force
        self.acceleration = current_force / self.mass
        self.velocity = self.acceleration * dt
        self.position = self.velocity * dt
        Logger.debug(f"Acceleration: {self.acceleration}")
        Logger.debug(f"Velocity: {self.velocity}")
        Logger.debug(f"Position: {self.position}")

    def get_friction_force(self, direction):
        weight_z = compute_force(self.mass, gravity_acc)
        friction_force_z = weight_z * mu_s
        th = compute_angle(direction)
        friction_force_x = friction_force_z * np.cos(th)
        friction_force_y = friction_force_z * np.sin(th)
        return np.array([friction_force_x[0], friction_force_y[0]])

def compute_magnitude(vector) -> float:
    return (vector[0] ** 2 + vector[1] ** 2) ** 0.5

def compute_angle(vector) -> float:
    return np.arctan2(vector[1], vector[0])

def normalize(vector):
    s = sum(vector)
    return np.array([vector[0]/s, vector[1]/s])

# mass [kg]
# acc [m / s^2]
def compute_force(mass: float, acc):
    return mass * acc

# air_density [kg / m^3]
# wing_area [m^2]
# wind_velocity [(km/h, km/h)]
def compute_wind_force(wind_density: float, wind_velocity, wing_area: float):
    return wind_density * wing_area * wind_velocity

if __name__ == '__main__':
    wind_density = 1.3
    wind_velocity = np.array([-15.0, 3.0])
    wing_area = 1
    force = compute_wind_force(wind_density, wing_area, wind_velocity)
    body = DynamicBody(50)
    body.apply_force(force)
    dt = 1
    for i in range(10):
        print(f'Iteration {i}')
        body.update(dt)