from customprint import Logger
import numpy as np
import matplotlib.pyplot as plt

class Wind:
    def __init__(self):
        self.density = 1.293
        self.velocity = np.zeros(2)

class Wing:
    def __init__(self, heading):
        self.area = 15
        # heading is perpendicular to the surface of the wing, pointing forward
        self.heading = heading

class Boat:
    def __init__(self):
        self.mass = 100
        self.position = np.zeros(2)
        self.velocity = np.zeros(2)
        self.acceleration = np.zeros(2)
        self.heading = np.zeros(2)
        self.wing = Wing(self.heading)

# m_velocity -= m_velocity * m_linearDrag * a_timeStep;

# if (glm::length(m_velocity) < 0.001f)
# {
#     if (glm::length(m_velocity) < glm::length(a_gravity) * m_linearDrag * a_timeStep)
#     {
#         m_velocity = glm::vec2(0);
#     }
# }

# m_position += GetVelocity() * a_timeStep;
# ApplyForce(a_gravity * GetMass() * a_timeStep, glm::vec2(0));

class World:
    def __init__(self, wind: Wind, boat: Boat):
        self.gravity = 9.81
        self.wind = wind
        self.boat = boat
    
    def update(self, dt):
        # apply friction to the boat
        boat_weight = self.boat.mass * self.gravity
        damping_factor = 10 / boat_weight
        self.boat.velocity -= self.boat.velocity * damping_factor * dt

        if compute_magnitude(self.boat.velocity) < 0.01:
            self.boat.velocity = np.zeros(2)
        
        # apply wind force to the boat
        wind_force = compute_wind_force(self.wind.density, self.wind.velocity, self.boat.wing.area)

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
def compute_wind_force(air_density: float, wind_velocity, wing_area: float):
    air_mass = air_density * wing_area
    return air_mass * wind_velocity

if __name__ == '__main__':
    wind = Wind()
    boat = Boat()
    world = World(wind, boat)

    velocities = []
    positions = []
    times = []

    dt = 0.5

    world.wind.velocity = np.array([15.0, 15.0])

    for time_elapsed in np.arange(0, 100, dt):
        if time_elapsed % 5 == 0 and  0 < time_elapsed < 10:
            world.wind.velocity = np.zeros(2)
        velocities.append(world.boat.velocity.copy())
        positions.append(world.boat.position.copy())
        times.append(time_elapsed)

        Logger.debug(f'Boat velocity: {world.boat.velocity}, Boat position: {world.boat.position}')
        
        world.update(dt)
        
    plt.plot(times, list(map(lambda p: p[0], velocities)))
    # plt.plot(times, list(map(lambda p: p[0] / 10, positions)))
    plt.show()