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
        self.friction = 0.1
        self.mass = 100
        self.position = np.zeros(3)
        self.velocity = np.zeros(3)

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
        self.gravity = np.array([0.0, 0.0, 9.81])
        self.wind = wind
        self.boat = boat
    
    def update(self, dt):
        # https://github.com/duncansykes/PhysicsForGames/blob/main/Physics_Project/Rigidbody.cpp

        # apply friction to the boat
        boat.velocity -= boat.velocity * boat.friction * dt
        if compute_magnitude(boat.velocity) < 0.01 and compute_magnitude(boat.velocity) < compute_magnitude(self.gravity * boat.friction * dt):
            boat.velocity = np.zeros(3)
        
        # apply wind force to the boat
        wind_force = compute_wind_force(self.wind.density, self.wind.velocity, self.boat.wing_area)

        acceleration = compute_acceleration(wind_force, self.boat.mass)

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

    velocities = []
    positions = []
    times = []

    MAX_SPEED = 20

    dt = 0.5

    world.wind.velocity = np.array([25.0, 25.0, 0.0])

    for time_elapsed in np.arange(0, 100, dt):
        if time_elapsed % 5 == 0 and  0 < time_elapsed < 10:
            world.wind.velocity = np.zeros(3)
        velocities.append(world.boat.velocity.copy())
        positions.append(world.boat.position.copy())
        times.append(time_elapsed)

        Logger.debug(f'Boat velocity: {world.boat.velocity}, Boat position: {world.boat.position}')
        
        world.update(dt)
        
    plt.plot(times, list(map(lambda p: p[0], velocities)))
    # plt.plot(times, list(map(lambda p: p[1], velocities)))
    plt.show()