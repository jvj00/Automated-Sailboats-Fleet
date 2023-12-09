from matplotlib import pyplot as plt
import numpy as np
from disegnino import Drawer
from entities import Boat, Wind, World, compute_angle

from logger import Logger

if __name__ == '__main__':
    wind = Wind()
    boat = Boat()
    world = World(wind, boat)

    width = 900
    height = 500
    drawer = Drawer(width, height)

    velocities = []
    positions = []
    times = []

    dt = 0.1

    # spawn the boat in the center of the map
    world.boat.position = np.array([width * 0.5, height * 0.5])

    world.wind.velocity = np.array([-10, 15])
    
    wing_angle = np.pi * 0.2
    world.boat.wing.heading = np.array([np.cos(wing_angle),np.cos(wing_angle)]) 

    for time_elapsed in np.arange(0, 100, dt):
        if time_elapsed % 5 == 0 and  0 < time_elapsed < 10:
            world.wind.velocity = np.zeros(2)
        velocities.append(world.boat.velocity.copy())
        positions.append(world.boat.position.copy())
        times.append(time_elapsed)

        world.update(dt)

        drawer.clear()
        drawer.draw_boat(world.boat)
        drawer.draw_wind(world.wind, [width * 0.9, height * 0.1])

        plt.cla()
        plt.plot(times, list(map(lambda p: p[0], velocities)), label='Velocity X')
        plt.plot(times, list(map(lambda p: p[1], velocities)), label='Velocity Y')

        plt.legend()

        plt.pause(dt)
    
    plt.show()