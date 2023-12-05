from matplotlib import pyplot as plt
import numpy as np
from disegnino import Drawer
from entities import Boat, Wind, World

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
    world.boat.position[0] = width * 0.5
    world.boat.position[1] = height * 0.5

    Logger.debug(world.boat.heading)
    Logger.debug(world.boat.wing.heading)

    world.wind.velocity[0] = 10.0
    world.wind.velocity[1] = 0
    
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