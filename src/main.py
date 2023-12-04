from matplotlib import pyplot as plt
import numpy as np
from disegnino import Drawer
from entities import Boat, Wind, World

from logger import Logger

if __name__ == '__main__':
    wind = Wind()
    boat = Boat()
    world = World(wind, boat)
    drawer = Drawer(900, 500)

    velocities = []
    positions = []
    times = []

    dt = 0.1

    world.wind.velocity = np.array([20.0, 10.0])
    
    for time_elapsed in np.arange(0, 100, dt):
        if time_elapsed % 5 == 0 and  0 < time_elapsed < 10:
            world.wind.velocity = np.zeros(2)
        velocities.append(world.boat.velocity.copy())
        positions.append(world.boat.position.copy())
        times.append(time_elapsed)

        world.update(dt)

        drawer.clear()
        drawer.draw_boat(world.boat)

        plt.cla()
        plt.plot(times, list(map(lambda p: p[0], velocities)))
        plt.plot(times, list(map(lambda p: p[1], velocities)))

        plt.pause(dt)
    
    plt.show()