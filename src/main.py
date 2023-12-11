from matplotlib import pyplot as plt
import numpy as np
from actuator import Stepper
from disegnino import Drawer
from entities import Boat, Wind, Wing, Rudder, World
from sensor import Anemometer

from logger import Logger

if __name__ == '__main__':
    wind = Wind(1.291)
    boat = Boat(100, Wing(15, Stepper(100, 0.02)), Rudder(Stepper(100, 0.05)))
    anemo = Anemometer(0.5)
    world = World(9.81, wind, boat)

    width = 900
    height = 500
    drawer = Drawer(width, height)

    velocities = []
    positions = []
    anemo_meas = []
    anemo_truth = []
    times = []

    dt = 0.1

    # spawn the boat in the center of the map
    world.boat.position = np.array([width * 0.5, height * 0.5])
    world.boat.heading = np.array([0, -1])
    world.wind.velocity = np.array([-10, 15])
    world.boat.rudder.set_target(np.pi * 0.5)
    world.boat.wing.set_target(np.pi)
    
    wing_angle = np.pi * 0.2

    for time_elapsed in np.arange(0, 1000, dt):
        if time_elapsed == 20:
            world.wind.velocity = np.zeros(2)
        if time_elapsed == 10:
            world.boat.rudder.set_target(np.pi)
        
        velocities.append(world.boat.velocity.copy())
        positions.append(world.boat.position.copy())

        truth, meas = anemo.measure(world.wind, world.boat)
        anemo_truth.append(truth)
        anemo_meas.append(meas)

        times.append(time_elapsed)

        world.update(dt)

        drawer.clear()
        drawer.draw_boat(world.boat)
        drawer.draw_wind(world.wind, [width * 0.9, height * 0.1])
        drawer.draw_vector(world.boat.position, world.wind.velocity, 'blue')
        drawer.draw_vector(world.boat.position, world.boat.velocity, 'green')

        #Plot anemometer measurements
        plt.figure(1)
        plt.cla()
        plt.plot(times, anemo_truth, label="Anemo truth")
        plt.plot(times, anemo_meas, label="Anemo meas")
        # plt.plot(times, list(map(lambda p: p[0], velocities)), label='Boat Velocity X')
        # plt.plot(times, list(map(lambda p: p[1], velocities)), label='Boat Velocity Y')

        # plt.legend()
        plt.pause(dt)
    
    plt.show()