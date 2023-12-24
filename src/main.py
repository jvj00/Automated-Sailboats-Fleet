from matplotlib import pyplot as plt
import numpy as np
from actuator import Stepper
from disegnino import Drawer
from entities import Boat, Wind, Wing, Rudder, World
from sensor import Anemometer
from pid import PID
from logger import Logger
from utils import polar_to_cartesian

if __name__ == '__main__':
    wind = Wind(1.291)
    rudder_pid = PID(1, 0.1, 0.1, limits=(-np.pi * 0.25, np.pi * 0.25))
    wing_pid = PID(1, 0.1, 0.1, limits=(0, np.pi * 2))
    boat = Boat(100, Wing(15, Stepper(100, 1)), Rudder(Stepper(100, 1)), rudder_pid, wing_pid)
    # anemo = Anemometer(0.5)
    world = World(9.81, wind, boat)

    win_width = 900
    win_height = 500

    world_width = 450
    world_height = 250
    drawer = Drawer(win_width, win_height, world_width, world_height)
    drawer.debug = True

    velocities = []
    wind_velocities = []
    positions = []
    anemo_meas = []
    anemo_truth = []
    times = []

    dt = 0.05

    world.boat.position = np.array([0.0, 0.0])
    world.wind.velocity = np.array([-15.0, 8.0])
    world.boat.set_target(np.array([world_width * 0.2, world_width * 0.2]))

    for time_elapsed in np.arange(0, 500, dt):

        velocities.append(world.boat.velocity.copy())
        wind_velocities.append(world.wind.velocity.copy())
        # positions.append(world.boat.position.copy())

        # truth, meas = anemo.measure(world.wind, world.boat)
        # anemo_truth.append(truth)
        # anemo_meas.append(meas)

        times.append(time_elapsed)

        world.update(dt)

        # Logger.debug(world.boat.position)
        Logger.debug(world.boat.target)

        drawer.clear()
        drawer.draw_boat(world.boat)
        drawer.draw_wind(world.wind, np.array([world_width * 0.3, world_height * 0.3]))
        drawer.draw_target(world.boat.target)
        drawer.draw_axis()

        #Plot anemometer measurements
        plt.figure(1)
        plt.cla()
        # plt.plot(times, anemo_truth, label="Anemo truth")
        # plt.plot(times, anemo_meas, label="Anemo meas")
        plt.plot(times, list(map(lambda p: p[0], velocities)), label='Boat Velocity X')
        plt.plot(times, list(map(lambda p: p[1], velocities)), label='Boat Velocity Y')
        plt.plot(times, list(map(lambda p: p[0], wind_velocities)), label='Wind Velocity X')
        plt.plot(times, list(map(lambda p: p[1], wind_velocities)), label='Wind Velocity Y')

        plt.legend()
        plt.pause(dt)
    
    plt.show()