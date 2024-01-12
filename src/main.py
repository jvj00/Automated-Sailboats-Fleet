from matplotlib import pyplot as plt
import numpy as np
from EKF import EKF
from actuator import Stepper, StepperController
from disegnino import Drawer
from entities import Boat, Wind, Wing, Rudder, World
from pid import PID
from logger import Logger
from utils import polar_to_cartesian

if __name__ == '__main__':
    wind = Wind(1.291)
    wind.velocity = np.array([15.0, 8.0])

    rudder_controller = StepperController(Stepper(100, 1), PID(1, 0.1, 0.1))
    wing_controller = StepperController(Stepper(100, 1), PID(1, 0.1, 0.1))

    boat = Boat(10, 10, Wing(15, wing_controller), Rudder(rudder_controller))
    boat.position = np.array([0.0, 0.0])
    boat.heading = polar_to_cartesian(1, np.pi * 0.25)
    boat.rudder.controller.set_angle(np.pi * 0.10)

    world = World(9.81, wind)

    world.add_boat(boat)

    ekf = EKF(world.boats[0], world)

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

    dt = 0.1
    
    # boat.set_target(np.array([world_width * 0.2, world_width * 0.2]))

    update_gnss = False
    update_compass = False

    for time_elapsed in np.arange(0, 100, dt):
        # if time_elapsed == 5:
        #     world.wind.velocity = -world.wind.velocity

        if time_elapsed % 10 == 0:
            update_gnss = True
            update_compass = True
        else:
            update_gnss = False
            update_compass = False

        velocities.append(world.boat.velocity.copy())
        wind_velocities.append(world.wind.velocity.copy())

        times.append(time_elapsed)

        world.update(dt)

        drawer.clear()
        drawer.draw_boat(world.boat)
        drawer.draw_wind(world.wind, np.array([world_width * 0.3, world_height * 0.3]))
        if world.boat.target is not None:
            drawer.draw_target(world.boat.target)
        drawer.draw_axis()

        x, P = ekf.get_filtered_state(update_gnss, update_compass)

        # Logger.debug(f'Wind velocity: {world.wind.velocity}')
        # Logger.debug(f'Boat velocity: {world.boat.velocity}')
        # Logger.debug(f'Boat heading: {world.boat.heading}')

        #Plot anemometer measurements
        plt.figure(1)
        plt.cla()
        plt.plot(times, list(map(lambda p: p[0], velocities)), label='Boat Velocity X')
        plt.plot(times, list(map(lambda p: p[1], velocities)), label='Boat Velocity Y')
        plt.plot(times, list(map(lambda p: p[0], wind_velocities)), label='Wind Velocity X')
        plt.plot(times, list(map(lambda p: p[1], wind_velocities)), label='Wind Velocity Y')

        plt.legend()
        plt.pause(dt)
    
    plt.show()