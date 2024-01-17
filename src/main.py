from matplotlib import pyplot as plt
import numpy as np
from EKF import EKF
from actuator import Motor, Stepper, StepperController, MotorController
from disegnino import Drawer
from entities import Boat, Wind, Wing, Rudder, World
from environment import SeabedMap
from pid import PID
from logger import Logger
from sensor import GNSS, AbsoluteError, Anemometer, Compass, MixedError, RelativeError, Speedometer
from utils import polar_to_cartesian

if __name__ == '__main__':
    wind = Wind(1.291)
    wind.velocity = np.array([15.0, 0.0])

    seabed_map = SeabedMap(min_x=-100, max_x=100, min_y=-100, max_y=100, resolution=5)

    world = World(9.81, wind, seabed_map)

    rudder_controller = StepperController(Stepper(100, 0.2), PID(0.5, 0, 0))
    wing_controller = StepperController(Stepper(100, 1), PID(1, 0.1, 0.1))
    motor_controller = MotorController(Motor(200))

    boats: list[Boat] = []

    boat = Boat(50, 10, Wing(15, wing_controller), Rudder(rudder_controller), motor_controller, seabed_map)
    boat.position = np.array([5.0, 5.0])
    boat.heading = polar_to_cartesian(1, 0)
    # boat.rudder.controller.set_angle(np.pi * 0.10)
    boat.anemometer = Anemometer(RelativeError(0.05), AbsoluteError(3 * np.pi / 180))
    boat.speedometer = Speedometer(MixedError(0.01, 5))
    boat.compass = Compass(AbsoluteError(3*np.pi/180))
    boat.gnss = GNSS(AbsoluteError(1.5), AbsoluteError(1.5))

    boats.append(boat)

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
    
    boats[0].set_target(np.array([world_width * 0.2, -world_width * 0.2]))

    boats[0].motor_controller.set_power(200)

    ekf = EKF(boats[0], world)

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

        velocities.append(boats[0].velocity.copy())
        wind_velocities.append(world.wind.velocity.copy())

        times.append(time_elapsed)

        x, P = ekf.get_filtered_state(dt, update_gnss, update_compass)
        boats[0].set_filtered_state(x)

        world.update(boats, dt)

        drawer.clear()
        drawer.draw_boat(boats[0])
        drawer.draw_wind(world.wind, np.array([world_width * 0.3, world_height * 0.3]))
        if boats[0].target is not None:
            drawer.draw_target(boats[0].target)
        drawer.draw_axis()

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