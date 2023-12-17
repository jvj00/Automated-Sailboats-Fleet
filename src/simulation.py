from matplotlib import pyplot as plt
import numpy as np
from actuator import Stepper
from disegnino import Drawer
from entities import Boat, Wind, Wing, Rudder, World
from simple_pid import PID
from logger import Logger
from time import sleep

def main():
    width = 900
    height = 500
    # drawer = Drawer(width, height)
    min = 0
    max = 1
    step = 0.01
    dt = 0.1
    for kd in np.arange(min, max, step):
        for ki in np.arange(min, max, step):
            for kp in np.arange(min, max, step):
                wind = Wind(1.291)
                boat_pid = PID(kp, ki, kd, setpoint=0)
                boat = Boat(100, Wing(15, Stepper(100, 1)), Rudder(Stepper(100, 1)), boat_pid)
                # anemo = Anemometer(0.5)
                world = World(9.81, wind, boat)
                world.wind.velocity = np.array([-10.0, 15.0])
                world.boat.set_target(np.array([400, 400]))
                simulate(world, dt)
                Logger.debug(f'KP: {kp} KI: {ki} KD: {kd} Final position: {world.boat.position}')

def simulate(world: World, dt):
    times = []
    boat_velocities = []
    wind_velocities = []

    for time_elapsed in np.arange(0, 100, dt):
        world.update(dt)
        # Logger.debug(f'Boat position: {world.boat.position}')
        # Logger.debug(f'Rudder angle: {world.boat.rudder.get_angle()}')
        # drawer.draw_boat(world.boat)
        # drawer.draw_wind(world.wind, [drawer.win.width * 0.9, drawer.win.height * 0.1])
        # drawer.draw_vector(world.boat.position, world.wind.velocity, 'blue')
        # drawer.draw_vector(world.boat.position, world.boat.velocity, 'green')
        times.append(time_elapsed)
        boat_velocities.append(world.boat.velocity.copy())
        wind_velocities.append(world.wind.velocity.copy())

        plt.cla()
        plt.plot(times, list(map(lambda p: p[0], boat_velocities)), label='Boat Velocity X')
        plt.plot(times, list(map(lambda p: p[0], wind_velocities)), label='Wind Velocity X')

        plt.legend()
        plt.pause(dt)

if __name__ == '__main__':
    main()