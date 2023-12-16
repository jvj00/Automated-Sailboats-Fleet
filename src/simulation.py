from matplotlib import pyplot as plt
import numpy as np
from actuator import Stepper
from entities import Boat, Wind, Wing, Rudder, World
from simple_pid import PID
from logger import Logger

def main():
    min = 0
    max = 1
    step = 0.01
    dt = 0.1
    for kd in np.arange(min, max, step):
        for ki in np.arange(min, max, step):
            for kp in np.arange(min, max, step):
                wind = Wind(1.291)
                boat_pid = PID(kp, ki, kd, setpoint=0)
                boat = Boat(50, Wing(15, Stepper(100, 1)), Rudder(Stepper(100, 1)), boat_pid)
                # anemo = Anemometer(0.5)
                world = World(9.81, wind, boat)
                world.wind.velocity = np.array([-10.0, 15.0])
                world.boat.set_target(np.array([400, 400]))
                simulate(world, dt)
                Logger.debug(f'KP: {kp} KI: {ki} KD: {kd} Final position: {world.boat.position}')

def simulate(world: World, dt):
    for time_elapsed in np.arange(0, 100, dt):
        world.update(dt)

if __name__ == '__main__':
    main()