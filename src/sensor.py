import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from logger import Logger
from utils import value_from_gaussian
from entities import Boat, Wind, World
from utils import *

class Error:
    def __init__(self):
        pass
    def get_sigma(self, value):
        pass

class AbsoluteError(Error):
    def __init__(self, error):
        self.error = error
    def get_sigma(self, value):
        return self.error/3.0

class RelativeError(Error):
    def __init__(self, error):
        self.error = error
    def get_sigma(self, value):
        return (self.error*value)/3.0

class Anemometer:
    def __init__(self, err_velocity: Error, err_direction: Error):
        self.err_velocity = err_velocity
        self.err_direction = err_direction
    

    def measure(self, wind: Wind, boat: Boat):
        wind_mag, wind_angle = cartesian_to_polar(wind.velocity)
        boat_mag, boat_angle = cartesian_to_polar(boat.velocity)
        wind_vel_anemo = wind_mag - boat_mag * np.cos(mod2pi(wind_angle - boat_angle))
        wind_dir_anemo = mod2pi(wind_angle - boat_angle)
        truth = np.array([wind_vel_anemo, wind_dir_anemo])
        measured = np.array([value_from_gaussian(wind_vel_anemo, self.err_velocity.get_sigma(wind_vel_anemo)), value_from_gaussian(wind_dir_anemo, self.err_direction.get_sigma(wind_dir_anemo))])
        return truth, measured



if __name__ == "__main__":
    from entities import Wing, Rudder, Stepper
    test_repetition = 20
    err_velocity = RelativeError(0.05)
    err_direction = AbsoluteError(np.pi/180)
    sensor = Anemometer(err_velocity=err_velocity, err_direction=err_direction)
    boat = Boat(100, Wing(15, Stepper(100, 0.05)), Rudder(Stepper(100, 0.2)))
    wind = Wind(1.291)
    anemo_truth = []
    anemo_meas = []
    outliers_direction = 0
    outliers_velocity = 0
    
    for i in range(test_repetition):
        wind.velocity = np.array([20.0, 0.0])
        boat.velocity = np.array([5.0, 5.0])
        truth, meas = sensor.measure(wind, boat)
        if np.abs(meas[0] - truth[0]) > err_velocity.error * truth[0]:
            outliers_velocity += 1
        if np.abs(meas[1] - truth[1]) > err_direction.error:
            outliers_direction += 1
        anemo_truth.append(truth)
        anemo_meas.append(meas)

    # Outliers (values over 3 times sigma, must be below 0.4%)
    print("Outliers velocity: ", outliers_velocity/test_repetition * 100, "%")
    print("Outliers direction: ", outliers_direction/test_repetition * 100, "%")

    plt.figure(1)
    plt.cla()
    plt.plot(range(test_repetition), list(map(lambda p: p[0], anemo_truth)), label="Anemo truth velocity")
    plt.plot(range(test_repetition), list(map(lambda p: p[0], anemo_meas)), label="Anemo meas velocity")
    plt.legend()

    plt.figure(2)
    plt.cla()
    plt.plot(range(test_repetition), list(map(lambda p: p[1], anemo_truth)), label="Anemo truth direction")
    plt.plot(range(test_repetition), list(map(lambda p: p[1], anemo_meas)), label="Anemo meas direction")
    plt.legend()

    plt.show()