import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from logger import Logger
from utils import value_from_gaussian
from entities import Boat, Wind, World
from utils import *

class Sensor:
    def __init__(self, name, address):
        self.name = name
        self.address = address
        self.data = {}
    
    def add_data(self, time, data):
        self.data[time] = data
    
    def plot(self):
        sorted_data = dict(sorted(self.data.items()))
        x = sorted_data.keys()
        y = sorted_data.values()
        plt.plot(x, y)
        plt.show()

class Anemometer:
    def __init__(self, sigma):
        self.sigma = sigma
    
    def measure(self, wind: Wind, boat: Boat):
        wind_vel_anemo = (np.cos(compute_angle(wind.velocity)-boat.position_matrix()[2]) * compute_magnitude(wind.velocity)) - compute_magnitude(boat.velocity)
        return wind_vel_anemo, value_from_gaussian(wind_vel_anemo, self.sigma)

if __name__ == "__main__":
    sensor = Sensor("test", 0)
    for i in range(20):
        data = value_from_gaussian(1, 0.2)
        sensor.add_data(i, data)
        Logger.debug(data)
    sensor.plot()