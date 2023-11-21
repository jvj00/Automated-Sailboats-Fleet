import matplotlib.pyplot as plt
import pandas as pd
from customprint import Logger
from utils import value_from_gaussian

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
    

if __name__ == "__main__":
    sensor = Sensor("test", 0)
    for i in range(20):
        data = value_from_gaussian(1, 0.2)
        sensor.add_data(i, data)
        Logger.debug(data)
    sensor.plot()