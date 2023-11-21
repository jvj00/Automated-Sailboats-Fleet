import matplotlib.pyplot as plt
import pandas as pd

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
    sensor.add_data(0, 3)
    sensor.add_data(4, 2)
    sensor.add_data(2, 9)
    sensor.plot()