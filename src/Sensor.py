import matplotlib.pyplot as plt
import pandas as pd

class Sensor:
    def __init__(self, name, address):
        self.name = name
        self.address = address
        self.sensor_data = {}
    
    def add_data(self, time, data):
        self.sensor_data[time] = data
    
    def plot(self):
        sorted_sensor_data = dict(sorted(self.sensor_data.items()))
        x = sorted_sensor_data.keys()
        y = sorted_sensor_data.values()
        plt.plot(x, y)
        plt.show()

if __name__ == "__main__":
    sensor = Sensor("test", 0)
    sensor.add_data(0, 3)
    sensor.add_data(4, 2)
    sensor.add_data(2, 9)
    sensor.plot()