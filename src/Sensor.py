import matplotlib.pyplot as plt
import pandas as pd

class Sensor:
    def __init__(self, name, address):
        self.name = name
        self.address = address
        self.sensor_data = pd.DataFrame()
    
    def add_data(self, time, data):
        self.sensor_data[time] = data
    
    def plot(self):
        plt.plot(self.sensor_data)
        plt.show()

if __name__ == "__main__":
    sensor = Sensor("test", 0)
    sensor.add_data(0, 3)
    sensor.add_data(1, 2)
    sensor.add_data(2, 9)
    sensor.plot()