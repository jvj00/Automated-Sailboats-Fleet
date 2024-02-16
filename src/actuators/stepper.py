import numpy as np

class StepperDirection:
    Clockwise = 1
    CounterClockwise = -1

class Stepper:
    # max_speed: max speed of the stepper (revolution/s)
    # resolution: number of steps per revolution (step/revolution)
    def __init__(self, resolution: int, max_speed: float):
        self.resolution = resolution
        self.max_speed = max_speed
    
    def get_error(self):
        return (2 * np.pi) / self.resolution
    
    def get_sigma(self): # consider 3 sigma rule: encapsulate 99.7% of the values in 3 sigma
        return self.get_error() / 3.0
    
    def get_variance(self):
        return self.get_sigma()**2
