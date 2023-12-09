import numpy as np
from logger import Logger

class StepperDirection:
    Clockwise = 1
    CounterClockwise = -1

# resolution is the number of steps per revolution
# higher resolution comes with a smaller angle
class Stepper:
    def __init__(self, resolution: int, max_speed):
        # [step/revolution]
        self.resolution = resolution
        # [revolution/s]
        self.max_speed = max_speed
        self.direction = StepperDirection.Clockwise
        self.steps = 0
    
    def move(self, dt):
        revolutions = self.max_speed * self.direction * dt
        self.steps += (revolutions * self.resolution)
        self.steps %= self.resolution
    
    def get_steps(self) -> int:
        return np.floor(self.steps)
    
    def get_angle(self):
        return (self.get_steps() / self.resolution) * 2 * np.pi
