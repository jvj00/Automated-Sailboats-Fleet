import numpy as np
from logger import Logger
from utils import polar_to_cartesian

class StepperDirection:
    Clockwise = 1
    CounterClockwise = -1

# resolution is the number of steps per revolution
# higher resolution comes with a smaller angle
# reference angles match with a normal clock
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
        self.steps += np.floor((revolutions * self.resolution))
        self.steps %= self.resolution
        if self.steps < 0:
            self.steps += self.resolution
    
    def get_steps(self) -> int:
        return np.floor(self.steps)
    
    def get_angle(self):
        return (self.get_steps() / self.resolution) * 2 * np.pi
    
    def set_angle(self, angle):
        self.steps = np.floor((angle / (2 * np.pi)) * self.resolution)
        self.steps %= self.resolution
        if self.steps < 0:
            self.steps += self.resolution

class StepperController:
    def __init__(self, stepper: Stepper):
        self.stepper = stepper
        self.target = stepper.get_angle()
    
    def set_target(self, angle: float):
        self.target = angle
    
    def get_angle(self):
        return self.stepper.get_angle()
    
    def get_heading(self):
        return polar_to_cartesian(1, self.get_angle())

    def move(self, dt):
        angle_delta = self.target - self.get_angle()
        stepper_resolution_angle = (2 * np.pi) / self.stepper.resolution
        if np.abs(angle_delta) < stepper_resolution_angle:
            return
        self.stepper.direction = StepperDirection.Clockwise if angle_delta > 0 else StepperDirection.CounterClockwise
        self.stepper.move(dt)
