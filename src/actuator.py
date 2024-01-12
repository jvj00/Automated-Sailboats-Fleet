import numpy as np
from logger import Logger
from pid import PID
from utils import *

class StepperDirection:
    Clockwise = 1
    CounterClockwise = -1

# resolution is the number of steps per revolution
# higher resolution comes with a smaller angle
# reference angles match with a normal clock
class Stepper:
    def __init__(self, resolution: int, max_speed: float):
        # [step/revolution]
        self.resolution = resolution
        # [revolution/s]
        self.max_speed = max_speed
    
    def get_error(self):
        return (2 * np.pi) / self.resolution
    
    def get_sigma(self): # consider 3 sigma rule: encapsulate 99.7% of the values in 3 sigma
        return self.get_error() / 3.0

# controls the movement of a stepper
# the PID takes the current angle of the stepper as input, and return a speed
# for the stepper to reach the target angle
class StepperController:
    def __init__(self, stepper: Stepper, pid: PID, max_angle = np.pi * 0.2):
        self.stepper = stepper
        # [revolution/s]
        self.speed = 0
        self.steps = 0
        self.direction = StepperDirection.Clockwise
        self.pid = pid
        self.max_angle = max_angle
        self.pid.limits = (-stepper.max_speed, stepper.max_speed)

    def move(self, dt):
        speed = self.pid.compute(self.get_angle(), dt)
        self.set_speed(speed)
        steps_new = self.steps + np.floor(self.speed * self.stepper.resolution * self.direction * dt)
        steps_new %= self.stepper.resolution
        angle_new = angle_from_steps(steps_new, self.stepper.resolution)
        if steps_new < 0:
            steps_new += self.stepper.resolution
        
        if self.max_angle < angle_new <= np.pi:
            steps_new = steps_from_angle(self.max_angle, self.stepper.resolution)
        elif np.pi < angle_new < self.max_angle + np.pi:
            steps_new = steps_from_angle(self.max_angle + np.pi, self.stepper.resolution)
        
        self.steps = steps_new
        # Logger.debug(f'Steps: {self.steps}')
        # Logger.debug(f'Angle: {new_angle}')
        # self.steps = steps_from_angle(new_angle, self.stepper.resolution)
        
    def set_speed(self, speed):
        if speed < 0:
            self.speed = -speed
            self.direction = StepperDirection.CounterClockwise
        else:
            self.speed = speed
            self.direction = StepperDirection.Clockwise
    
    def set_angle(self, angle):
        self.steps = steps_from_angle(angle, self.stepper.resolution)
    
    def get_angle(self):
        return mod2pi(angle_from_steps(self.steps, self.stepper.resolution))
    
    def measure_angle(self):
        return value_from_gaussian(self.get_angle(), self.stepper.get_sigma())

    def set_target(self, angle: float):
        self.pid.set_target(angle)

def angle_from_steps(steps, resolution):
    return (steps / resolution) * (2 * np.pi)

def steps_from_angle(angle, resolution):
    return np.floor((angle / (2 * np.pi)) * resolution)
