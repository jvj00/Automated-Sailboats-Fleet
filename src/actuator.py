import numpy as np
from logger import Logger
from pid import PID
from utils import mod2pi

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

# controls the movement of a stepper
# the PID takes the current angle of the stepper as input, and return a speed
# for the stepper to reach the target angle
class StepperController:
    def __init__(self, stepper: Stepper, pid: PID, limits = (0, np.pi * 2)):
        self.stepper = stepper
        # [revolution/s]
        self.speed = 0
        self.steps = 0
        self.direction = StepperDirection.Clockwise
        self.pid = pid
        self.limits = (mod2pi(limits[0]), mod2pi(limits[1]))
        self.pid.limits = (-stepper.max_speed, stepper.max_speed)

    def move(self, current_value, dt):
        speed = self.pid.compute(current_value, dt)
        self.set_speed(speed)
        steps_new = self.steps + np.floor(self.speed * self.stepper.resolution * self.direction * dt)
        steps_new %= self.stepper.resolution
        # if steps_new < 0:
            # steps_new += self.stepper.resolution
        # if is_bounded_2pi(angle_from_steps(steps_new, self.stepper.resolution), self.limits[0], self.limits[1]):
            # self.steps = steps_new
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
    
    def get_angle(self):
        return angle_from_steps(self.steps, self.stepper.resolution)

    def set_target(self, angle: float):
        self.pid.set_target(angle)

def angle_from_steps(steps, resolution):
    return (steps / resolution) * (2 * np.pi)

def steps_from_angle(angle, resolution):
    return (angle / (2 * np.pi)) * resolution

def is_bounded_2pi(angle, min, max):
        angle = mod2pi(angle)
        if min < max:
            return min < angle < max
        else:
            return 0 < angle < min or max < angle < 2 * np.pi