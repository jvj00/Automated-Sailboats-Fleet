import numpy as np
from pid import PID

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
    def __init__(self, stepper: Stepper, pid: PID):
        self.stepper = stepper
        # [revolution/s]
        self.speed = 0
        self.steps = 0
        self.direction = StepperDirection.Clockwise
        self.pid = pid
        self.pid.limits = (0, stepper.max_speed)

    def move(self, dt):
        self.speed = self.pid.compute(self.get_angle(), dt)
        self.steps += np.floor(self.speed * self.stepper.resolution * self.direction * dt)
        self.steps %= self.stepper.resolution
        if self.steps < 0:
            self.steps += self.stepper.resolution
    
    def get_angle(self):
        return (self.steps / self.stepper.resolution) * (2 * np.pi)

    def set_target(self, angle: float):
        self.pid.set_target(angle)