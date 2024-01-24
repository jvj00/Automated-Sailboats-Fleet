from typing import Optional
from actuators.stepper import Stepper, StepperDirection
from pid import PID
from tools.utils import angle_from_steps, mod2pi, modpi, steps_from_angle, value_from_gaussian
import numpy as np

# controls the movement of a stepper
# the PID takes the current angle of the stepper as input, and return a speed
# for the stepper to reach the target angle
class StepperController:
    def __init__(self, stepper: Stepper, pid: PID, max_angle: Optional[float] = None):
        self.stepper = stepper
        # [revolution/s]
        self.speed = 0
        self.steps = 0
        self.direction = StepperDirection.Clockwise
        self.pid = pid
        self.max_angle = max_angle
        self.pid.limits = (-stepper.max_speed, stepper.max_speed)

    def move(self, dt):
        # the target is set with modpi, so we need to read the current angle in the same way
        current_angle = modpi(angle_from_steps(self.steps, self.stepper.resolution))
        speed = self.pid.compute(current_angle, dt)
        self.set_speed(speed)
        steps_new = self.steps + np.floor(self.speed * self.stepper.resolution * self.direction * dt)
        steps_new %= self.stepper.resolution
        if steps_new < 0:
            steps_new += self.stepper.resolution
        
        if self.max_angle is None:
            self.steps = steps_new
            return

        angle_new = angle_from_steps(steps_new, self.stepper.resolution)

        if self.max_angle < angle_new <= np.pi:
            steps_new = steps_from_angle(self.max_angle, self.stepper.resolution)
        elif np.pi < angle_new < 2*np.pi - self.max_angle:
            steps_new = steps_from_angle(mod2pi(-self.max_angle), self.stepper.resolution)
        
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
    
    # set the target angle for the PID
    # the PID control works much better if the angle is between -pi and pi,
    # not between 0 and 2pi
    def set_target(self, angle: float):
        self.pid.set_target(modpi(angle))
