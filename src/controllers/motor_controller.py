from actuators.motor import Motor
from tools.utils import value_from_gaussian

class MotorController:
    def __init__(self, motor: Motor):
        self.motor = motor
        self.power = 0
    
    def set_power(self, power):
        self.power = power if power < self.motor.max_power else self.motor.max_power

    def get_power(self):
        return self.power
    
    def measure_power(self):
        return value_from_gaussian(self.power, self.motor.get_sigma())
