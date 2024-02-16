from errors.error import Error
from sensors.sensor import Sensor
from tools.utils import compute_angle, compute_magnitude, value_from_gaussian
import numpy as np

# DX900+ (set velocity error MIXED with threshold of 5m/s and 1% of error)
class Speedometer(Sensor):
    def __init__(self, err_speed: Error, offset_angle: float = 0, update_probability: float = 1):
        super().__init__(update_probability)
        self.err_speed = err_speed
        self.offset_angle = offset_angle
    
    # use the offset angle to compute the correct translation
    def measure_with_truth(self, boat_velocity, boat_heading, mult_var: float = 1.0):
        boat_speed = compute_magnitude(boat_velocity) * np.cos(compute_angle(boat_velocity)-(compute_angle(boat_heading)+self.offset_angle))
        truth = boat_speed
        measured = value_from_gaussian(boat_speed, self.err_speed.get_sigma(boat_speed) * mult_var)
        return truth, measured

    def measure(self, boat_velocity, boat_heading, mult_var: float = 1.0) -> float:
        if not self.can_measure():
            return None
        self.value = self.measure_with_truth(boat_velocity, boat_heading, mult_var)[1]
        return self.value
