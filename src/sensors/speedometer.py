from errors.error import Error
from tools.utils import compute_angle, compute_magnitude, value_from_gaussian
import numpy as np

# DX900+ (set velocity error MIXED with threshold of 5m/s and 1% of error)
class Speedometer:
    def __init__(self, err_speed: Error, offset_angle: float = 0):
        self.err_speed = err_speed
        self.offset_angle = offset_angle
    
    # use the offset angle to compute the correct translation
    def measure_with_truth(self, boat_velocity, boat_heading, mult_var: float = 1.0):
        boat_speed = compute_magnitude(boat_velocity) * np.cos(compute_angle(boat_velocity)-(compute_angle(boat_heading)+self.offset_angle))
        truth = boat_speed
        measured = value_from_gaussian(boat_speed, self.err_speed.get_sigma(boat_speed) * mult_var)
        return truth, measured

    def measure(self, boat_velocity, boat_heading, mult_var: float = 1.0) -> float:
        return self.measure_with_truth(boat_velocity, boat_heading, mult_var)[1]
