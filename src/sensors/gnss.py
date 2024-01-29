from errors.error import Error
from tools.utils import value_from_gaussian
import numpy as np

# SAM-M10Q (set position error ABSOLUTE to 1.5m for both x and y)
class GNSS:
    def __init__(self, err_position_x: Error, err_position_y: Error):
        self.err_position_x = err_position_x
        self.err_position_y = err_position_y
    
    def measure_with_truth(self, boat_position, mult_var_x: float = 1.0, mult_var_y: float = 1.0):
        truth_x = boat_position[0]
        truth_y = boat_position[1]
        measured_x = value_from_gaussian(truth_x, self.err_position_x.get_sigma(truth_x) * mult_var_x)
        measured_y = value_from_gaussian(truth_y, self.err_position_y.get_sigma(truth_y) * mult_var_y)
        return np.array([truth_x, truth_y]), np.array([measured_x, measured_y])
    
    def measure(self, boat_position, mult_var_x: float = 1.0, mult_var_y: float = 1.0):
        return self.measure_with_truth(boat_position, mult_var_x, mult_var_y)[1]
