from errors.error import Error
from tools.utils import value_from_gaussian

class Sonar:
    def __init__(self, err_distance: Error):
        self.err_distance = err_distance
    
    def measure_with_truth(self, value, mult_var: float = 1.0):
        measured = value_from_gaussian(value, self.err_distance.get_sigma(value) * mult_var)
        return value, measured
    
    def measure(self, value, mult_var: float = 1.0) -> float:
        return self.measure_with_truth(value, mult_var)[1]
