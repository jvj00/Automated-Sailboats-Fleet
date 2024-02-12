from errors.error import Error
from sensors.sensor import Sensor
from tools.utils import value_from_gaussian

class Sonar(Sensor):
    def __init__(self, err_distance: Error, update_probability: float = 1):
        super().__init__(update_probability)
        self.err_distance = err_distance
    
    def measure_with_truth(self, value, mult_var: float = 1.0):
        measured = value_from_gaussian(value, self.err_distance.get_sigma(value) * mult_var)
        return value, measured
    
    def measure(self, value, mult_var: float = 1.0) -> float:
        if not self.can_measure():
            return None
        self.value = self.measure_with_truth(value, mult_var)[1]
        return self.value