from errors.error import Error
from sensors.sensor import Sensor
from tools.utils import compute_angle, mod2pi, value_from_gaussian

# HSC100 (set direction error ABSOLUTE to 3*pi/180 rad)
class Compass(Sensor):
    def __init__(self, err_angle: Error, update_probability: float = 1):
        super().__init__(update_probability)
        self.err_angle = err_angle
    
    def measure_with_truth(self, boat_heading, mult_var: float = 1.0):
        truth = compute_angle(boat_heading)
        measured = mod2pi(value_from_gaussian(truth, self.err_angle.get_sigma(truth) * mult_var))
        return truth, measured

    def measure(self, boat_heading, mult_var: float = 1.0) -> float:
        if not self.can_measure():
            return None
        self.value = self.measure_with_truth(boat_heading, mult_var)[1]
        return self.value
