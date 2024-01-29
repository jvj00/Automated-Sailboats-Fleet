from errors.error import Error
from tools.utils import compute_angle, mod2pi, value_from_gaussian

# HSC100 (set direction error ABSOLUTE to 3*pi/180 rad)
class Compass:
    def __init__(self, err_angle: Error):
        self.err_angle = err_angle
    
    def measure_with_truth(self, boat_heading, mult_var: float = 1.0):
        truth = compute_angle(boat_heading)
        measured = mod2pi(value_from_gaussian(truth, self.err_angle.get_sigma(truth) * mult_var))
        return truth, measured

    def measure(self, boat_heading, mult_var: float = 1.0) -> float:
        return self.measure_with_truth(boat_heading, mult_var)[1]

