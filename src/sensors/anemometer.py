from errors.error import Error
from tools.utils import compute_angle, compute_magnitude, mod2pi, value_from_gaussian

# JRC WS-12 (set velocity error RELATIVE to 5% and direction error ABSOLUTE to 1*pi/180 rad)
class Anemometer:
    def __init__(self, err_speed: Error, err_angle: Error):
        self.err_speed = err_speed
        self.err_angle = err_angle

    def measure(self, wind_velocity, boat_velocity, boat_heading, mult_var_speed: float = 1.0, mult_var_angle: float = 1.0) -> tuple[float, float]:
        _, measured = self.measure_with_truth(wind_velocity, boat_velocity, boat_heading, mult_var_speed, mult_var_angle)
        return measured

    # use the correct value of the wind velocity to compute its apparent velocity, then add the error to it
    def measure_with_truth(self, wind_velocity, boat_velocity, boat_heading, mult_var_speed: float = 1.0, mult_var_angle: float = 1.0):
        wind_relative_to_vel = wind_velocity - boat_velocity
        local_wind_speed = compute_magnitude(wind_relative_to_vel)
        local_wind_dir = mod2pi(compute_angle(wind_relative_to_vel) - compute_angle(boat_heading))
        
        local_wind_speed_measured = value_from_gaussian(
            local_wind_speed,
            self.err_speed.get_sigma(local_wind_speed) * mult_var_speed
        )
        local_wind_angle_measured = value_from_gaussian(
            local_wind_dir,
            self.err_angle.get_sigma(local_wind_dir) * mult_var_angle
        )

        return (local_wind_speed, local_wind_dir), (local_wind_speed_measured, local_wind_angle_measured)
