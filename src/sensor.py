import matplotlib.pyplot as plt
import numpy as np
from logger import Logger
from pid import PID
from utils import value_from_gaussian
from utils import *

class Error:
    def __init__(self):
        pass
    def get_sigma(self, value): # consider 3 sigma rule: encapsulate 99.7% of the values in 3 sigma
        pass
    def get_variance(self, value):
        return self.get_sigma(value) ** 2

class AbsoluteError(Error):
    def __init__(self, error):
        self.error = error
    def get_sigma(self, value):
        return self.error /3.0
    def get_variance(self, value):
        return super().get_variance(value)

class RelativeError(Error):
    def __init__(self, error):
        self.error = error
    def get_sigma(self, value):
        return (self.error*np.abs(value)) /3.0
    def get_variance(self, value):
        return super().get_variance(value)

class MixedError(Error):
    def __init__(self, error, threshold):
        self.error = error
        self.threshold = threshold
    def get_sigma(self, value):
        return self.error * (self.threshold if np.abs(value) < self.threshold else np.abs(value)) /3.0
    def get_variance(self, value):
        return super().get_variance(value)

# JRC WS-12 (set velocity error RELATIVE to 5% and direction error ABSOLUTE to 1*pi/180 rad)
class Anemometer:
    def __init__(self, err_speed: Error, err_angle: Error):
        self.err_speed = err_speed
        self.err_angle = err_angle

    def measure(self, wind_velocity, boat_velocity, boat_heading) -> tuple[float, float]:
        _, measured = self.measure_with_truth(wind_velocity, boat_velocity, boat_heading)
        return measured

    # use the correct value of the wind velocity to compute its apparent velocity, then add the error to it
    def measure_with_truth(self, wind_velocity, boat_velocity, boat_heading):
        wind_relative_to_vel = wind_velocity - boat_velocity
        local_wind_speed = compute_magnitude(wind_relative_to_vel)
        local_wind_dir = mod2pi(compute_angle(wind_relative_to_vel) - compute_angle(boat_heading))
        
        local_wind_speed_measured = value_from_gaussian(
            local_wind_speed,
            self.err_speed.get_sigma(local_wind_speed)
        )
        local_wind_angle_measured = value_from_gaussian(
            local_wind_dir,
            self.err_angle.get_sigma(local_wind_dir)
        )

        return (local_wind_speed, local_wind_dir), (local_wind_speed_measured, local_wind_angle_measured)

# DX900+ (set velocity error MIXED with threshold of 5m/s and 1% of error)
class Speedometer:
    def __init__(self, err_speed: Error, parallel: bool):
        self.err_speed = err_speed
        self.parallel = parallel
    
    def measure_with_truth(self, boat_velocity, boat_heading):
        if self.parallel:
            translation = np.cos(compute_angle(boat_velocity)-compute_angle(boat_heading))
        else:
            translation = -np.sin(compute_angle(boat_velocity)-compute_angle(boat_heading))
        boat_speed = compute_magnitude(boat_velocity) * translation
        truth = boat_speed
        measured = value_from_gaussian(boat_speed, self.err_speed.get_sigma(boat_speed))
        return truth, measured

    def measure(self, boat_velocity, boat_heading) -> float:
        return self.measure_with_truth(boat_velocity, boat_heading)[1]

# HSC100 (set direction error ABSOLUTE to 3*pi/180 rad)
class Compass:
    def __init__(self, err_angle: Error):
        self.err_angle = err_angle
    
    def measure_with_truth(self, boat_heading):
        truth = compute_angle(boat_heading)
        measured = value_from_gaussian(truth, self.err_angle.get_sigma(truth))
        return truth, measured

    def measure(self, boat_heading) -> float:
        return self.measure_with_truth(boat_heading)[1]

# DW3000 (set distance error ABSOLUTE to 0.2m) (TWR)
class UWB:
    def __init__(self, err_distance: Error):
        self.err_distance = err_distance
    
    def measure_with_truth(self, b_actual_position, b_target_position):
        truth = compute_magnitude(b_actual_position - b_target_position)
        measured = value_from_gaussian(truth, self.err_distance.get_sigma(truth))
        return truth, measured
    
    def measure(self, b_actual_position, b_target_position) -> float:
        return self.measure_with_truth(b_actual_position, b_target_position)[1]

# SAM-M10Q (set position error ABSOLUTE to 1.5m for both x and y)
class GNSS:
    def __init__(self, err_position_x: Error, err_position_y: Error):
        self.err_position_x = err_position_x
        self.err_position_y = err_position_y
    
    def measure_with_truth(self, boat_position):
        truth_x = boat_position[0]
        truth_y = boat_position[1]
        measured_x = value_from_gaussian(truth_x, self.err_position_x.get_sigma(truth_x))
        measured_y = value_from_gaussian(truth_y, self.err_position_y.get_sigma(truth_y))
        return np.array([truth_x, truth_y]), np.array([measured_x, measured_y])
    
    def measure(self, boat_position):
        return self.measure_with_truth(boat_position)[1]
    
class Sonar:
    def __init__(self, err_distance: Error):
        self.err_distance = err_distance
    
    def measure_with_truth(self, value):
        measured = value_from_gaussian(value, self.err_distance.get_sigma(value))
        return value, measured
    
    def measure(self, value) -> float:
        return self.measure_with_truth(value)[1]

def test_sensor():
    from entities import Wing, Rudder, Stepper, Boat, Wind
    test_repetition = 20
    err_speed = RelativeError(0.05)
    err_angle = AbsoluteError(np.pi/180)
    sensor = Anemometer(err_speed, err_angle)
    boat = Boat(
        100,
        Wing(15, Stepper(100, 0.05)),
        Rudder(Stepper(100, 0.2)),
        PID(1, 0.1, 0.1),
        PID(1, 0.1, 0.1)
    )
    wind = Wind(1.291)
    anemo_truth = []
    anemo_meas = []
    outliers_direction = 0
    outliers_velocity = 0
    
    for i in range(test_repetition):
        wind.velocity = np.array([20.0, 0.0])
        boat.velocity = np.array([5.0, 5.0])
        truth, meas = sensor.measure_with_truth(wind.velocity, boat.velocity)
        if np.abs(meas[0] - truth[0]) > err_speed.error * truth[0]:
            outliers_velocity += 1
        if np.abs(meas[1] - truth[1]) > err_angle.error:
            outliers_direction += 1
        anemo_truth.append(truth)
        anemo_meas.append(meas)

    # Outliers (values over 3 times sigma, must be below 0.4%)
    print("Outliers velocity: ", outliers_velocity/test_repetition * 100, "%")
    print("Outliers direction: ", outliers_direction/test_repetition * 100, "%")

    plt.figure(1)
    plt.cla()
    plt.plot(range(test_repetition), list(map(lambda p: p[0], anemo_truth)), label="Anemo truth velocity")
    plt.plot(range(test_repetition), list(map(lambda p: p[0], anemo_meas)), label="Anemo meas velocity")
    plt.legend()

    plt.figure(2)
    plt.cla()
    plt.plot(range(test_repetition), list(map(lambda p: p[1], anemo_truth)), label="Anemo truth direction")
    plt.plot(range(test_repetition), list(map(lambda p: p[1], anemo_meas)), label="Anemo meas direction")
    plt.legend()

    plt.show()

if __name__ == '__main__':
    test_sensor()