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

class AbsoluteError(Error):
    def __init__(self, error):
        self.error = error
    def get_sigma(self, value):
        return self.error /3.0

class RelativeError(Error):
    def __init__(self, error):
        self.error = error
    def get_sigma(self, value):
        return (self.error*value) /3.0

class MixedError(Error):
    def __init__(self, error, threshold):
        self.error = error
        self.threshold = threshold
    def get_sigma(self, value):
        return self.error * (self.threshold if value < self.threshold else value) /3.0

# JRC WS-12 (set velocity error RELATIVE to 5% and direction error ABSOLUTE to 1*pi/180 rad)
class Anemometer:
    def __init__(self, err_speed: Error, err_angle: Error):
        self.err_speed = err_speed
        self.err_angle = err_angle
    
    # def measure_with_truth(self, wind_velocity, boat_velocity):
    #     wind_mag, wind_angle = cartesian_to_polar(wind_velocity)
    #     boat_mag, boat_angle = cartesian_to_polar(boat_velocity)
    #     wind_vel_anemo = wind_mag - boat_mag * np.cos(mod2pi(wind_angle - boat_angle))
    #     wind_dir_anemo = mod2pi(wind_angle - boat_angle)
    #     truth = np.array([wind_vel_anemo, wind_dir_anemo])
    #     measured = np.array([value_from_gaussian(wind_vel_anemo, self.err_velocity.get_sigma(wind_vel_anemo)), value_from_gaussian(wind_dir_anemo, self.err_direction.get_sigma(wind_dir_anemo))])
    #     return truth, measured

    def measure(self, wind_velocity, boat_velocity):
        _, measured = self.measure_with_truth(wind_velocity, boat_velocity)
        return measured

    # use the correct value of the wind velocity to compute its apparent velocity, then add the error to it
    def measure_with_truth(self, true_wind_velocity, boat_velocity):
        apparent_wind_velocity = true_wind_velocity + boat_velocity
        apparent_wind_speed_truth, apparent_wind_angle_truth = cartesian_to_polar(apparent_wind_velocity)
        
        apparent_wind_speed_measured = value_from_gaussian(
            apparent_wind_speed_truth,
            self.err_speed.get_sigma(apparent_wind_speed_truth)
        )
        apparent_wind_angle_measured = value_from_gaussian(
            apparent_wind_angle_truth,
            self.err_angle.get_sigma(apparent_wind_angle_truth)
        )

        return (apparent_wind_speed_truth, apparent_wind_angle_truth), (apparent_wind_speed_measured, apparent_wind_angle_measured)

# DX900+ (set velocity error MIXED with threshold of 5m/s and 1% of error)
class Speedometer:
    def __init__(self, err_speed: Error):
        self.err_speed = err_speed
    
    def measure_with_truth(self, boat_velocity):
        boat_speed = compute_magnitude(boat_velocity)
        truth = boat_speed
        measured = value_from_gaussian(boat_speed, self.err_speed.get_sigma(boat_speed))
        return truth, measured

    def measure(self, boat_velocity):
        return self.measure_with_truth(boat_velocity)[1]

# HSC100 (set direction error ABSOLUTE to 3*pi/180 rad)
class Compass:
    def __init__(self, err_angle: Error):
        self.err_angle = err_angle
    
    def measure_with_truth(self, boat_heading):
        truth = compute_angle(boat_heading)
        measured = value_from_gaussian(truth, self.err_angle.get_sigma(truth))
        return truth, measured

    def measure(self, boat_heading):
        return self.measure_with_truth(boat_heading)[1]

# DW3000 (set distance error ABSOLUTE to 0.2m) (TWR)
class UWB:
    def __init__(self, err_distance: Error):
        self.err_distance = err_distance
    
    def measure_with_truth(self, b_actual_position, b_target_position):
        truth = compute_magnitude(b_actual_position - b_target_position)
        measured = value_from_gaussian(truth, self.err_distance.get_sigma(truth))
        return truth, measured
    
    def measure(self, b_actual_position, b_target_position):
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