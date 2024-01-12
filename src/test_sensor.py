import unittest
import numpy as np
from entities import Wing, Rudder, Stepper, Boat, Wind
from pid import PID
from sensor import Anemometer, Speedometer, Compass, UWB, GNSS, RelativeError, AbsoluteError, MixedError
from utils import polar_to_cartesian

test_repetition = 10000

class AnemometerTest(unittest.TestCase):
    err_speed = RelativeError(0.05)
    err_angle = AbsoluteError(np.pi/180)
    sensor = Anemometer(err_speed, err_angle)
    boat = Boat(100, Wing(15, Stepper(100, 0.05)), Rudder(Stepper(100, 0.2)), PID(), PID())
    wind = Wind(1.291)
    
    def test_outliers(self):
        self.wind.velocity = np.array([np.random.rand()*20, np.random.rand()*20])
        self.boat.velocity = np.array([np.random.rand()*20, np.random.rand()*20])
        list_truth = []
        list_meas = []
        outliers_direction = 0
        outliers_velocity = 0
        
        for i in range(test_repetition):
            truth, meas = self.sensor.measure_with_truth(self.wind.velocity, self.boat.velocity, self.boat.heading)
            if np.abs(meas[0] - truth[0]) > np.abs(self.err_speed.error * truth[0]):
                outliers_velocity += 1
            if np.abs(meas[1] - truth[1]) > self.err_angle.error:
                outliers_direction += 1
            list_truth.append(truth)
            list_meas.append(meas)
        
        # print(outliers_velocity/test_repetition)
        # print(outliers_direction/test_repetition)
        self.assertLess(outliers_velocity/test_repetition, 0.004)
        self.assertLess(outliers_direction/test_repetition, 0.004)
    
    def test_sensor_boat_wind_aligned_same_direction(self):
        self.wind.velocity = np.array([10.0, 0.0])
        self.boat.velocity = np.array([0.0, 0.0])
        self.boat.heading = polar_to_cartesian(1, 0)
        truth, meas = self.sensor.measure_with_truth(self.wind.velocity, self.boat.velocity, self.boat.heading)
        self.assertEqual(truth[0], 10.0)
        self.assertEqual(truth[1], 0)

    def test_sensor_boat_wind_perpendicular_1(self):
        self.wind.velocity = np.array([10.0, 0.0])
        self.boat.velocity = np.array([5.0, 0.0])
        self.boat.heading = polar_to_cartesian(1, np.pi * 0.5)
        truth, meas = self.sensor.measure_with_truth(self.wind.velocity, self.boat.velocity, self.boat.heading)
        self.assertEqual(truth[0], 5.0)
        self.assertEqual(truth[1], np.pi * 1.5)
    
    def test_sensor_boat_wind_aligned_opposite_direction(self):
        self.wind.velocity = np.array([-10.0, 0.0])
        self.boat.velocity = np.array([5.0, 0.0])
        self.boat.heading = polar_to_cartesian(1, 0)
        truth, meas = self.sensor.measure_with_truth(self.wind.velocity, self.boat.velocity, self.boat.heading)
        self.assertEqual(truth[0], 15.0)
        self.assertEqual(truth[1], np.pi)
    
    def test_sensor_boat_wind_perpendicular_2(self):
        self.wind.velocity = np.array([-10.0, 0.0])
        self.boat.velocity = np.array([5.0, 0.0])
        self.boat.heading = polar_to_cartesian(1, np.pi * 1.5)
        truth, meas = self.sensor.measure_with_truth(self.wind.velocity, self.boat.velocity, self.boat.heading)
        self.assertEqual(truth[0], 15.0)
        self.assertEqual(truth[1], np.pi * 1.5)

class SpeedometerTest(unittest.TestCase):

    def test_outliers_high_velocities(self):
        err_speed = MixedError(0.01, 5)
        sensor = Speedometer(err_speed)
        boat = Boat(100, Wing(15, Stepper(100, 0.05)), Rudder(Stepper(100, 0.2)), PID(1, 0.1, 0.1, 0.1), PID(1, 0.1, 0.1, 0.1))
        x=np.random.rand()*15+4
        y=np.random.rand()*15+4
        boat.velocity = np.array([np.random.rand()*15+5, np.random.rand()*15+5])
        list_truth = []
        list_meas = []
        outliers_velocity = 0

        for i in range(test_repetition):
            truth, meas = sensor.measure_with_truth(boat.velocity)
            if np.abs(meas - truth) > np.abs(err_speed.error * truth):
                outliers_velocity += 1
            list_truth.append(truth)
            list_meas.append(meas)

        self.assertLess(outliers_velocity/test_repetition, 0.004)

    def test_outliers_low_velocities(self):
        err_speed = MixedError(0.01, 5)
        sensor = Speedometer(err_speed)
        boat = Boat(100, Wing(15, Stepper(100, 0.05)), Rudder(Stepper(100, 0.2)), PID(1, 0.1, 0.1, 0.1), PID(1, 0.1, 0.1, 0.1))
        x=np.random.rand()*5
        y=np.random.rand()*np.sqrt(25-x**2)
        boat.velocity = np.array([x, y])
        list_truth = []
        list_meas = []
        outliers_velocity = 0

        for i in range(test_repetition):
            truth, meas = sensor.measure_with_truth(boat.velocity)
            if np.abs(meas - truth) > np.abs(err_speed.error * err_speed.threshold):
                outliers_velocity += 1
            list_truth.append(truth)
            list_meas.append(meas)

        self.assertLess(outliers_velocity/test_repetition, 0.004)

class CompassTest:
    
    def test_outliers(self):
        err_angle = AbsoluteError(3*np.pi/180)
        sensor = Compass(err_angle)
        boat = Boat(100, Wing(15, Stepper(100, 0.05)), Rudder(Stepper(100, 0.2)), PID(1, 0.1, 0.1, 0.1), PID(1, 0.1, 0.1, 0.1))
        boat.heading = np.array([np.random.rand(), np.random.rand()])
        list_truth = []
        list_meas = []
        outliers_direction = 0
        
        for i in range(test_repetition):
            truth, meas = sensor.measure_with_truth(boat.heading)
            if np.abs(meas - truth) > err_angle.error:
                outliers_direction += 1
            list_truth.append(truth)
            list_meas.append(meas)

        self.assertLess(outliers_direction/test_repetition, 0.004)

class UWBSensorTest(unittest.TestCase):

    def test_outliers(self):
        err_distance = AbsoluteError(0.2)
        sensor = UWB(err_distance=err_distance)
        boat_actual = Boat(100, Wing(15, Stepper(100, 0.05)), Rudder(Stepper(100, 0.2)), PID(1, 0.1, 0.1, 0.1), PID(1, 0.1, 0.1, 0.1))
        boat_target = Boat(100, Wing(15, Stepper(100, 0.05)), Rudder(Stepper(100, 0.2)), PID(1, 0.1, 0.1, 0.1), PID(1, 0.1, 0.1, 0.1))
        boat_actual.position = np.array([np.random.rand()*100, np.random.rand()*100])
        boat_target.position = np.array([np.random.rand()*100, np.random.rand()*100])
        list_truth = []
        list_meas = []
        outliers_distance = 0
        
        for i in range(test_repetition):
            truth, meas = sensor.measure_with_truth(boat_actual.position, boat_target.position)
            if np.abs(meas - truth) > err_distance.error:
                outliers_distance += 1
            list_truth.append(truth)
            list_meas.append(meas)

        self.assertLess(outliers_distance/test_repetition, 0.004)

class GNSSSensorTest(unittest.TestCase):
    
        def test_outliers(self):
            err_position_x = AbsoluteError(1.5)
            err_position_y = AbsoluteError(1.5)
            sensor = GNSS(err_position_x=err_position_x, err_position_y=err_position_y)
            boat = Boat(100, Wing(15, Stepper(100, 0.05)), Rudder(Stepper(100, 0.2)), PID(1, 0.1, 0.1, 0.1), PID(1, 0.1, 0.1, 0.1))
            boat.position = np.array([np.random.rand()*100, np.random.rand()*100])
            list_truth = []
            list_meas = []
            outliers_position_x = 0
            outliers_position_y = 0
            
            for i in range(test_repetition):
                truth, meas = sensor.measure_with_truth(boat.position)
                if np.abs(meas[0] - truth[0]) > err_position_x.error:
                    outliers_position_x += 1
                if np.abs(meas[1] - truth[1]) > err_position_y.error:
                    outliers_position_y += 1
                list_truth.append(truth)
                list_meas.append(meas)
    
            self.assertLess(outliers_position_x/test_repetition, 0.004)
            self.assertLess(outliers_position_y/test_repetition, 0.004)


if __name__ == '__main__':
    unittest.main()