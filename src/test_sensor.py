import unittest
from actuators.motor import Motor
from actuators.stepper import Stepper
from controllers.motor_controller import MotorController
from controllers.stepper_controller import StepperController
from estimation_algs.ekf import EKF
from entities.boat import Boat
from entities.wind import Wind
from entities.environment import SeabedMap
from errors.absolute_error import AbsoluteError
from errors.mixed_error import MixedError
from errors.relative_error import RelativeError
from controllers.pid import PID
from sensors.anemometer import Anemometer
from sensors.compass import Compass
from sensors.gnss import GNSS
from sensors.sonar import Sonar
from sensors.speedometer import Speedometer
from components.rudder import Rudder
from components.wing import Wing
import numpy as np

from tools.utils import polar_to_cartesian

test_repetition = 10000

rudder_controller = StepperController(Stepper(100, 0.3), PID(0.5, 0, 0), np.pi * 0.25)
wing_controller = StepperController(Stepper(100, 0.3), PID(0.5, 0, 0))
motor_controller = MotorController(Motor(200, 0.85, 1024))
seabed = SeabedMap(0,0,0,0)
boat = Boat(50, 10, Wing(10, wing_controller), Rudder(rudder_controller), motor_controller, seabed)
wind = Wind(1.291)
class AnemometerTest(unittest.TestCase):
    sensor = Anemometer(RelativeError(0.05), AbsoluteError(np.pi/180))
    
    def test_outliers(self):
        wind.velocity = np.array([np.random.uniform(0, 20), np.random.uniform(0, 20)])
        boat.velocity = np.array([np.random.uniform(0, 20), np.random.uniform(0, 20)])
        list_truth = []
        list_meas = []
        outliers_direction = 0
        outliers_velocity = 0
        
        for i in range(test_repetition):
            truth, meas = self.sensor.measure_with_truth(wind.velocity, boat.velocity, boat.heading)
            if np.abs(meas[0] - truth[0]) > np.abs(self.sensor.err_speed.error * truth[0]):
                outliers_velocity += 1
            if np.abs(meas[1] - truth[1]) > self.sensor.err_angle.error:
                outliers_direction += 1
            list_truth.append(truth)
            list_meas.append(meas)
        
        max_error = 0.002
        # 100 - 99.7% due to 3 sigma rules
        limit = 0.003
        self.assertLess(outliers_velocity/test_repetition, limit + max_error)
        self.assertLess(outliers_direction/test_repetition, limit + max_error)
    
    def test_sensor_boat_wind_aligned_same_direction(self):
        wind.velocity = np.array([10.0, 0.0])
        boat.velocity = np.array([0.0, 0.0])
        boat.heading = polar_to_cartesian(1, 0)
        truth, meas = self.sensor.measure_with_truth(wind.velocity, boat.velocity, boat.heading)
        self.assertEqual(truth[0], 10.0)
        self.assertEqual(truth[1], 0)

    def test_sensor_boat_wind_perpendicular_1(self):
        wind.velocity = np.array([10.0, 0.0])
        boat.velocity = np.array([5.0, 0.0])
        boat.heading = polar_to_cartesian(1, np.pi * 0.5)
        truth, meas = self.sensor.measure_with_truth(wind.velocity, boat.velocity, boat.heading)
        self.assertEqual(truth[0], 5.0)
        self.assertEqual(truth[1], np.pi * 1.5)
    
    def test_sensor_boat_wind_aligned_opposite_direction(self):
        wind.velocity = np.array([-10.0, 0.0])
        boat.velocity = np.array([5.0, 0.0])
        boat.heading = polar_to_cartesian(1, 0)
        truth, meas = self.sensor.measure_with_truth(wind.velocity, boat.velocity, boat.heading)
        self.assertEqual(truth[0], 15.0)
        self.assertEqual(truth[1], np.pi)
    
    def test_sensor_boat_wind_perpendicular_2(self):
        wind.velocity = np.array([-10.0, 0.0])
        boat.velocity = np.array([5.0, 0.0])
        boat.heading = polar_to_cartesian(1, np.pi * 1.5)
        truth, meas = self.sensor.measure_with_truth(wind.velocity, boat.velocity, boat.heading)
        self.assertEqual(truth[0], 15.0)
        self.assertEqual(truth[1], np.pi * 1.5)

class SpeedometerTest(unittest.TestCase):

    sensor = Speedometer(MixedError(0.01, 5))

    def test_outliers_high_velocities(self):
        x = np.random.uniform(4, 15)
        y = np.random.uniform(4, 15)
        boat.velocity = np.array([x, y])
        list_truth = []
        list_meas = []
        outliers_velocity = 0

        for i in range(test_repetition):
            truth, meas = self.sensor.measure_with_truth(boat.velocity, boat.heading)
            if np.abs(meas - truth) > np.abs(self.sensor.err_speed.error * truth):
                outliers_velocity += 1
            list_truth.append(truth)
            list_meas.append(meas)

        # FIXME values are far from 0.004 by nearly 1
        self.assertLess(outliers_velocity/test_repetition, 0.004)

    def test_outliers_low_velocities(self):
        x = np.random.rand() * 5
        y = np.random.rand() * np.sqrt(25 - x**2)
        boat.velocity = np.array([x, y])
        list_truth = []
        list_meas = []
        outliers_velocity = 0

        for i in range(test_repetition):
            truth, meas = self.sensor.measure_with_truth(boat.velocity, boat.heading)
            if np.abs(meas - truth) > np.abs(self.sensor.err_speed.error * self.sensor.err_speed.threshold):
                outliers_velocity += 1
            list_truth.append(truth)
            list_meas.append(meas)
        
        # FIXME values are far from 0.004 by nearly 1
        self.assertLess(outliers_velocity/test_repetition, 0.005)

class CompassTest(unittest.TestCase):
    
    sensor = Compass(AbsoluteError(3 * np.pi/180))

    def test_outliers(self):
        boat.heading = np.array([np.random.rand(), np.random.rand()])
        list_truth = []
        list_meas = []
        outliers_direction = 0
        
        for i in range(test_repetition):
            truth, meas = self.sensor.measure_with_truth(boat.heading)
            if np.abs(meas - truth) > self.sensor.err_angle.error:
                outliers_direction += 1
            list_truth.append(truth)
            list_meas.append(meas)
        
        # FIXME values are far from 0.004 by nearly 1
        self.assertLess(outliers_direction/test_repetition, 0.004)

class GNSSSensorTest(unittest.TestCase):
        
        sensor = GNSS(AbsoluteError(1.5), AbsoluteError(1.5))
        
        def test_outliers(self):
            boat.position = np.array([np.random.rand()*100, np.random.rand()*100])
            list_truth = []
            list_meas = []
            outliers_position_x = 0
            outliers_position_y = 0
            
            for i in range(test_repetition):
                truth, meas = self.sensor.measure_with_truth(boat.position)
                if np.abs(meas[0] - truth[0]) > self.sensor.err_position_x.error:
                    outliers_position_x += 1
                if np.abs(meas[1] - truth[1]) > self.sensor.err_position_y.error:
                    outliers_position_y += 1
                list_truth.append(truth)
                list_meas.append(meas)
    
            self.assertLess(outliers_position_x/test_repetition, 0.004)
            self.assertLess(outliers_position_y/test_repetition, 0.004)


if __name__ == '__main__':
    unittest.main()