import unittest
import numpy as np
from actuator import Stepper
from entities import Wing, Rudder, Wind, Boat, compute_wind_force

class ToolTest(unittest.TestCase):
    
    # boat, wind and wing have the same heading, with angle = PI/2
    def test_compute_wind_force_max_y(self):
        wind = Wind(1.28)
        wind.velocity = np.array([0, 10])
        boat = Boat(100, Wing(10, Stepper(100, 1)), Rudder(Stepper(100, 1)))
        boat.heading = np.array([0, 1])
        boat.wing.stepper.set_angle(np.pi * 0.5)
        f = compute_wind_force(wind, boat)
        self.assertEqual(f[0], 0)
        self.assertEqual(f[1], 128)
    
    # boat, wind and wing have the same heading, with angle = 0
    def test_compute_wind_force_max_x(self):
        wind = Wind(1.28)
        wind.velocity = np.array([10, 0])
        boat = Boat(100, Wing(10, Stepper(100, 1)), Rudder(Stepper(100, 1)))
        boat.heading = np.array([1, 0])
        f = compute_wind_force(wind, boat)
        self.assertEqual(f[0], 128)
        self.assertEqual(f[1], 0)
    
    # boat, wind have the same heading, with angle = 0
    # wing has an opposite heading of PI
    # the result must be the same as above
    def test_compute_wind_force_max_x_opposite(self):
        wind = Wind(1.28)
        wind.velocity = np.array([10, 0])
        boat = Boat(100, Wing(10, Stepper(100, 1)), Rudder(Stepper(100, 1)))
        boat.heading = np.array([1, 0])
        boat.wing.stepper.set_angle(np.pi)
        f = compute_wind_force(wind, boat)
        self.assertEqual(f[0], 128)
        self.assertEqual(f[1], 0)
    
    # wind and wing have the same heading, with angle = 0
    # boat has a perpendicular heading with respect to the wind/wing, with angle = PI/2
    # the produced force must be 0
    def test_compute_wind_force_boat_perp(self):
        wind = Wind(1.28)
        wind.velocity = np.array([10, 0])
        boat = Boat(100, Wing(10, Stepper(100, 1)), Rudder(Stepper(100, 1)))
        boat.heading = np.array([0, 1])
        f = compute_wind_force(wind, boat)
        self.assertEqual(f[0], 0)
        self.assertEqual(f[1], 0)
    
    # wind and boat have the same heading, with angle = 0
    # wing has a perpendicular heading with respect to the wind/boat, with angle = PI/2
    # the produced force must be 0, as above
    def test_compute_wind_force_wing_perp(self):
        wind = Wind(1.28)
        wind.velocity = np.array([10, 0])
        boat = Boat(100, Wing(10, Stepper(100, 1)), Rudder(Stepper(100, 1)))
        boat.heading = np.array([1, 0])
        boat.wing.stepper.set_angle(np.pi * 0.5)
        f = compute_wind_force(wind, boat)
        self.assertAlmostEqual(f[0], 0)
        self.assertAlmostEqual(f[1], 0)
    
    def test_compute_wind_force_none(self):
        wind = Wind(1.28)
        wind.velocity = np.array([0, 0])
        boat = Boat(100, Wing(10, Stepper(100, 1)), Rudder(Stepper(100, 1)))
        boat.heading = np.array([1, 0])
        f = compute_wind_force(wind, boat)
        self.assertAlmostEqual(f[0], 0)
        self.assertAlmostEqual(f[1], 0)
    
if __name__ == '__main__':
    unittest.main()