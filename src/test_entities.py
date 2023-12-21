import unittest
import numpy as np
from actuator import Stepper
from entities import Wing, Rudder, Wind, Boat, compute_wind_force
from pid import PID
from utils import normalize

class EntitiesTest(unittest.TestCase):
    wind = Wind(1.28)
    wing = Wing(10, Stepper(100, 1))
    rudder = Wing(10, Stepper(100, 1))
    rudder_pid = PID(1, 0.1, 0.1, limits=(-np.pi * 0.25, np.pi * 0.25))
    wing_pid = PID(1, 0.1, 0.1, limits=(0, np.pi * 2))
    mass = 100
    boat = Boat(mass, wing, rudder, rudder_pid, wing_pid)
    
    # boat, wind and wing have the same heading, with angle = PI/2
    def test_compute_wind_force_max_y(self):
        self.wind.velocity = np.array([0, 10])
        self.boat.heading = np.array([0, 1])
        self.boat.wing.stepper.set_angle(np.pi * 0.5)
        f = compute_wind_force(self.wind, self.boat)
        self.assertEqual(f[0], 0)
        self.assertEqual(f[1], 128)
    
    # boat, wind and wing have the same heading, with angle = 0
    def test_compute_wind_force_max_x(self):
        self.wind.velocity = np.array([10, 0])
        self.boat.heading = np.array([1, 0])
        f = compute_wind_force(self.wind, self.boat)
        self.assertEqual(f[0], 128)
        self.assertEqual(f[1], 0)
    
    # boat, wind have the same heading, with angle = 0
    # wing has an opposite heading of PI
    # the result must be the same as above
    def test_compute_wind_force_max_x_opposite(self):
        self.wind.velocity = np.array([10, 0])
        self.boat.heading = np.array([1, 0])
        self.boat.wing.stepper.set_angle(np.pi)
        f = compute_wind_force(self.wind, self.boat)
        self.assertEqual(f[0], 128)
        self.assertEqual(f[1], 0)
    
    # wind and wing have the same heading, with angle = 0
    # boat has a perpendicular heading with respect to the wind/wing, with angle = PI/2
    # the produced force must be 0
    def test_compute_wind_force_boat_perp(self):
        self.wind.velocity = np.array([10, 0])
        self.boat.heading = np.array([0, 1])
        f = compute_wind_force(self.wind, self.boat)
        self.assertEqual(f[0], 0)
        self.assertEqual(f[1], 0)
    
    # wind and boat have the same heading, with angle = 0
    # wing has a perpendicular heading with respect to the wind/boat, with angle = PI/2
    # the produced force must be 0, as above
    def test_compute_wind_force_wing_perp(self):
        self.wind.velocity = np.array([10, 0])
        self.boat.heading = np.array([1, 0])
        self.boat.wing.stepper.set_angle(np.pi * 0.5)
        f = compute_wind_force(self.wind, self.boat)
        self.assertAlmostEqual(f[0], 0)
        self.assertAlmostEqual(f[1], 0)
    
    def test_compute_wind_force_boat_steady_no_wind(self):
        self.wind.velocity = np.zeros(2)
        self.boat.heading = np.array([1, 0])
        f = compute_wind_force(self.wind, self.boat)
        self.assertAlmostEqual(f[0], 0)
        self.assertAlmostEqual(f[1], 0)
    
    def test_compute_wind_force_boat_moving_no_wind(self):
        self.wind.velocity = np.zeros(2)
        self.boat.heading = np.array([1, 0])
        self.boat.velocity = normalize(np.array([5.0, 2.0]))
        f = compute_wind_force(self.wind, self.boat)
        self.assertAlmostEqual(f[0], 0)
        self.assertAlmostEqual(f[1], 0)
    
if __name__ == '__main__':
    unittest.main()