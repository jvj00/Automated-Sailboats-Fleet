import unittest
import numpy as np
from actuator import Rudder, Stepper

from entities import Boat, Wind, Wing, compute_wind_force
from utils import normalize

class ToolTest(unittest.TestCase):

    def test_compute_wind_force_1(self):
        wind = Wind(1.28)
        wind.velocity = np.array([0, 10])
        boat = Boat(100, Wing(10), Rudder(Stepper(100, 5)))
        boat.heading = np.array([0, 1])
        boat.wing.heading = np.array([0, 1])
        f = compute_wind_force(wind, boat)
        self.assertEqual(f[0], 0)
        self.assertEqual(f[1], 128)
    
    def test_compute_wind_force_2(self):
        wind = Wind(1.28)
        wind.velocity = np.array([10, 0])
        boat = Boat(100, Wing(10), Rudder(Stepper(100, 5)))
        boat.heading = np.array([1, 0])
        boat.wing.heading = np.array([0, 1])
        f = compute_wind_force(wind, boat)
        self.assertEqual(f[0], 0)
        self.assertEqual(f[1], 0)
    
    def test_compute_wind_force_2(self):
        wind = Wind(1.28)
        wind.velocity = np.array([10, 0])
        boat = Boat(100, Wing(10), Rudder(Stepper(100, 5)))
        boat.heading = normalize(np.array([0.5, 0.5]))
        boat.wing.heading = np.array([1, 0])
        f = compute_wind_force(wind, boat)
        self.assertAlmostEqual(f[0], 64)
        self.assertAlmostEqual(f[1], 64)

if __name__ == '__main__':
    unittest.main()