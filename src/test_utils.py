import unittest
import numpy as np

from utils import compute_angle, compute_angle_between, polar_to_cartesian, normalize

class TestComputeAngle(unittest.TestCase):

    def test_compute_angle_0(self):
        v1 = polar_to_cartesian(1, 0)
        angle = compute_angle(v1)
        self.assertEqual(angle, 0)
    
    def test_compute_angle_1(self):
        v1 = polar_to_cartesian(1, np.pi)
        angle = compute_angle(v1)
        self.assertEqual(angle, np.pi)
    
    def test_compute_angle_2(self):
        v1 = polar_to_cartesian(1, np.pi * 1.5)
        angle = compute_angle(v1)
        self.assertEqual(angle, np.pi * 1.5)
    
    def test_compute_angle_3(self):
        v1 = polar_to_cartesian(1, np.pi * 2)
        angle = compute_angle(v1)
        self.assertEqual(angle, 0)

class TestNormalize(unittest.TestCase):

    def test_normalize_0(self):
        v1 = polar_to_cartesian(1, 0)
        norm = normalize(v1)
        self.assertEqual(norm[0], 1)
        self.assertEqual(norm[1], 0)

    def test_normalize_1(self):
        v1 = np.zeros(2)
        norm = normalize(v1)
        self.assertEqual(norm, None)

    def test_normalize_2(self):
        v1 = polar_to_cartesian(1, -np.pi * 0.5)
        norm = normalize(v1)
        self.assertAlmostEqual(norm[0], 0)
        self.assertAlmostEqual(norm[1], -1)
    
    def test_normalize_3(self):
        v1 = polar_to_cartesian(8, np.pi * 0.25)
        norm = normalize(v1)
        self.assertAlmostEqual(norm[0], 0.70710678)
        self.assertAlmostEqual(norm[1], 0.70710678)

class TestComputeAngleBetween(unittest.TestCase):

    def test_compute_angle_between_parallel(self):
        v1 = polar_to_cartesian(4, 0)
        v2 = polar_to_cartesian(1, 0)
        angle = compute_angle_between(v1, v2)
        self.assertEqual(angle, 0)
    
    def test_compute_angle_between_parallel_opposite(self):
        v1 = polar_to_cartesian(4, np.pi * 0.5)
        v2 = polar_to_cartesian(1, np.pi * 1.5)
        angle = compute_angle_between(v1, v2)
        self.assertEqual(angle, np.pi)
        angle = compute_angle_between(v2, v1)
        self.assertEqual(angle, np.pi)
    
    def test_compute_angle_between_0(self):
        v1 = polar_to_cartesian(4, np.pi * 0.5)
        v2 = polar_to_cartesian(1, np.pi)
        angle = compute_angle_between(v2, v1)
        self.assertEqual(angle, np.pi * 0.5)
        angle = compute_angle_between(v1, v2)
        self.assertEqual(angle, np.pi * 1.5)

if __name__ == '__main__':
    unittest.main()