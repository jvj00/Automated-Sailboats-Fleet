import unittest
import numpy as np

from tools.utils import check_intersection_circle_line, compute_angle, compute_angle_between, compute_turning_radius, is_angle_between, polar_to_cartesian, normalize

class TestComputeAngle(unittest.TestCase):

    def test_0(self):
        v1 = polar_to_cartesian(1, 0)
        angle = compute_angle(v1)
        self.assertEqual(angle, 0)
    
    def test_1(self):
        v1 = polar_to_cartesian(1, np.pi)
        angle = compute_angle(v1)
        self.assertEqual(angle, np.pi)
    
    def test_2(self):
        v1 = polar_to_cartesian(1, np.pi * 1.5)
        angle = compute_angle(v1)
        self.assertEqual(angle, np.pi * 1.5)
    
    def test_3(self):
        v1 = polar_to_cartesian(1, np.pi * 2)
        angle = compute_angle(v1)
        self.assertEqual(angle, 0)

class TestNormalize(unittest.TestCase):

    def test_0(self):
        v1 = polar_to_cartesian(1, 0)
        norm = normalize(v1)
        self.assertEqual(norm[0], 1)
        self.assertEqual(norm[1], 0)

    def test_1(self):
        v1 = np.zeros(2)
        norm = normalize(v1)
        self.assertEqual(norm, None)

    def test_2(self):
        v1 = polar_to_cartesian(1, -np.pi * 0.5)
        norm = normalize(v1)
        self.assertAlmostEqual(norm[0], 0)
        self.assertAlmostEqual(norm[1], -1)
    
    def test_3(self):
        v1 = polar_to_cartesian(8, np.pi * 0.25)
        norm = normalize(v1)
        self.assertAlmostEqual(norm[0], 0.70710678)
        self.assertAlmostEqual(norm[1], 0.70710678)

class TestComputeAngleBetween(unittest.TestCase):

    def test_parallel(self):
        v1 = polar_to_cartesian(4, 0)
        v2 = polar_to_cartesian(1, 0)
        angle = compute_angle_between(v1, v2)
        self.assertEqual(angle, 0)
    
    def test_opposite(self):
        v1 = polar_to_cartesian(4, np.pi * 0.5)
        v2 = polar_to_cartesian(1, np.pi * 1.5)
        angle = compute_angle_between(v1, v2)
        self.assertEqual(angle, np.pi)
        angle = compute_angle_between(v2, v1)
        self.assertEqual(angle, np.pi)
    
    def test_0(self):
        v1 = polar_to_cartesian(4, np.pi * 0.5)
        v2 = polar_to_cartesian(1, np.pi)
        angle = compute_angle_between(v2, v1)
        self.assertEqual(angle, np.pi * 0.5)
        angle = compute_angle_between(v1, v2)
        self.assertEqual(angle, np.pi * 1.5)

class TestComputeTurningRadius(unittest.TestCase):
    
    def test_clockwise(self):
        length = 7
        rudder_angle = np.pi * 0.2
        r = compute_turning_radius(length, rudder_angle)
        self.assertAlmostEqual(r, 9.634673443298215)
    
    def test_none_2(self):
        length = 7
        rudder_angle = np.pi * 1.5
        r = compute_turning_radius(length, rudder_angle)
        self.assertAlmostEqual(r, 0)
    
    def test_counterclockwise(self):
        length = 7
        rudder_angle = np.pi * 1.8
        r = compute_turning_radius(length, rudder_angle)
        self.assertAlmostEqual(r, -9.634673443298215)

class TestIsAngleBetween(unittest.TestCase):
    
    def test_0(self):
        min = 0
        max = np.pi
        angle = np.pi * 0.5
        between = is_angle_between(angle, min, max)
        self.assertTrue(between)
    
    def test_1(self):
        min = np.pi
        max = 0
        angle = np.pi * 0.5
        between = is_angle_between(angle, min, max)
        self.assertFalse(between)
    
    def test_2(self):
        min = np.pi * 1.5
        max = np.pi * 0.5
        angle = 0
        between = is_angle_between(angle, min, max)
        self.assertTrue(between)
    
    def test_3(self):
        min = np.pi * 1.5
        max = np.pi * 0.5
        angle = np.pi
        between = is_angle_between(angle, min, max)
        self.assertFalse(between)

class TestCheckIntersection(unittest.TestCase):

    def test_0(self):
        start = np.array([3.0, -4.0])
        end = start * 2
        center = np.array([3.0, -7.0])
        radius = 2
        interect = check_intersection_circle_line(center, radius, start, end)
        self.assertTrue(interect)
    
    def test_1(self):
        start = np.array([3.0, -4.0])
        end = start * 2
        center = np.array([3.0, -8.0])
        radius = 1
        interect = check_intersection_circle_line(center, radius, start, end)
        self.assertFalse(interect)
    
    def test_2(self):
        start = np.array([3.0, 4.0])
        end = start * 2
        center = np.array([3.0, 4.0])
        radius = 1
        interect = check_intersection_circle_line(center, radius, start, end)
        self.assertTrue(interect)

# class TestComputeTargetsFromMap(unittest.TestCase):

#     def test_0(self):
#         seabed = SeabedMap(-100, 100, -100, 100, 5)
#         groups = 2
#         targets = compute_targets_from_map(seabed, groups)
        
#         self.assertEqual(len(targets), 2)

#         self.assertEqual(len(targets[0]), 800)
        
#         self.assertAlmostEqual(targets[0][0][0], 2.5)
#         self.assertAlmostEqual(targets[0][0][1], 2.5)
#         self.assertAlmostEqual(targets[0][1][0], 7.5)
#         self.assertAlmostEqual(targets[0][1][1], 2.5)

#         self.assertEqual(len(targets[1]), 800)

#         self.assertAlmostEqual(targets[1][0][0], 2.5)
#         self.assertAlmostEqual(targets[1][0][1], 7.5)
#         self.assertAlmostEqual(targets[1][1][0], 7.5)
#         self.assertAlmostEqual(targets[1][1][1], 7.5)
        
#         # print(len(targets[0]))
#         # print(targets[0][1] - targets[0][0])
#         # print(targets[0][1], targets[0][0])
#         # print(len(targets[1]))

if __name__ == '__main__':
    unittest.main()