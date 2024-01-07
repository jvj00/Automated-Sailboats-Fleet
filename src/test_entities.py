import unittest
import numpy as np
from actuator import Stepper, StepperController
from entities import RigidBody, Wing, Rudder, Wind, Boat, compute_wind_force
from pid import PID
from utils import cartesian_to_polar, normalize, polar_to_cartesian

class TestRigidBody(unittest.TestCase):

    def test_translate_steady_still(self):
        e = RigidBody(10)
        e.translate(1)
        self.assertEqual(e.position[0], 0)
        self.assertEqual(e.position[1], 0)
    
    def test_translate_constant_velocity(self):
        e = RigidBody(10)
        e.velocity = np.array([-1.0, 2.0])
        e.translate(1)
        self.assertEqual(e.position[0], -1.0)
        self.assertEqual(e.position[1], 2.0)
        self.assertEqual(e.velocity[0], -1.0)
        self.assertEqual(e.velocity[1], 2.0)
        e.translate(1)
        self.assertEqual(e.position[0], -2.0)
        self.assertEqual(e.position[1], 4.0)
        self.assertEqual(e.velocity[0], -1.0)
        self.assertEqual(e.velocity[1], 2.0)

    
    def test_translate_constant_acceleration(self):
        e = RigidBody(10)
        e.acceleration = np.array([1.0, -2.0])
        e.translate(1)
        self.assertEqual(e.position[0], 0.5)
        self.assertEqual(e.position[1], -1.0)
        self.assertEqual(e.velocity[0], 1.0)
        self.assertEqual(e.velocity[1], -2.0)
        e.translate(1)
        self.assertEqual(e.position[0], 2.0)
        self.assertEqual(e.position[1], -4.0)
        self.assertEqual(e.velocity[0], 2.0)
        self.assertEqual(e.velocity[1], -4.0)

    def test_rotate_steady_still(self):
        e = RigidBody(10)
        e.rotate(1)
        _, curr_angle = cartesian_to_polar(e.heading)
        self.assertEqual(curr_angle, 0)
    
    def test_rotate_constant_speed(self):
        e = RigidBody(10)
        e.angular_speed = np.pi * 0.5
        e.rotate(1)
        _, curr_angle = cartesian_to_polar(e.heading)
        self.assertEqual(e.position[0], 0.0)
        self.assertEqual(e.position[1], 0.0)
        self.assertEqual(e.velocity[0], 0.0)
        self.assertEqual(e.velocity[1], 0.0)
        self.assertEqual(curr_angle, np.pi * 0.5)
        e.rotate(1)
        _, curr_angle = cartesian_to_polar(e.heading)
        self.assertEqual(e.position[0], 0.0)
        self.assertEqual(e.position[1], 0.0)
        self.assertEqual(e.velocity[0], 0.0)
        self.assertEqual(e.velocity[1], 0.0)
        self.assertEqual(curr_angle, np.pi)
    
    def test_rotate_constant_acceleration(self):
        e = RigidBody(10)
        e.angular_acceleration = np.pi * 0.5
        e.rotate(1)
        _, curr_angle = cartesian_to_polar(e.heading)
        self.assertEqual(e.position[0], 0.0)
        self.assertEqual(e.position[1], 0.0)
        self.assertEqual(e.velocity[0], 0.0)
        self.assertEqual(e.velocity[1], 0.0)
        self.assertEqual(e.angular_speed, np.pi * 0.5)
        self.assertEqual(curr_angle, np.pi * 0.25)
        e.rotate(1)
        _, curr_angle = cartesian_to_polar(e.heading)
        self.assertEqual(e.position[0], 0.0)
        self.assertEqual(e.position[1], 0.0)
        self.assertEqual(e.velocity[0], 0.0)
        self.assertEqual(e.velocity[1], 0.0)
        self.assertEqual(e.angular_speed, np.pi)
        self.assertEqual(curr_angle, np.pi)

# class EntitiesTest(unittest.TestCase):
#     wind_velocity = np.zeros(2)
#     wind_density = 1.281
#     boat_velocity = np.zeros(2)
#     boat_heading = polar_to_cartesian(1, np.pi * 0.5)
#     wing_heading = polar_to_cartesian(1, np.pi * 0.5)
#     wing_area = 10
#     drag = 0.5
    
    # boat, wind and wing have the same heading, with angle = PI/2
    # def test_compute_wind_force_max_y(self):
    #     self.wind_velocity = np.array([10.0, 0.0])
    #     f = compute_wind_force(self.wind_velocity,
    #                            self.wind_density,
    #                            self.boat_velocity,
    #                            self.boat_heading,
    #                            self.wing_heading,
    #                            self.wing_area,
    #                            self.drag
    #     )
    #     self.assertAlmostEqual(f[0], 0)
    #     self.assertAlmostEqual(f[1], 128)
    
    # # boat, wind and wing have the same heading, with angle = 0
    # def test_compute_wind_force_max_x(self):
    #     self.wind.velocity = np.array([10, 0])
    #     self.boat.heading = np.array([1, 0])
    #     f = compute_wind_force(self.wind, self.boat)
    #     self.assertEqual(f[0], 128)
    #     self.assertEqual(f[1], 0)
    
    # # boat, wind have the same heading, with angle = 0
    # # wing has an opposite heading of PI
    # # the result must be the same as above
    # def test_compute_wind_force_max_x_opposite(self):
    #     self.wind.velocity = np.array([10, 0])
    #     self.boat.heading = np.array([1, 0])
    #     self.boat.wing.stepper.set_angle(np.pi)
    #     f = compute_wind_force(self.wind, self.boat)
    #     self.assertEqual(f[0], 128)
    #     self.assertEqual(f[1], 0)
    
    # # wind and wing have the same heading, with angle = 0
    # # boat has a perpendicular heading with respect to the wind/wing, with angle = PI/2
    # # the produced force must be 0
    # def test_compute_wind_force_boat_perp(self):
    #     self.wind.velocity = np.array([10, 0])
    #     self.boat.heading = np.array([0, 1])
    #     f = compute_wind_force(self.wind, self.boat)
    #     self.assertEqual(f[0], 0)
    #     self.assertEqual(f[1], 0)
    
    # # wind and boat have the same heading, with angle = 0
    # # wing has a perpendicular heading with respect to the wind/boat, with angle = PI/2
    # # the produced force must be 0, as above
    # def test_compute_wind_force_wing_perp(self):
    #     self.wind.velocity = np.array([10, 0])
    #     self.boat.heading = np.array([1, 0])
    #     self.boat.wing.stepper.set_angle(np.pi * 0.5)
    #     f = compute_wind_force(self.wind, self.boat)
    #     self.assertAlmostEqual(f[0], 0)
    #     self.assertAlmostEqual(f[1], 0)
    
    # def test_compute_wind_force_boat_steady_no_wind(self):
    #     self.wind.velocity = np.zeros(2)
    #     self.boat.heading = np.array([1, 0])
    #     f = compute_wind_force(self.wind, self.boat)
    #     self.assertAlmostEqual(f[0], 0)
    #     self.assertAlmostEqual(f[1], 0)
    
    # def test_compute_wind_force_boat_moving_no_wind(self):
    #     self.wind.velocity = np.zeros(2)
    #     self.boat.heading = np.array([1, 0])
    #     self.boat.velocity = normalize(np.array([5.0, 2.0]))
    #     f = compute_wind_force(self.wind, self.boat)
    #     self.assertAlmostEqual(f[0], 0)
    #     self.assertAlmostEqual(f[1], 0)
    
if __name__ == '__main__':
    unittest.main()