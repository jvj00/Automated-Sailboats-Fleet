import unittest
import numpy as np
from actuators.motor import Motor
from actuators.stepper import Stepper
from controllers.motor_controller import MotorController
from controllers.stepper_controller import StepperController
from entities.boat import Boat
from entities.rigid_body import RigidBody
from controllers.pid import PID
from surfaces.rudder import Rudder
from surfaces.wing import Wing
from tools.utils import cartesian_to_polar, normalize, polar_to_cartesian

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
        e.apply_acceleration_to_velocity(1)
        self.assertEqual(e.position[0], 0.5)
        self.assertEqual(e.position[1], -1.0)
        self.assertEqual(e.velocity[0], 1.0)
        self.assertEqual(e.velocity[1], -2.0)
        e.translate(1)
        e.apply_acceleration_to_velocity(1)
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
    
    def test_apply_friction_steady_still(self):
        e = RigidBody(10)
        e.friction_mu = 1
        gravity = 10
        dt = 1
        e.apply_friction(gravity, dt)
        e.translate(dt)
        self.assertEqual(e.position[0], 0.0)
        self.assertEqual(e.position[1], 0.0)
        self.assertEqual(e.velocity[0], 0.0)
        self.assertEqual(e.velocity[1], 0.0)
    
    def test_apply_friction_constant_velocity_1(self):
        e = RigidBody(10)
        e.friction_mu = 1
        e.velocity = np.array([1.0, -1.0])
        gravity = 10
        dt = 1
        e.apply_friction(gravity, dt)
        e.translate(dt)
        self.assertEqual(e.position[0], 0.0)
        self.assertEqual(e.position[1], 0.0)
        self.assertEqual(e.velocity[0], 0.0)
        self.assertEqual(e.velocity[1], 0.0)
    
    def test_apply_friction_constant_velocity_2(self):
        e = RigidBody(10)
        e.friction_mu = 0.001
        e.velocity = np.array([1.0, -1.0])
        gravity = 10
        dt = 1
        e.apply_friction(gravity, dt)
        e.translate(dt)
        self.assertEqual(e.position[0], 0.9)
        self.assertEqual(e.position[1], -0.9)
        self.assertEqual(e.velocity[0], 0.9)
        self.assertEqual(e.velocity[1], -0.9)

class TestBoat(unittest.TestCase):
    
    wing = Wing(10, StepperController(Stepper(100, 1), PID()))
    rudder = Rudder(StepperController(Stepper(100, 1), PID()))
    motor_controller = MotorController(Motor(100, 0.9, 2**10))
    boat = Boat(10, 10, None, wing, rudder, motor_controller)
    
    def test_rotate_aligned_rudder(self):
        self.boat.rudder.controller.set_angle(0.0)
        self.boat.velocity = np.array([2.0, 2.0])
        self.boat.rotate(1)
        self.assertEqual(self.boat.angular_speed, 0)
    
    def test_rotate_1(self):
        self.boat.rudder.controller.set_angle(np.pi * 0.25)
        self.boat.velocity = np.array([2.0, 2.0])
        self.boat.rotate(1)
        self.assertAlmostEqual(self.boat.rudder.controller.get_angle(), 0.753982236)
        self.assertAlmostEqual(self.boat.angular_speed, 0.18781250116349849)
        self.boat.rudder.controller.set_angle(-np.pi * 0.25)
        self.boat.rotate(1)
        self.assertAlmostEqual(self.boat.rudder.controller.get_angle(), 5.4663712)
        self.assertAlmostEqual(self.boat.angular_speed, -0.24899840390785477)

if __name__ == '__main__':
    unittest.main()