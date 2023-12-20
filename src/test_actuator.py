import unittest
import numpy as np
from actuator import Stepper, StepperDirection, StepperController

class StepperTest(unittest.TestCase):

    def test_1(self):
        stepper = Stepper(100, 1)
        stepper.move(1)
        self.assertEqual(stepper.get_steps(), 0)
        self.assertAlmostEqual(stepper.get_angle(), 0)
        stepper.move(0.5)
        self.assertEqual(stepper.get_steps(), 50)
        self.assertAlmostEqual(stepper.get_angle(), np.pi)
        stepper.move(0.6)
        self.assertEqual(stepper.get_steps(), 10)
        self.assertAlmostEqual(stepper.get_angle(), np.pi * 0.2)
    
    def test_2(self):
        stepper = Stepper(20, 5)
        stepper.move(0.2)
        self.assertEqual(stepper.get_steps(), 0)
        self.assertAlmostEqual(stepper.get_angle(), 0)
        stepper.move(0.3)
        self.assertEqual(stepper.get_steps(), 10)
        self.assertAlmostEqual(stepper.get_angle(), np.pi)

    def test_3(self):
        stepper = Stepper(100, 1)
        stepper.move(0.5)
        self.assertEqual(stepper.get_steps(), 50)
        self.assertAlmostEqual(stepper.get_angle(), np.pi)
        stepper.direction = StepperDirection.CounterClockwise
        stepper.move(0.1)
        self.assertEqual(stepper.get_steps(), 40)
        self.assertAlmostEqual(stepper.get_angle(), np.pi * 0.8)

class StepperControllerTest(unittest.TestCase):

    def test_1(self):
        stepper = Stepper(100, 0.5)
        controller = StepperController(stepper)
        controller.move(0.1)
        controller.move(0.2)
        controller.move(0.2)
        self.assertEqual(controller.get_angle(), 0)
    
    def test_2(self):
        stepper = Stepper(100, 0.5)
        controller = StepperController(stepper)
        controller.set_target(np.pi * 0.5)
        controller.move(0.1)
        self.assertAlmostEqual(controller.get_angle(), 0.31415926)
        controller.move(0.1)
        self.assertAlmostEqual(controller.get_angle(), 0.62831853)
        controller.move(0.2)
        self.assertAlmostEqual(controller.get_angle(), 1.25663706)
        controller.move(0.1)
        controller.move(0.1)
        controller.move(0.1)
        self.assertAlmostEqual(controller.get_angle(), np.pi * 0.5)
        controller.move(0.1)
        controller.move(0.1)
        self.assertAlmostEqual(controller.get_angle(), np.pi * 0.5)
        controller.move(0.1)
        self.assertAlmostEqual(controller.get_angle(), np.pi * 0.5)
        controller.move(0.1)
        controller.move(1.2)
        self.assertAlmostEqual(controller.get_angle(), np.pi * 0.5)

if __name__ == '__main__':
    unittest.main()