import unittest
import numpy as np
from actuator import Stepper

class StepperTest(unittest.TestCase):

    def test1(self):
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
    
    def test2(self):
        stepper = Stepper(20, 5)
        stepper.move(0.2)
        self.assertEqual(stepper.get_steps(), 0)
        self.assertAlmostEqual(stepper.get_angle(), 0)
        stepper.move(0.3)
        self.assertEqual(stepper.get_steps(), 10)
        self.assertAlmostEqual(stepper.get_angle(), np.pi)

if __name__ == '__main__':
    unittest.main()