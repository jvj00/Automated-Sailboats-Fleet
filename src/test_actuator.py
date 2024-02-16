import unittest
import numpy as np
from tools.logger import Logger
from controllers.pid import PID

# class StepperControllerTest(unittest.TestCase):

    # def test_clockwise(self):
    #     stepper = Stepper(100, 1)
    #     pid = PID(0.1, 0.1, 0.1)
    #     controller = StepperController(stepper, pid)
    #     controller.set_target(0.5)
    #     controller.move(0.1)
    #     self.assertEqual(controller.steps, 10.0)
    #     self.assertAlmostEqual(controller.get_angle(), 0.6283185307)
    #     controller.move(0.1)
    #     self.assertEqual(controller.steps, 10.0)
    #     # self.assertAlmostEqual(controller.get_angle(), 0)
    #     controller.move(0.1)
    #     # self.assertEqual(controller.steps, 0)
    #     self.assertAlmostEqual(controller.get_angle(), 0)
    
    # def test_counterclockwise(self):
    #     stepper = Stepper(20, 5)
    #     stepper.direction = StepperDirection.CounterClockwise
    #     stepper.move(0.2)
    #     self.assertEqual(stepper.get_steps(), 0)
    #     self.assertAlmostEqual(stepper.get_angle(), 0)
    #     stepper.move(0.3)
    #     self.assertEqual(stepper.get_steps(), 10)
    #     self.assertAlmostEqual(stepper.get_angle(), np.pi)

    # def test_mixed(self):
    #     stepper = Stepper(100, 1)
    #     stepper.move(0.5)
    #     self.assertEqual(stepper.get_steps(), 50)
    #     self.assertAlmostEqual(stepper.get_angle(), np.pi)
    #     stepper.direction = StepperDirection.CounterClockwise
    #     stepper.move(0.1)
    #     self.assertEqual(stepper.get_steps(), 40)
    #     self.assertAlmostEqual(stepper.get_angle(), np.pi * 0.8)

# class StepperControllerTest(unittest.TestCase):

#     def test_steady(self):
#         stepper = Stepper(100, 0.5)
#         controller = StepperController(stepper)
#         controller.move(0.1)
#         controller.move(0.2)
#         controller.move(0.2)
#         self.assertEqual(controller.get_angle(), 0)
    
#     def test_moving(self):
#         stepper = Stepper(100, 0.5)
#         controller = StepperController(stepper)
#         controller.set_target(np.pi * 0.5)
#         controller.move(0.1)
#         self.assertAlmostEqual(controller.get_angle(), 0.31415926)
#         controller.move(0.1)
#         self.assertAlmostEqual(controller.get_angle(), 0.62831853)
#         controller.move(0.2)
#         self.assertAlmostEqual(controller.get_angle(), 1.25663706)
#         controller.move(0.1)
#         controller.move(0.1)
#         controller.move(0.1)
#         self.assertAlmostEqual(controller.get_angle(), np.pi * 0.5)
#         controller.move(0.1)
#         controller.move(0.1)
#         self.assertAlmostEqual(controller.get_angle(), np.pi * 0.5)
#         controller.move(0.1)
#         self.assertAlmostEqual(controller.get_angle(), np.pi * 0.5)
#         controller.move(0.1)
#         controller.move(1.2)
#         self.assertAlmostEqual(controller.get_angle(), np.pi * 0.5)
    
#     def test_pid(self):
#         stepper = Stepper(100, 1)
#         setpoint = 0.5
#         dt = 0.1
#         simulation_period = 10
#         for kp in np.arange(0, 5, 0.1):
#             for ki in np.arange(0, 5, 0.1):
#                 for kd in np.arange(0, 5, 0.1):
#                     pid = PID(kp, ki, kd, setpoint)
#                     controller = StepperController(stepper, pid)
#                     Logger.debug(f'KP: {kp} KI: {ki} KD: {kd}')
#                     for time_elapsed in np.arange(0, simulation_period, dt):
#                         controller.move(dt)
#                         # if time_elapsed == (simulation_period - dt):
#                             # Logger.debug(f'Error: {setpoint - controller.get_angle()}')
#                             # Logger.debug(f'Speed: {controller.speed}')
#     def test_pid(self):
#         stepper = Stepper(100, 1)
#         pid = PID(0.5, 0.1, 0.1, 1.4)
#         controller = StepperController(stepper, pid)
#         dt = 0.1
#         simulation_period = 10
#         for time_elapsed in np.arange(0, simulation_period, dt):
#             controller.move(dt)
#             Logger.debug(f'Angle: {controller.get_angle()}, Speed: {controller.speed}, Steps: {controller.steps}')

# if __name__ == '__main__':
#     unittest.main()
    