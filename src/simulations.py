# seadbed initialization
import time
from typing import Any
import unittest
from actuators.motor import Motor
from actuators.stepper import Stepper
from controllers.motor_controller import MotorController
from controllers.stepper_controller import StepperController
from entities.boat import Boat
from entities.wind import Wind
from entities.world import World
from errors.absolute_error import AbsoluteError
from errors.mixed_error import MixedError
from errors.relative_error import RelativeError
from sensors.anemometer import Anemometer
from sensors.compass import Compass
from sensors.gnss import GNSS
from sensors.speedometer import Speedometer
from surfaces.rudder import Rudder
from surfaces.wing import Wing
from tools.disegnino import Drawer
from environment import SeabedMap
import numpy as np
import matplotlib.pyplot as plt
import copy



from controllers.pid import PID
from tools.utils import check_intersection_circle_circle, compute_angle_between, compute_distance, mod2pi, modpi, polar_to_cartesian

from tools.logger import Logger

def check_collisions(boats: list[Boat]) -> bool:
    for a in boats:
        for b in boats:
            if str(a.uuid) != str(b.uuid):
                if check_intersection_circle_circle(a.position, a.length * 0.5, b.position, b.length * 0.5):
                    return True
    return False

def simulate(
        boats: list[Boat],
        world: World,
        targets: list[list[np.ndarray]],
        dt: float,
        simulation_time: float,
        simulated_data: bool = False,
        measured_data: bool = False,
        motor_only: bool = False,
    ):
    
    # keeps, for each boat, its state at each instant dt
    states: list[list[Boat]] = [[] for _ in range(len(boats))]

    # keeps, for each boat, the index of its current target
    current_targets_idx = [0 for _ in range(len(targets))]

    # set the first target of each boat
    for i in range(len(boats)): 
        target = targets[i][0]
        boats[i].set_target(target)
    
    for time_elapsed in np.arange(0, simulation_time, dt):

        # check if every boat has covered every target
        completed = True

        for i in range(len(boats)):

            states[i].append(copy.deepcopy(boats[i]))
    
            completed &= (current_targets_idx[i] == len(targets[i]))

            if current_targets_idx[i] == len(targets[i]):
                    continue

            # check if the boat has reached the target        
            if check_intersection_circle_circle(boats[i].position, boats[i].length * 0.5, targets[i][current_targets_idx[i]], boats[i].length * 0.5):
                current_targets_idx[i] += 1
                if current_targets_idx[i] == len(targets[i]):
                    continue
                boats[i].set_target(targets[i][current_targets_idx[i]])

            boats[i].follow_target(world.wind, dt, simulated_data, measured_data, motor_only)

        world.update(boats, dt)

        if completed:
            break

        # input()
    
    completed = True

    for i in range(len(boats)):
        completed &= (current_targets_idx[i] == len(targets[i]))
    
    return completed, time_elapsed, states


class TestMotorOnly(unittest.TestCase):

    def test_straight_line_simulated(self):
        ## wind initialization
        wind = Wind(1.291)
        world = World(9.81, wind)

        rudder_controller = StepperController(Stepper(100, 0.3), PID(0.5, 0, 0), np.pi * 0.15)
        motor_controller = MotorController(Motor(200, 0.85, 1024))

        ## boat initialization
        boat = Boat(40, 5, None, None, Rudder(rudder_controller), motor_controller)

        boat.position = np.array([0.0, 0.0])
        boat.velocity = np.array([0.0, 0.0])
        boat.heading = polar_to_cartesian(1, 0)

        targets = [np.array([50, 0]), np.array([100, 0]), np.array([150, 0]), np.array([200, 0])]
        dt = 0.1
        simulation_time = 100

        completed, time_elapsed, routes = simulate([boat], world, [targets], dt, simulation_time, simulated_data=True, motor_only=True)

        self.assertTrue(completed)
        self.assertAlmostEqual(time_elapsed, 70.3)
    
    def test_straight_line_opposite_simulated(self):
        ## wind initialization
        wind = Wind(1.291)
        world = World(9.81, wind)

        rudder_controller = StepperController(Stepper(100, 0.3), PID(0.5, 0, 0), np.pi * 0.15)
        motor_controller = MotorController(Motor(200, 0.85, 1024))

        ## boat initialization
        boat = Boat(40, 5, None, None, Rudder(rudder_controller), motor_controller)

        boat.position = np.array([0.0, 0.0])
        boat.velocity = np.array([0.0, 0.0])
        boat.heading = polar_to_cartesian(1, 0)

        targets = [np.array([-50, 0]), np.array([-100, 0]), np.array([-150, 0]), np.array([-200, 0])]
        dt = 0.1
        simulation_time = 100

        completed, time_elapsed, routes = simulate([boat], world, [targets], dt, simulation_time, simulated_data=True, motor_only=True)

        self.assertTrue(completed)
        self.assertAlmostEqual(time_elapsed, 91.0)
    

    def test_slalom_easy_simulated(self):

        ## wind initialization
        wind = Wind(1.291)
        world = World(9.81, wind)

        rudder_controller = StepperController(Stepper(100, 0.3), PID(0.5, 0, 0), np.pi * 0.15)
        motor_controller = MotorController(Motor(200, 0.85, 1024))

        ## boat initialization
        boat = Boat(40, 5, None, None, Rudder(rudder_controller), motor_controller)

        boat.position = np.array([0.0, 0.0])
        boat.velocity = np.array([0.0, 0.0])
        boat.heading = polar_to_cartesian(1, 0)

        targets = [np.array([50, 20]), np.array([100, -20]), np.array([150, 20]), np.array([200, -20])]
        dt = 0.1
        simulation_time = 100

        completed, time_elapsed, routes = simulate([boat], world, [targets], dt, simulation_time, simulated_data=True, motor_only=True)

        self.assertTrue(completed)
        self.assertAlmostEqual(time_elapsed, 96.0)
    
    def test_slalom_medium_simulated(self):

        ## wind initialization
        wind = Wind(1.291)
        world = World(9.81, wind)

        rudder_controller = StepperController(Stepper(100, 0.3), PID(0.5, 0, 0), np.pi * 0.15)
        motor_controller = MotorController(Motor(200, 0.85, 1024))

        ## boat initialization
        boat = Boat(40, 5, None, None, Rudder(rudder_controller), motor_controller)

        boat.position = np.array([0.0, 0.0])
        boat.velocity = np.array([0.0, 0.0])
        boat.heading = polar_to_cartesian(1, 0)

        targets = [np.array([50, 50]), np.array([100, -50]), np.array([150, 50]), np.array([200, -50])]
        dt = 0.1
        simulation_time = 200

        completed, time_elapsed, routes = simulate([boat], world, [targets], dt, simulation_time, simulated_data=True, motor_only=True)

        self.assertTrue(completed)
        self.assertAlmostEqual(time_elapsed, 171.4)
    
    def test_sparse_simulated(self):

        ## wind initialization
        wind = Wind(1.291)
        world = World(9.81, wind)

        rudder_controller = StepperController(Stepper(100, 0.3), PID(0.5, 0, 0), np.pi * 0.15)
        motor_controller = MotorController(Motor(200, 0.85, 1024))

        ## boat initialization
        boat = Boat(40, 5, None, None, Rudder(rudder_controller), motor_controller)

        boat.position = np.array([0.0, 0.0])
        boat.velocity = np.array([0.0, 0.0])
        boat.heading = polar_to_cartesian(1, 0)

        targets = [np.array([50, 50]), np.array([-100, -50]), np.array([150, 50]), np.array([-200, -50])]
        dt = 0.1
        simulation_time = 450

        completed, time_elapsed, routes = simulate([boat], world, [targets], dt, simulation_time, simulated_data=True, motor_only=True)

        self.assertTrue(completed)
        self.assertAlmostEqual(time_elapsed, 347.3)
    
    def test_straight_line_measured(self):
        ## wind initialization
        wind = Wind(1.291)
        world = World(9.81, wind)

        rudder_controller = StepperController(Stepper(100, 0.3), PID(0.5, 0, 0), np.pi * 0.15)
        motor_controller = MotorController(Motor(200, 0.85, 1024))

        ## sensor intialization
        anemometer = Anemometer(RelativeError(0.05), AbsoluteError(np.pi/180))
        speedometer_par = Speedometer(MixedError(0.01, 5))
        speedometer_per = Speedometer(MixedError(0.01, 5))
        compass = Compass(AbsoluteError(3*np.pi/180))
        gnss = GNSS(AbsoluteError(1.5), AbsoluteError(1.5))


        ## boat initialization
        boat = Boat(40, 5, None, None, Rudder(rudder_controller), motor_controller, gnss, compass, anemometer, speedometer_par, speedometer_per)

        boat.position = np.array([0.0, 0.0])
        boat.velocity = np.array([0.0, 0.0])
        boat.heading = polar_to_cartesian(1, 0)

        targets = [np.array([50, 0]), np.array([100, 0]), np.array([150, 0]), np.array([200, 0])]
        dt = 0.1
        simulation_time = 100

        completed, time_elapsed, routes = simulate([boat], world, [targets], dt, simulation_time, measured_data=True, motor_only=True)

        self.assertTrue(completed)
        self.assertLess(np.abs(time_elapsed - 70.0), 20)
    
    def test_straight_line_opposite_measured(self):
        ## wind initialization
        wind = Wind(1.291)
        world = World(9.81, wind)

        rudder_controller = StepperController(Stepper(100, 0.3), PID(0.5, 0, 0), np.pi * 0.15)
        motor_controller = MotorController(Motor(200, 0.85, 1024))

        ## sensor intialization
        anemometer = Anemometer(RelativeError(0.05), AbsoluteError(np.pi/180))
        speedometer_par = Speedometer(MixedError(0.01, 5))
        speedometer_per = Speedometer(MixedError(0.01, 5))
        compass = Compass(AbsoluteError(3*np.pi/180))
        gnss = GNSS(AbsoluteError(1.5), AbsoluteError(1.5))


        ## boat initialization
        boat = Boat(40, 5, None, None, Rudder(rudder_controller), motor_controller, gnss, compass, anemometer, speedometer_par, speedometer_per)

        boat.position = np.array([0.0, 0.0])
        boat.velocity = np.array([0.0, 0.0])
        boat.heading = polar_to_cartesian(1, 0)

        targets = [np.array([-50, 0]), np.array([-100, 0]), np.array([-150, 0]), np.array([-200, 0])]
        dt = 0.1
        simulation_time = 100

        completed, time_elapsed, routes = simulate([boat], world, [targets], dt, simulation_time, measured_data=True, motor_only=True)

        self.assertTrue(completed)
        self.assertLess(np.abs(time_elapsed - 90.0), 20)
    
    def test_slalom_easy_measured(self):

        ## wind initialization
        wind = Wind(1.291)
        world = World(9.81, wind)

        # actuators initialization
        rudder_controller = StepperController(Stepper(100, 0.3), PID(0.5, 0, 0), np.pi * 0.15)
        motor_controller = MotorController(Motor(200, 0.85, 1024))

        ## sensor intialization
        anemometer = Anemometer(RelativeError(0.05), AbsoluteError(np.pi/180))
        speedometer_par = Speedometer(MixedError(0.01, 5))
        speedometer_per = Speedometer(MixedError(0.01, 5))
        compass = Compass(AbsoluteError(3*np.pi/180))
        gnss = GNSS(AbsoluteError(1.5), AbsoluteError(1.5))

        ## boat initialization
        boat = Boat(40, 5, None, None, Rudder(rudder_controller), motor_controller, gnss, compass, anemometer, speedometer_par, speedometer_per)

        boat.position = np.array([0.0, 0.0])
        boat.velocity = np.array([0.0, 0.0])
        boat.heading = polar_to_cartesian(1, 0)

        targets = [np.array([50, 20]), np.array([100, -20]), np.array([150, 20]), np.array([200, -20])]
        dt = 0.1
        simulation_time = 200

        completed, time_elapsed, routes = simulate([boat], world, [targets], dt, simulation_time, measured_data=True, motor_only=True)

        self.assertTrue(completed)
        self.assertLess(np.abs(time_elapsed - 90), 20)
    
    def test_slalom_medium_measured(self):

        ## wind initialization
        wind = Wind(1.291)
        world = World(9.81, wind)

        rudder_controller = StepperController(Stepper(100, 0.3), PID(0.5, 0, 0), np.pi * 0.15)
        motor_controller = MotorController(Motor(200, 0.85, 1024))

        ## sensor intialization
        anemometer = Anemometer(RelativeError(0.05), AbsoluteError(np.pi/180))
        speedometer_par = Speedometer(MixedError(0.01, 5))
        speedometer_per = Speedometer(MixedError(0.01, 5))
        compass = Compass(AbsoluteError(3*np.pi/180))
        gnss = GNSS(AbsoluteError(1.5), AbsoluteError(1.5))

        ## boat initialization
        boat = Boat(40, 5, None, None, Rudder(rudder_controller), motor_controller, gnss, compass, anemometer, speedometer_par, speedometer_per)

        boat.position = np.array([0.0, 0.0])
        boat.velocity = np.array([0.0, 0.0])
        boat.heading = polar_to_cartesian(1, 0)

        targets = [np.array([50, 50]), np.array([100, -50]), np.array([150, 50]), np.array([200, -50])]
        dt = 0.1
        simulation_time = 300

        completed, time_elapsed, routes = simulate([boat], world, [targets], dt, simulation_time, measured_data=True, motor_only=True)

        self.assertTrue(completed)
        self.assertLess(np.abs(time_elapsed - 160), 20)
    
    def test_sparse_measured(self):

        ## wind initialization
        wind = Wind(1.291)
        world = World(9.81, wind)

        rudder_controller = StepperController(Stepper(100, 0.3), PID(0.5, 0, 0), np.pi * 0.15)
        motor_controller = MotorController(Motor(200, 0.85, 1024))

        ## sensor intialization
        anemometer = Anemometer(RelativeError(0.05), AbsoluteError(np.pi/180))
        speedometer_par = Speedometer(MixedError(0.01, 5))
        speedometer_per = Speedometer(MixedError(0.01, 5))
        compass = Compass(AbsoluteError(3*np.pi/180))
        gnss = GNSS(AbsoluteError(1.5), AbsoluteError(1.5))

        ## boat initialization
        boat = Boat(40, 5, None, None, Rudder(rudder_controller), motor_controller, gnss, compass, anemometer, speedometer_par, speedometer_per)

        boat.position = np.array([0.0, 0.0])
        boat.velocity = np.array([0.0, 0.0])
        boat.heading = polar_to_cartesian(1, 0)

        targets = [np.array([50, 50]), np.array([-100, -50]), np.array([150, 50]), np.array([-200, -50])]
        dt = 0.1
        simulation_time = 500

        completed, time_elapsed, routes = simulate([boat], world, [targets], dt, simulation_time, measured_data=True, motor_only=True)

        self.assertTrue(completed)
        self.assertLess(np.abs(time_elapsed - 350), 20)

class TestWingMotor(unittest.TestCase):

    def test_straight_line_simulated(self):
        ## wind initialization
        wind = Wind(1.291)
        world = World(9.81, wind)

        rudder_controller = StepperController(Stepper(100, 0.3), PID(0.5, 0, 0), np.pi * 0.15)
        wing_controller = StepperController(Stepper(100, 0.3), PID(0.5, 0, 0))
        motor_controller = MotorController(Motor(200, 0.85, 1024))

        ## boat initialization
        boat = Boat(40, 5, None, Wing(8, wing_controller), Rudder(rudder_controller), motor_controller)

        boat.position = np.array([0.0, 0.0])
        boat.velocity = np.array([0.0, 0.0])
        boat.heading = polar_to_cartesian(1, 0)

        targets = [np.array([50, 0]), np.array([100, 0]), np.array([150, 0]), np.array([200, 0])]
        dt = 0.1
        simulation_time = 100

        completed, time_elapsed, routes = simulate([boat], world, [targets], dt, simulation_time, simulated_data=True)

        self.assertTrue(completed)
        self.assertAlmostEqual(time_elapsed, 70.6)
    
    def test_straight_line_opposite_simulated(self):
        ## wind initialization
        wind = Wind(1.291)
        world = World(9.81, wind)

        rudder_controller = StepperController(Stepper(100, 0.3), PID(0.5, 0, 0), np.pi * 0.15)
        wing_controller = StepperController(Stepper(100, 0.3), PID(0.5, 0, 0))
        motor_controller = MotorController(Motor(200, 0.85, 1024))

        ## boat initialization
        boat = Boat(40, 5, None, Wing(8, wing_controller), Rudder(rudder_controller), motor_controller)

        boat.position = np.array([0.0, 0.0])
        boat.velocity = np.array([0.0, 0.0])
        boat.heading = polar_to_cartesian(1, 0)

        targets = [np.array([-50, 0]), np.array([-100, 0]), np.array([-150, 0]), np.array([-200, 0])]
        dt = 0.1
        simulation_time = 100

        completed, time_elapsed, routes = simulate([boat], world, [targets], dt, simulation_time, simulated_data=True)

        self.assertTrue(completed)
        self.assertAlmostEqual(time_elapsed, 91.1)

    def test_slalom_easy_simulated(self):
        ## wind initialization
        wind = Wind(1.291)
        world = World(9.81, wind)

        rudder_controller = StepperController(Stepper(100, 0.3), PID(0.5, 0, 0), np.pi * 0.15)
        wing_controller = StepperController(Stepper(100, 0.3), PID(0.5, 0, 0))
        motor_controller = MotorController(Motor(200, 0.85, 1024))

        wing_area = 8
        ## boat initialization
        boat = Boat(40, 5, None, Wing(wing_area, wing_controller), Rudder(rudder_controller), motor_controller)

        boat.position = np.array([0.0, 0.0])
        boat.velocity = np.array([0.0, 0.0])
        boat.heading = polar_to_cartesian(1, 0)
        
        wind.velocity = np.array([12.0, -6.0])

        targets = [np.array([50, 20]), np.array([100, -20]), np.array([150, 20]), np.array([200, -20])]
        dt = 0.1
        simulation_time = 100

        completed, time_elapsed, routes = simulate([boat], world, [targets], dt, simulation_time, simulated_data=True)

        self.assertTrue(completed)
        self.assertAlmostEqual(time_elapsed, 71.2)
    
    def test_slalom_medium_simulated(self):
        ## wind initialization
        wind = Wind(1.291)
        world = World(9.81, wind)

        rudder_controller = StepperController(Stepper(100, 0.3), PID(0.5, 0, 0), np.pi * 0.15)
        wing_controller = StepperController(Stepper(100, 0.3), PID(0.5, 0, 0))
        motor_controller = MotorController(Motor(200, 0.85, 1024))

        wing_area = 8
        ## boat initialization
        boat = Boat(40, 5, None, Wing(wing_area, wing_controller), Rudder(rudder_controller), motor_controller)

        boat.position = np.array([0.0, 0.0])
        boat.velocity = np.array([0.0, 0.0])
        boat.heading = polar_to_cartesian(1, 0)
        
        wind.velocity = np.array([12.0, -6.0])

        targets = [np.array([50, 50]), np.array([100, -50]), np.array([150, 50]), np.array([200, -50])]
        dt = 0.1
        simulation_time = 150

        completed, time_elapsed, routes = simulate([boat], world, [targets], dt, simulation_time, simulated_data=True)

        self.assertTrue(completed)
        self.assertAlmostEqual(time_elapsed, 128.5)
    
    def test_sparse_simulated(self):
        ## wind initialization
        wind = Wind(1.291)
        world = World(9.81, wind)

        rudder_controller = StepperController(Stepper(100, 0.3), PID(0.5, 0, 0), np.pi * 0.15)
        wing_controller = StepperController(Stepper(100, 0.3), PID(0.5, 0, 0))
        motor_controller = MotorController(Motor(200, 0.85, 1024))

        wing_area = 8
        ## boat initialization
        boat = Boat(40, 5, None, Wing(wing_area, wing_controller), Rudder(rudder_controller), motor_controller)

        boat.position = np.array([0.0, 0.0])
        boat.velocity = np.array([0.0, 0.0])
        boat.heading = polar_to_cartesian(1, 0)
        
        wind.velocity = np.array([12.0, -6.0])

        targets = [np.array([50, 50]), np.array([-100, -50]), np.array([150, 50]), np.array([-200, -50])]
        dt = 0.1
        simulation_time = 400

        completed, time_elapsed, routes = simulate([boat], world, [targets], dt, simulation_time, simulated_data=True)

        self.assertTrue(completed)
        self.assertAlmostEqual(time_elapsed, 319.5)
    
    def test_straight_line_measured(self):
        ## wind initialization
        wind = Wind(1.291)
        world = World(9.81, wind)

        rudder_controller = StepperController(Stepper(100, 0.3), PID(0.5, 0, 0), np.pi * 0.15)
        wing_controller = StepperController(Stepper(100, 0.3), PID(0.5, 0, 0))
        motor_controller = MotorController(Motor(200, 0.85, 1024))

        ## sensor intialization
        anemometer = Anemometer(RelativeError(0.05), AbsoluteError(np.pi/180))
        speedometer_par = Speedometer(MixedError(0.01, 5))
        speedometer_per = Speedometer(MixedError(0.01, 5))
        compass = Compass(AbsoluteError(3*np.pi/180))
        gnss = GNSS(AbsoluteError(1.5), AbsoluteError(1.5))


        ## boat initialization
        boat = Boat(40, 5, None, Wing(8, wing_controller), Rudder(rudder_controller), motor_controller, gnss, compass, anemometer, speedometer_par, speedometer_per)

        boat.position = np.array([0.0, 0.0])
        boat.velocity = np.array([0.0, 0.0])
        boat.heading = polar_to_cartesian(1, 0)

        targets = [np.array([50, 0]), np.array([100, 0]), np.array([150, 0]), np.array([200, 0])]
        dt = 0.1
        simulation_time = 100

        completed, time_elapsed, routes = simulate([boat], world, [targets], dt, simulation_time, measured_data=True)

        self.assertTrue(completed)
        self.assertLess(np.abs(time_elapsed - 70.0), 20)
    
    def test_straight_line_opposite_measured(self):
        ## wind initialization
        wind = Wind(1.291)
        world = World(9.81, wind)

        rudder_controller = StepperController(Stepper(100, 0.3), PID(0.5, 0, 0), np.pi * 0.15)
        wing_controller = StepperController(Stepper(100, 0.3), PID(0.5, 0, 0))
        motor_controller = MotorController(Motor(200, 0.85, 1024))

        ## sensor intialization
        anemometer = Anemometer(RelativeError(0.05), AbsoluteError(np.pi/180))
        speedometer_par = Speedometer(MixedError(0.01, 5))
        speedometer_per = Speedometer(MixedError(0.01, 5))
        compass = Compass(AbsoluteError(3*np.pi/180))
        gnss = GNSS(AbsoluteError(1.5), AbsoluteError(1.5))


        ## boat initialization
        boat = Boat(40, 5, None, Wing(8, wing_controller), Rudder(rudder_controller), motor_controller, gnss, compass, anemometer, speedometer_par, speedometer_per)

        boat.position = np.array([0.0, 0.0])
        boat.velocity = np.array([0.0, 0.0])
        boat.heading = polar_to_cartesian(1, 0)

        targets = [np.array([-50, 0]), np.array([-100, 0]), np.array([-150, 0]), np.array([-200, 0])]
        dt = 0.1
        simulation_time = 100

        completed, time_elapsed, routes = simulate([boat], world, [targets], dt, simulation_time, measured_data=True)

        self.assertTrue(completed)
        self.assertLess(np.abs(time_elapsed - 90.0), 20)

    
    def test_slalom_easy_measured(self):
        ## wind initialization
        wind = Wind(1.291)
        world = World(9.81, wind)

        rudder_controller = StepperController(Stepper(100, 0.3), PID(0.5, 0, 0), np.pi * 0.15)
        wing_controller = StepperController(Stepper(100, 0.3), PID(0.5, 0, 0))
        motor_controller = MotorController(Motor(200, 0.85, 1024))

        ## sensor intialization
        anemometer = Anemometer(RelativeError(0.05), AbsoluteError(np.pi/180))
        speedometer_par = Speedometer(MixedError(0.01, 5))
        speedometer_per = Speedometer(MixedError(0.01, 5))
        compass = Compass(AbsoluteError(3*np.pi/180))
        gnss = GNSS(AbsoluteError(1.5), AbsoluteError(1.5))

        wing_area = 8
        ## boat initialization
        boat = Boat(40, 5, None, Wing(wing_area, wing_controller), Rudder(rudder_controller), motor_controller, gnss, compass, anemometer, speedometer_par, speedometer_per)

        boat.position = np.array([0.0, 0.0])
        boat.velocity = np.array([0.0, 0.0])
        boat.heading = polar_to_cartesian(1, 0)
        
        wind.velocity = np.array([12.0, -6.0])

        targets = [np.array([50, 20]), np.array([100, -20]), np.array([150, 20]), np.array([200, -20])]
        dt = 0.1
        simulation_time = 150

        completed, time_elapsed, routes = simulate([boat], world, [targets], dt, simulation_time, measured_data=True)

        self.assertTrue(completed)
        self.assertLess(np.abs(time_elapsed - 70), 20)
     
    def test_slalom_medium_measured(self):
        ## wind initialization
        wind = Wind(1.291)
        world = World(9.81, wind)

        rudder_controller = StepperController(Stepper(100, 0.3), PID(0.5, 0, 0), np.pi * 0.15)
        wing_controller = StepperController(Stepper(100, 0.3), PID(0.5, 0, 0))
        motor_controller = MotorController(Motor(200, 0.85, 1024))

        ## sensor intialization
        anemometer = Anemometer(RelativeError(0.05), AbsoluteError(np.pi/180))
        speedometer_par = Speedometer(MixedError(0.01, 5))
        speedometer_per = Speedometer(MixedError(0.01, 5))
        compass = Compass(AbsoluteError(3*np.pi/180))
        gnss = GNSS(AbsoluteError(1.5), AbsoluteError(1.5))

        wing_area = 8
        ## boat initialization
        boat = Boat(40, 5, None, Wing(wing_area, wing_controller), Rudder(rudder_controller), motor_controller, gnss, compass, anemometer, speedometer_par, speedometer_per)

        boat.position = np.array([0.0, 0.0])
        boat.velocity = np.array([0.0, 0.0])
        boat.heading = polar_to_cartesian(1, 0)
        
        wind.velocity = np.array([12.0, -6.0])

        targets = [np.array([50, 50]), np.array([100, -50]), np.array([150, 50]), np.array([200, -50])]
        dt = 0.1
        simulation_time = 200

        completed, time_elapsed, routes = simulate([boat], world, [targets], dt, simulation_time, measured_data=True)

        self.assertTrue(completed)
        self.assertLess(np.abs(time_elapsed - 135), 20)
     
    
    def test_sparse_measured(self):
        ## wind initialization
        wind = Wind(1.291)
        world = World(9.81, wind)

        rudder_controller = StepperController(Stepper(100, 0.3), PID(0.5, 0, 0), np.pi * 0.15)
        wing_controller = StepperController(Stepper(100, 0.3), PID(0.5, 0, 0))
        motor_controller = MotorController(Motor(200, 0.85, 1024))

        ## sensor intialization
        anemometer = Anemometer(RelativeError(0.05), AbsoluteError(np.pi/180))
        speedometer_par = Speedometer(MixedError(0.01, 5))
        speedometer_per = Speedometer(MixedError(0.01, 5))
        compass = Compass(AbsoluteError(3*np.pi/180))
        gnss = GNSS(AbsoluteError(1.5), AbsoluteError(1.5))

        wing_area = 8
        ## boat initialization
        boat = Boat(40, 5, None, Wing(wing_area, wing_controller), Rudder(rudder_controller), motor_controller, gnss, compass, anemometer, speedometer_par, speedometer_per)

        boat.position = np.array([0.0, 0.0])
        boat.velocity = np.array([0.0, 0.0])
        boat.heading = polar_to_cartesian(1, 0)
        
        wind.velocity = np.array([12.0, -6.0])

        targets = [np.array([50, 50]), np.array([-100, -50]), np.array([150, 50]), np.array([-200, -50])]
        dt = 0.1
        simulation_time = 500

        completed, time_elapsed, routes = simulate([boat], world, [targets], dt, simulation_time, measured_data=True)

        self.assertTrue(completed)
        self.assertLess(np.abs(time_elapsed - 325), 20)

if __name__ == '__main__':
    unittest.main()

    # wind initialization
    # wind = Wind(1.291)
    # world = World(9.81, wind)

    # rudder_controller = StepperController(Stepper(100, 0.3), PID(0.5, 0, 0), np.pi * 0.15)
    # wing_controller = StepperController(Stepper(100, 0.3), PID(0.5, 0, 0))
    # motor_controller = MotorController(Motor(200, 0.85, 1024))

    # ## sensor intialization
    # anemometer = Anemometer(RelativeError(0.05), AbsoluteError(np.pi/180))
    # speedometer_par = Speedometer(MixedError(0.01, 5))
    # speedometer_per = Speedometer(MixedError(0.01, 5))
    # compass = Compass(AbsoluteError(3*np.pi/180))
    # gnss = GNSS(AbsoluteError(1.5), AbsoluteError(1.5))

    # wing_area = 8
    # ## boat initialization
    # boat = Boat(40, 5, None, Wing(wing_area, wing_controller), Rudder(rudder_controller), motor_controller, gnss, compass, anemometer, speedometer_par, speedometer_per)

    # boat.position = np.array([0.0, 0.0])
    # boat.velocity = np.array([0.0, 0.0])
    # boat.heading = polar_to_cartesian(1, 0)
    # wind.velocity = np.array([13.0, 8.0])

    # targets = [np.array([50, 0]), np.array([100, 0]), np.array([150, 0]), np.array([200, 0])]
    # dt = 0.1
    # simulation_time = 100

    # completed, time_elapsed, states, collisions = simulate([boat], world, [targets], dt, simulation_time, measured_data=True)

    # win_width = 1400
    # win_height = 800

    # world_width = 700
    # world_height = 300
    # drawer = Drawer(win_width, win_height, world_width, world_height)
    # drawer.debug = True
    
    # drawer.draw_axis()
    # route = [s.position for s in states[0]]

    # drawer.draw_route(route, 'blue')

    # while True:
    #     pass

    # drawer.draw_route(route_b, 'red')
    
    # wind initialization
    # wind = Wind(1.291)
    # wind.velocity = np.array([10.0, -3.0])
    # world = World(9.81, wind)

    # rudder_controller = StepperController(Stepper(100, 0.3), PID(0.5, 0, 0), np.pi * 0.15)
    # motor_controller = MotorController(Motor(200, 0.85, 1024))
    # wing_controller = StepperController(Stepper(100, 0.3), PID(0.5, 0, 0))

    # ## boat initialization
    # boat_a = Boat(40, 5, Wing(10, wing_controller), Rudder(rudder_controller), motor_controller)

    # boat_a.position = np.array([0.0, 0.0])
    # boat_a.velocity = np.array([0.0, 0.0])
    # boat_a.heading = polar_to_cartesian(1, 0)

    # # targets_a = [np.array([-100.0, 0.0])]

    # boat_b = Boat(40, 5, Wing(10, wing_controller), Rudder(copy.deepcopy(rudder_controller)), copy.deepcopy(motor_controller))

    # boat_b.position = np.array([0.0, 10.0])
    # boat_b.velocity = np.array([0.0, 0.0])
    # boat_b.heading = polar_to_cartesian(1, 0)
    
    # # targets_b = [np.array([100.0, 0.0])]

    # targets = [np.array([50, 20]), np.array([100, -20]), np.array([150, 20]), np.array([200, -20])]

    # dt = 0.1
    # simulation_time = 100

    # completed, time_elapsed, states, collisions = simulate([boat_a, boat_b], world, [targets, targets], dt, simulation_time, simulated_data=True, collision_avoidance=True)

    # Logger.info(f'Collisions: {collisions}')

    # win_width = 1400
    # win_height = 800

    # world_width = 700
    # world_height = 300
    # drawer = Drawer(win_width, win_height, world_width, world_height)
    # drawer.debug = True
    
    # # for p in states[0]:
    #     # print(p)
    # i = 0
    # for t in np.arange(0, simulation_time, dt):
    #     drawer.clear()
    #     for t in targets:
    #         drawer.draw_target(t)
    #     drawer.draw_axis()
    #     for b in states:
    #         drawer.draw_boat(b[i])
    #     time.sleep(0.01)
    #     i+=1
    
    # # route_a = list(map(lambda state: state.position, states[0]))
    # route_b = list(map(lambda state: state.position, states[1]))

    # drawer.draw_route(route_a, 'blue')
    # drawer.draw_route(route_b, 'red')

    # time = [t for t in np.arange(0, time_elapsed, dt)]
    # time.append(time_elapsed)

    # # print(len(routes[0]), len(time))
    
    # plt.figure(1)
    # plt.cla()
    # plt.plot(time, list(map(lambda state: state.position[0], states[0])), label='Boat A X')
    # plt.plot(time, list(map(lambda state: state.position[0], states[1])), label='Boat B X')
    # plt.plot(time, list(map(lambda state: state.position[1], states[0])), label='Boat A Y')
    # plt.plot(time, list(map(lambda state: state.position[1], states[1])), label='Boat B Y')


    # plt.figure(2)
    # plt.cla()
    # plt.plot(time, list(map(lambda state: modpi(state.rudder.controller.get_angle()), states[0])), label='Boat A rudder')
    # plt.plot(time, list(map(lambda state: modpi(state.rudder.controller.get_angle()), states[1])), label='Boat B rudder')
    # plt.plot(time, list(map(lambda state: modpi(state.rudder.controller.pid.setpoint), states[0])), label='Boat A target')
    # plt.plot(time, list(map(lambda state: modpi(state.rudder.controller.pid.setpoint), states[1])), label='Boat B target')
    # plt.legend()
    # plt.show()

    # while True:
    #     pass

    # plt.plot(time, list(map(lambda p: p[1], routes[0])), label='Boat Velocity Y')
    # plt.legend()
    # plt.show()


    # simulation_motor_only([np.array([50, 20]), np.array([100, -20]), np.array([150, 20]), np.array([200, -20])])
    # simulation_motor_only([np.array([50, 50]), np.array([100, -50]), np.array([150, 50]), np.array([200, -50])])
    # simulation_motor_only([np.array([50, 50]), np.array([-100, -50]), np.array([150, 50]), np.array([-200, -50])])
    
    # simulation_motor_wing([np.array([50, 20]), np.array([100, -20]), np.array([150, 20]), np.array([200, -20])])
    # simulation_motor_wing([np.array([50, 50]), np.array([100, -50]), np.array([150, 50]), np.array([200, -50])])
    # simulation_motor_wing([np.array([50, 50]), np.array([-100, -50]), np.array([150, 50]), np.array([-200, -50])])
    