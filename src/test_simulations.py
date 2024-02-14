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
from estimation_algs.ekf import EKF
from sensors.anemometer import Anemometer
from sensors.compass import Compass
from sensors.gnss import GNSS
from sensors.speedometer import Speedometer
from components.rudder import Rudder
from components.wing import Wing
from tools.disegnino import Drawer
from entities.environment import SeabedMap
import numpy as np
import matplotlib.pyplot as plt
import copy

from controllers.pid import PID
from tools.utils import check_intersection_circle_circle, compute_angle, compute_angle_between, compute_distance, compute_magnitude, is_multiple, mod2pi, modpi, polar_to_cartesian, random_color, value_from_gaussian

from tools.logger import Logger

class BoatConfiguration:
    def __init__(
        self, 
        simulated_data: bool = True,
        measured_data: bool = False,
        filtered_data: bool = False,
        motor_only: bool = False,
        gnss_period: float = 1,
        compass_period: float = 1,
        other_sensors_period: float = 1,
        follow_target_period: float = 1,
        update_filtered_state_period: float = 1,
        gnss_probability: float = 1,
        compass_probability: float = 1,
        other_sensors_probability: float = 1,
    ):
        self.simulated_data = simulated_data
        self.measured_data = measured_data
        self.filtered_data = filtered_data
        self.motor_only = motor_only
        self.gnss_period = gnss_period
        self.compass_period = compass_period
        self.other_sensors_period = other_sensors_period
        self.follow_target_period = follow_target_period
        self.update_filtered_state_period = update_filtered_state_period
        self.gnss_probability = gnss_probability
        self.compass_probability = compass_probability
        self.other_sensors_probability = other_sensors_probability

def simulate(
        boats: list[Boat],
        world: World,
        targets: list[list[np.ndarray]],
        dt: float,
        simulation_time: float,
        configurations: list[BoatConfiguration],
        variable_wind: bool = False
    ):
    
    # keeps, for each boat, its state at each instant dt
    states: list[list[Boat]] = [[] for _ in range(len(boats))]

    # keeps, for each boat, the index of its current target
    current_targets_idx = [0 for _ in range(len(targets))]

    # set the first target of each boat
    for i in range(len(boats)): 
        target = targets[i][0]
        boats[i].set_target(target)
    
    time_elapsed = 0

    wind_derivative = np.array([0.0, 0.0])
    
    for i in range(int(simulation_time/dt)):
        time_elapsed = round(i * dt, 2)

        # update wind conditions
        if variable_wind:
            wind_derivative_mag = value_from_gaussian(2*(1-compute_magnitude(world.wind.velocity)/15), 6)
            wind_derivative_angle = compute_angle(world.wind.velocity) + value_from_gaussian(0, 0.8)
            wind_derivative = polar_to_cartesian(wind_derivative_mag, wind_derivative_angle)
            world.wind.velocity += wind_derivative * dt

        # check if every boat has covered every target
        completed = True

        for b_i in range(len(boats)):

            b = boats[b_i]
            c = configurations[b_i]
    
            completed &= (current_targets_idx[b_i] == len(targets[b_i]))

            if current_targets_idx[b_i] == len(targets[b_i]):
                    continue
                                
            # check if the boat has reached the target        
            if check_intersection_circle_circle(b.position, b.length * 0.5, targets[b_i][current_targets_idx[b_i]], b.length * 0.5):
                current_targets_idx[b_i] += 1
                if current_targets_idx[b_i] == len(targets[b_i]):
                    continue
                b.set_target(targets[b_i][current_targets_idx[b_i]])
            
            states[b_i].append(copy.deepcopy(b))

            if c.measured_data or c.filtered_data:
            
                if is_multiple(time_elapsed, c.gnss_period):
                    b.measure_gnss()
                
                if is_multiple(time_elapsed, c.compass_period):
                    b.measure_compass()
                
                if is_multiple(time_elapsed, c.other_sensors_period):
                    b.measure_anemometer(world.wind)
                    b.measure_speedometer_par()
                    b.measure_speedometer_perp()
                
            if is_multiple(time_elapsed, c.follow_target_period):
                b.follow_target(
                    world.wind,
                    c.follow_target_period,
                    c.simulated_data,
                    c.measured_data,
                    c.filtered_data,
                    c.motor_only
                )

            if c.filtered_data and is_multiple(time_elapsed, c.update_filtered_state_period):
                try:
                    b.update_filtered_state(c.update_filtered_state_period, is_multiple(time_elapsed, c.gnss_period), is_multiple(time_elapsed, c.compass_period))
                except Exception as e:
                    Logger.error(e)

        world.update(boats, dt)

        if completed:
            break

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

        completed, time_elapsed, routes = simulate([boat], world, [targets], dt, simulation_time, [BoatConfiguration(True, False, False, True, dt, dt, dt, dt)])

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

        completed, time_elapsed, routes = simulate([boat], world, [targets], dt, simulation_time, [BoatConfiguration(True, False, False, True, dt, dt, dt, dt)])

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

        completed, time_elapsed, routes = simulate([boat], world, [targets], dt, simulation_time, [BoatConfiguration(True, False, False, True, dt, dt, dt, dt)])

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

        completed, time_elapsed, routes = simulate([boat], world, [targets], dt, simulation_time, [BoatConfiguration(True, False, False, True, dt, dt, dt, dt)])

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

        completed, time_elapsed, routes = simulate([boat], world, [targets], dt, simulation_time, [BoatConfiguration(True, False, False, True, dt, dt, dt, dt)])

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

        completed, time_elapsed, routes = simulate([boat], world, [targets], dt, simulation_time, [BoatConfiguration(False, True, False, True, dt, dt, dt, dt)])

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

        completed, time_elapsed, routes = simulate([boat], world, [targets], dt, simulation_time, [BoatConfiguration(False, True, False, True, dt, dt, dt, dt)])

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

        completed, time_elapsed, routes = simulate([boat], world, [targets], dt, simulation_time, [BoatConfiguration(False, True, False, True, dt, dt, dt, dt)])

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

        completed, time_elapsed, routes = simulate([boat], world, [targets], dt, simulation_time, [BoatConfiguration(False, True, False, True, dt, dt, dt, dt)])

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

        completed, time_elapsed, routes = simulate([boat], world, [targets], dt, simulation_time, [BoatConfiguration(False, True, False, True, dt, dt, dt, dt)])

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

        completed, time_elapsed, routes = simulate([boat], world, [targets], dt, simulation_time, [BoatConfiguration(True, False, False, False, dt, dt, dt, dt)])

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

        completed, time_elapsed, routes = simulate([boat], world, [targets], dt, simulation_time, [BoatConfiguration(True, False, False, False, dt, dt, dt, dt)])

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

        completed, time_elapsed, routes = simulate([boat], world, [targets], dt, simulation_time, [BoatConfiguration(True, False, False, False, dt, dt, dt, dt)])

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

        completed, time_elapsed, routes = simulate([boat], world, [targets], dt, simulation_time, [BoatConfiguration(True, False, False, False, dt, dt, dt, dt)])

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

        completed, time_elapsed, routes = simulate([boat], world, [targets], dt, simulation_time, [BoatConfiguration(True, False, False, False, dt, dt, dt, dt)])

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

        completed, time_elapsed, routes = simulate([boat], world, [targets], dt, simulation_time, [BoatConfiguration(False, True, False, False, dt, dt, dt, dt)])

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

        completed, time_elapsed, routes = simulate([boat], world, [targets], dt, simulation_time, [BoatConfiguration(False, True, False, False, dt, dt, dt, dt)])

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

        completed, time_elapsed, routes = simulate([boat], world, [targets], dt, simulation_time, [BoatConfiguration(False, True, False, False, dt, dt, dt, dt)])

        self.assertTrue(completed)
        self.assertLess(np.abs(time_elapsed - 70), 20)
     
    def test_slalom_medium_measured(self):
        ## wind initialization
        wind = Wind(1.291)
        wind_mag = np.random.random() * 15 + 5
        wind_ang = np.random.random() * 2 * np.pi
        wind.velocity = polar_to_cartesian(wind_mag, wind_ang)
    
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

        completed, time_elapsed, routes = simulate([boat], world, [targets], dt, simulation_time, [BoatConfiguration(False, True, False, False, dt, dt, dt, dt)])

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

        completed, time_elapsed, routes = simulate([boat], world, [targets], dt, simulation_time, [BoatConfiguration(False, True, False, False, dt, dt, dt, dt)])

        self.assertTrue(completed)
        self.assertLess(np.abs(time_elapsed - 325), 20)


def test_ekf_noekf_comparison(targets, simulation_time, world_update_period, gnss_compass_periods, others_periods, retries):
        # wind initialization
    wind = Wind(1.291)
    wind.velocity = np.array([10.0, -10.0])
    
    # world initialization
    world = World(9.81, wind)

    # boats initialization
    boats: list[Boat] = []
    boats_n = 3

    tgts = []

    for i in range(boats_n):

        ## sensor intialization
        anemometer = Anemometer(RelativeError(0.05), AbsoluteError(np.pi/180))
        speedometer_par = Speedometer(MixedError(0.01, 5), offset_angle=0)
        speedometer_per = Speedometer(MixedError(0.01, 5), offset_angle=-np.pi/2)
        compass = Compass(AbsoluteError(3*np.pi/180))
        gnss = GNSS(AbsoluteError(1.5), AbsoluteError(1.5))

        # actuators initialization
        rudder_controller = StepperController(Stepper(100, 0.3), PID(0.5, 0, 0), np.pi * 0.20)
        wing_controller = StepperController(Stepper(100, 0.3), PID(0.5, 0, 0))
        motor_controller = MotorController(Motor(200, 0.85, 1024))

        # boat initialization
        boat = Boat(80, 5, None, Wing(15, wing_controller), Rudder(rudder_controller), motor_controller, gnss, compass, anemometer, speedometer_par, speedometer_per, None, EKF())
        boat.position = np.zeros(2)
        boat.velocity = np.zeros(2)
        boat.heading = polar_to_cartesian(1, 0)

        # boat ekf setup
        ekf_constants = boat.mass, boat.length, boat.drag_damping, boat.wing.area, wind.density, boat.motor_controller.motor.efficiency
        boat.ekf.set_initial_state(boat.measure_state())
        boat.ekf.set_initial_state_variance(boat.get_state_variance())
        boat.ekf.set_constants(ekf_constants)
        
        boats.append(boat)
        
        tgts.append(targets)
        
    dt = world_update_period
    sensor_reading_prob = 1

    data = {"no_ekf": [], "ekf": []}

    for gnss_compass_dt in gnss_compass_periods:
        for other_sensors_dt in others_periods:
            
            position_error_measured_min = []
            position_error_measured_max = []
            position_error_measured_avg = []

            position_error_filtered_min = []
            position_error_filtered_max = []
            position_error_filtered_avg = []

            print(f"GNSS/Compass period: {gnss_compass_dt}, other sensors period: {other_sensors_dt}")

            for r in range(retries):

                for b in boats:
                    b.reset()
                
                completed, time_elapsed, states = simulate(
                    boats,
                    world,
                    tgts,
                    dt,
                    simulation_time,
                    [
                        BoatConfiguration(True, False, False, False, follow_target_period=dt),
                        BoatConfiguration(False, True, False, False, gnss_compass_dt, gnss_compass_dt, other_sensors_dt, dt, dt, sensor_reading_prob, sensor_reading_prob, sensor_reading_prob),
                        BoatConfiguration(False, False, True, False, gnss_compass_dt, gnss_compass_dt, other_sensors_dt, dt, dt, sensor_reading_prob, sensor_reading_prob, sensor_reading_prob),
                    ]
                )

                for i in range(2):

                    z_measured = list(zip(states[0], states[i+1]))

                    error = [np.round(compute_magnitude(state[0].position - state[1].position), 2) for state in z_measured]

                    error_min = min(*error)
                    error_max = max(*error)
                    error_avg = sum(error) / len(error)

                    if i == 0:
                        position_error_measured_min.append(error_min)
                        position_error_measured_max.append(error_max)
                        position_error_measured_avg.append(error_avg)
                    else:
                        position_error_filtered_min.append(error_min)
                        position_error_filtered_max.append(error_max)
                        position_error_filtered_avg.append(error_avg)
                    
                # drawer.clear()
                # for i in range(len(boats)):
                #     route = [s.position for s in states[i]]
                #     drawer.draw_route(route, colors[i])
        
            position_error_measured_min = round(sum(position_error_measured_min) / len(position_error_measured_min), 2)
            position_error_measured_max = round(sum(position_error_measured_max) / len(position_error_measured_max), 2)
            position_error_measured_avg = round(sum(position_error_measured_avg) / len(position_error_measured_avg), 2)

            d = {"gnss_compass_period": gnss_compass_dt, "sensors_period": other_sensors_dt, "min": position_error_measured_min, "max": position_error_measured_max, "avg": position_error_measured_avg}
            data["no_ekf"].append(d)
            
            position_error_filtered_min = round(sum(position_error_filtered_min) / len(position_error_filtered_min), 2)
            position_error_filtered_max = round(sum(position_error_filtered_max) / len(position_error_filtered_max), 2)
            position_error_filtered_avg = round(sum(position_error_filtered_avg) / len(position_error_filtered_avg), 2)

            d = {"gnss_compass_period": gnss_compass_dt, "sensors_period": other_sensors_dt, "min": position_error_filtered_min, "max": position_error_filtered_max, "avg": position_error_filtered_avg}
            data["ekf"].append(d)

    plt.figure(1)
    plt.cla()
    plt.title('Position')

    x = gnss_compass_periods

    for i in others_periods:
        measured = [d["avg"] for d in data["no_ekf"] if d["sensors_period"] == i]
        plt.plot(x, measured, label=f"No EKF / Other sensors: {i}")

    for i in others_periods:
        filtered = [d["avg"] for d in data["ekf"] if d["sensors_period"] == i]
        plt.plot(x, filtered, label=f"EKF / Other sensors: {i}")

    plt.xlabel("GNSS/Compass udpate period")
    plt.ylabel("Position error")
    
    plt.legend()
    plt.show()


if __name__ == '__main__':
    targets = [np.array([50, 20]), np.array([100, -20]), np.array([150, 20]), np.array([200, -20])]
    simulation_time = 500
    world_update_period = 0.1
    gnss_compass_periods = [i for i in range(1, 10 + 1, 1)]
    others_periods = [2**i for i in range(3 + 1)]
    retries = 3

    test_ekf_noekf_comparison(targets, simulation_time, world_update_period, gnss_compass_periods, others_periods, retries)