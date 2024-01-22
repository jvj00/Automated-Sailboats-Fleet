# seadbed initialization
from typing import Any
import unittest
from actuators import Motor, MotorController, Rudder, Stepper, StepperController, Wing
from disegnino import Drawer
from entities import Boat, Wind, World
from environment import SeabedMap
import numpy as np
import matplotlib.pyplot as plt

from pid import PID
from sensor import GNSS, AbsoluteError, Anemometer, Compass, MixedError, RelativeError, Speedometer
from utils import compute_angle_between, compute_distance, mod2pi, modpi, polar_to_cartesian

# win_width = 900
# win_height = 500

# world_width = 450
# world_height = 250
# drawer = Drawer(win_width, win_height, world_width, world_height)
# drawer.debug = True

def simulate(boats: list[Boat], world: World, targets: list[list[Any]], dt, simulation_time, simulated_data = False, measured_data = False, motor_only = False):
    routes: list[list[Any]] = [[]] * len(boats)

    current_targets_idx = [0] * len(targets)
    target_radius = 5

    for i in range(len(boats)): 
        target = targets[i][current_targets_idx[i]]
        boats[i].set_target(target)
    
    for time_elapsed in np.arange(0, simulation_time, dt):
        completed = True

        for i in range(len(boats)):
    
            completed &= (current_targets_idx[i] == len(targets[i]))

            if current_targets_idx[i] == len(targets[i]):
                    continue
            if compute_distance(targets[i][current_targets_idx[i]], boats[i].position) <= target_radius:
                current_targets_idx[i] += 1
                if current_targets_idx[i] == len(targets[i]):
                    continue
                boats[i].set_target(targets[i][current_targets_idx[i]])

            boats[i].follow_target(world.wind, dt, None, simulated_data, measured_data, motor_only)

            routes[i].append(boats[i].position)
        
        world.update(boats, dt)

        if completed:
            break
    
    completed = True

    for i in range(len(boats)):
        completed &= (current_targets_idx[i] == len(targets[i]))
    
    return completed, time_elapsed


class TestMotorOnly(unittest.TestCase):

    def test_slalom_easy_simulated(self):

        ## wind initialization
        wind = Wind(1.291)
        world = World(9.81, wind)

        rudder_controller = StepperController(Stepper(100, 0.3), PID(0.5, 0, 0), np.pi * 0.15)
        motor_controller = MotorController(Motor(200, 0.85, 1024))

        ## boat initialization
        boat = Boat(40, 5, None, Rudder(rudder_controller), motor_controller)

        boat.position = np.array([0.0, 0.0])
        boat.velocity = np.array([0.0, 0.0])
        boat.heading = polar_to_cartesian(1, 0)

        targets = [np.array([50, 20]), np.array([100, -20]), np.array([150, 20]), np.array([200, -20])]
        dt = 0.1
        simulation_time = 100

        completed, time_elapsed = simulate([boat], world, [targets], dt, simulation_time, simulated_data=True, motor_only=True)

        self.assertTrue(completed)
        self.assertAlmostEqual(time_elapsed, 91.7)
    
    def test_slalom_medium_simulated(self):

        ## wind initialization
        wind = Wind(1.291)
        world = World(9.81, wind)

        rudder_controller = StepperController(Stepper(100, 0.3), PID(0.5, 0, 0), np.pi * 0.15)
        motor_controller = MotorController(Motor(200, 0.85, 1024))

        ## boat initialization
        boat = Boat(40, 5, None, Rudder(rudder_controller), motor_controller)

        boat.position = np.array([0.0, 0.0])
        boat.velocity = np.array([0.0, 0.0])
        boat.heading = polar_to_cartesian(1, 0)

        targets = [np.array([50, 50]), np.array([100, -50]), np.array([150, 50]), np.array([200, -50])]
        dt = 0.1
        simulation_time = 200

        completed, time_elapsed = simulate([boat], world, [targets], dt, simulation_time, simulated_data=True, motor_only=True)

        self.assertTrue(completed)
        self.assertAlmostEqual(time_elapsed, 159.3)
    
    def test_sparse_simulated(self):

        ## wind initialization
        wind = Wind(1.291)
        world = World(9.81, wind)

        rudder_controller = StepperController(Stepper(100, 0.3), PID(0.5, 0, 0), np.pi * 0.15)
        motor_controller = MotorController(Motor(200, 0.85, 1024))

        ## boat initialization
        boat = Boat(40, 5, None, Rudder(rudder_controller), motor_controller)

        boat.position = np.array([0.0, 0.0])
        boat.velocity = np.array([0.0, 0.0])
        boat.heading = polar_to_cartesian(1, 0)

        targets = [np.array([50, 50]), np.array([-100, -50]), np.array([150, 50]), np.array([-200, -50])]
        dt = 0.1
        simulation_time = 450

        completed, time_elapsed = simulate([boat], world, [targets], dt, simulation_time, simulated_data=True, motor_only=True)

        self.assertTrue(completed)
        self.assertAlmostEqual(time_elapsed, 347.3)
    
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
        boat = Boat(40, 5, None, Rudder(rudder_controller), motor_controller, None, gnss, compass, anemometer, speedometer_par, speedometer_per)

        boat.position = np.array([0.0, 0.0])
        boat.velocity = np.array([0.0, 0.0])
        boat.heading = polar_to_cartesian(1, 0)

        targets = [np.array([50, 20]), np.array([100, -20]), np.array([150, 20]), np.array([200, -20])]
        dt = 0.1
        simulation_time = 100

        completed, time_elapsed = simulate([boat], world, [targets], dt, simulation_time, measured_data=True, motor_only=True)

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
        boat = Boat(40, 5, None, Rudder(rudder_controller), motor_controller, None, gnss, compass, anemometer, speedometer_par, speedometer_per)

        boat.position = np.array([0.0, 0.0])
        boat.velocity = np.array([0.0, 0.0])
        boat.heading = polar_to_cartesian(1, 0)

        targets = [np.array([50, 50]), np.array([100, -50]), np.array([150, 50]), np.array([200, -50])]
        dt = 0.1
        simulation_time = 200

        completed, time_elapsed = simulate([boat], world, [targets], dt, simulation_time, measured_data=True, motor_only=True)

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
        boat = Boat(40, 5, None, Rudder(rudder_controller), motor_controller, None, gnss, compass, anemometer, speedometer_par, speedometer_per)

        boat.position = np.array([0.0, 0.0])
        boat.velocity = np.array([0.0, 0.0])
        boat.heading = polar_to_cartesian(1, 0)

        targets = [np.array([50, 50]), np.array([-100, -50]), np.array([150, 50]), np.array([-200, -50])]
        dt = 0.1
        simulation_time = 450

        completed, time_elapsed = simulate([boat], world, [targets], dt, simulation_time, measured_data=True, motor_only=True)

        self.assertTrue(completed)
        self.assertLess(np.abs(time_elapsed - 350), 20)

class TestWingMotor(unittest.TestCase):

    def test_slalom_easy_simulated(self):
        ## wind initialization
        wind = Wind(1.291)
        world = World(9.81, wind)

        rudder_controller = StepperController(Stepper(100, 0.3), PID(0.5, 0, 0), np.pi * 0.15)
        wing_controller = StepperController(Stepper(100, 0.3), PID(0.5, 0, 0))
        motor_controller = MotorController(Motor(200, 0.85, 1024))

        wing_area = 8
        ## boat initialization
        boat = Boat(40, 5, Wing(wing_area, wing_controller), Rudder(rudder_controller), motor_controller)

        boat.position = np.array([0.0, 0.0])
        boat.velocity = np.array([0.0, 0.0])
        boat.heading = polar_to_cartesian(1, 0)
        
        wind.velocity = np.array([12.0, -6.0])

        targets = [np.array([50, 20]), np.array([100, -20]), np.array([150, 20]), np.array([200, -20])]
        dt = 0.1
        simulation_time = 100

        completed, time_elapsed = simulate([boat], world, [targets], dt, simulation_time, simulated_data=True)

        self.assertTrue(completed)
        self.assertAlmostEqual(time_elapsed, 70.3)
    
    def test_slalom_medium_simulated(self):
        ## wind initialization
        wind = Wind(1.291)
        world = World(9.81, wind)

        rudder_controller = StepperController(Stepper(100, 0.3), PID(0.5, 0, 0), np.pi * 0.15)
        wing_controller = StepperController(Stepper(100, 0.3), PID(0.5, 0, 0))
        motor_controller = MotorController(Motor(200, 0.85, 1024))

        wing_area = 8
        ## boat initialization
        boat = Boat(40, 5, Wing(wing_area, wing_controller), Rudder(rudder_controller), motor_controller)

        boat.position = np.array([0.0, 0.0])
        boat.velocity = np.array([0.0, 0.0])
        boat.heading = polar_to_cartesian(1, 0)
        
        wind.velocity = np.array([12.0, -6.0])

        targets = [np.array([50, 50]), np.array([100, -50]), np.array([150, 50]), np.array([200, -50])]
        dt = 0.1
        simulation_time = 150

        completed, time_elapsed = simulate([boat], world, [targets], dt, simulation_time, simulated_data=True)

        self.assertTrue(completed)
        self.assertAlmostEqual(time_elapsed, 135.6)
    
    def test_sparse_simulated(self):
        ## wind initialization
        wind = Wind(1.291)
        world = World(9.81, wind)

        rudder_controller = StepperController(Stepper(100, 0.3), PID(0.5, 0, 0), np.pi * 0.15)
        wing_controller = StepperController(Stepper(100, 0.3), PID(0.5, 0, 0))
        motor_controller = MotorController(Motor(200, 0.85, 1024))

        wing_area = 8
        ## boat initialization
        boat = Boat(40, 5, Wing(wing_area, wing_controller), Rudder(rudder_controller), motor_controller)

        boat.position = np.array([0.0, 0.0])
        boat.velocity = np.array([0.0, 0.0])
        boat.heading = polar_to_cartesian(1, 0)
        
        wind.velocity = np.array([12.0, -6.0])

        targets = [np.array([50, 50]), np.array([-100, -50]), np.array([150, 50]), np.array([-200, -50])]
        dt = 0.1
        simulation_time = 400

        completed, time_elapsed = simulate([boat], world, [targets], dt, simulation_time, simulated_data=True)

        self.assertTrue(completed)
        self.assertAlmostEqual(time_elapsed, 325.5)

    
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
        boat = Boat(40, 5, Wing(wing_area, wing_controller), Rudder(rudder_controller), motor_controller, None, gnss, compass, anemometer, speedometer_par, speedometer_per)

        boat.position = np.array([0.0, 0.0])
        boat.velocity = np.array([0.0, 0.0])
        boat.heading = polar_to_cartesian(1, 0)
        
        wind.velocity = np.array([12.0, -6.0])

        targets = [np.array([50, 20]), np.array([100, -20]), np.array([150, 20]), np.array([200, -20])]
        dt = 0.1
        simulation_time = 100

        completed, time_elapsed = simulate([boat], world, [targets], dt, simulation_time, measured_data=True)

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
        boat = Boat(40, 5, Wing(wing_area, wing_controller), Rudder(rudder_controller), motor_controller, None, gnss, compass, anemometer, speedometer_par, speedometer_per)

        boat.position = np.array([0.0, 0.0])
        boat.velocity = np.array([0.0, 0.0])
        boat.heading = polar_to_cartesian(1, 0)
        
        wind.velocity = np.array([12.0, -6.0])

        targets = [np.array([50, 50]), np.array([100, -50]), np.array([150, 50]), np.array([200, -50])]
        dt = 0.1
        simulation_time = 150

        completed, time_elapsed = simulate([boat], world, [targets], dt, simulation_time, measured_data=True)

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
        boat = Boat(40, 5, Wing(wing_area, wing_controller), Rudder(rudder_controller), motor_controller, None, gnss, compass, anemometer, speedometer_par, speedometer_per)

        boat.position = np.array([0.0, 0.0])
        boat.velocity = np.array([0.0, 0.0])
        boat.heading = polar_to_cartesian(1, 0)
        
        wind.velocity = np.array([12.0, -6.0])

        targets = [np.array([50, 50]), np.array([-100, -50]), np.array([150, 50]), np.array([-200, -50])]
        dt = 0.1
        simulation_time = 400

        completed, time_elapsed = simulate([boat], world, [targets], dt, simulation_time, measured_data=True)

        self.assertTrue(completed)
        self.assertLess(np.abs(time_elapsed - 325), 20)

class TestCollisionAvoidance(unittest.TestCase):

    def test_motor_only(self):
        ## wind initialization
        wind = Wind(1.291)
        world = World(9.81, wind)

        rudder_controller = StepperController(Stepper(100, 0.3), PID(0.5, 0, 0), np.pi * 0.15)
        wing_controller = StepperController(Stepper(100, 0.3), PID(0.5, 0, 0))
        motor_controller = MotorController(Motor(200, 0.85, 1024))

        wing_area = 8
        ## boat initialization
        boat = Boat(40, 5, Wing(wing_area, wing_controller), Rudder(rudder_controller), motor_controller)

        boat.position = np.array([0.0, 0.0])
        boat.velocity = np.array([0.0, 0.0])
        boat.heading = polar_to_cartesian(1, 0)
        
        wind.velocity = np.array([12.0, -6.0])

        targets = [np.array([50, 50]), np.array([-100, -50]), np.array([150, 50]), np.array([-200, -50])]
        dt = 0.1
        simulation_time = 400

        completed, time_elapsed = simulate([boat], world, [targets], dt, simulation_time, simulated_data=True, motor_only=True)

      
        # current_target_idx = 0
        # target_radius = 5
        # dt = 0.1
        # simulation_time = 100
        # route = []
        # time = []
        # rudder_angle = []
        # rudder_speed = []

        # boat.position = np.array([0.0, 0.0])
        # boat.velocity = np.array([0.0, 0.0])
        # boat.heading = polar_to_cartesian(1, 0)
        # boat.wing = None

        # boat.set_target(targets[current_target_idx])

        # for time_elapsed in np.arange(0, simulation_time, dt):
        #     if compute_distance(targets[current_target_idx], boat.position) <= target_radius:
        #         current_target_idx += 1
        #         if current_target_idx >= len(targets):
        #             break
        #         boat.set_target(targets[current_target_idx])

        #     boat.follow_target(world.wind, dt, simulated_data=True, motor_only=True)

        #     world.update([boat], dt)
        
        

    # route.append(np.copy(boat.position))
    # time.append(time_elapsed)
    # if boat.rudder is not None:
    #     rudder_angle.append(modpi(boat.rudder.controller.get_angle()))
    #     rudder_speed.append(boat.rudder.controller.speed * boat.rudder.controller.direction)

    # drawer.draw_axis()
    # drawer.draw_route(route, 'blue')
    # for t in targets:
    #     drawer.draw_target(t)
    # drawer.draw_wind(world.wind, np.array([world_width * 0.4, world_height * 0.4]))

    # plt.plot(time, list(map(lambda p: p[0], route)), label='Boat Velocity X')
    # plt.plot(time, list(map(lambda p: p[1], route)), label='Boat Velocity Y')
    # if len(time) == len(rudder_angle):
    #     plt.plot(time, rudder_angle, label='Rudder angle')
    #     plt.plot(time, rudder_speed, label='Rudder speed')

    # plt.legend()
    # plt.show()

    # while True:
    #     pass

# def simulation_motor_wing(targets):

#     current_target_idx = 0
#     target_radius = 5
#     route = []
#     time = []
#     rudder_angle = []
#     rudder_speed = []
#     wing_angle = []
#     wing_speed = []

#     boat.position = np.array([0.0, 0.0])
#     boat.velocity = np.array([0.0, 0.0])
#     boat.heading = polar_to_cartesian(1, 0)

#     wind.velocity = np.array([12.0, -8.0])

#     dt = 0.1

#     boat.set_target(targets[current_target_idx])

#     for time_elapsed in np.arange(0, 500, dt):
#         if compute_distance(targets[current_target_idx], boat.position) <= target_radius:
#             current_target_idx += 1
#             if current_target_idx >= len(targets):
#                 break
#             boat.set_target(targets[current_target_idx])

#         boat.follow_target(world.wind, dt, simulated_data=True)

#         world.update([boat], dt)

#         route.append(np.copy(boat.position))
#         time.append(time_elapsed)
#         if boat.rudder is not None:
#             rudder_angle.append(modpi(boat.rudder.controller.get_angle()))
#             rudder_speed.append(boat.rudder.controller.speed * boat.rudder.controller.direction)
#         if boat.wing is not None:
#             wing_angle.append(modpi(boat.wing.controller.get_angle()))
#             wing_speed.append(boat.wing.controller.speed * boat.wing.controller.direction)

#     drawer.draw_axis()
#     drawer.draw_route(route, 'blue')
#     for t in targets:
#         drawer.draw_target(t)
#     drawer.draw_wind(world.wind, np.array([world_width * 0.4, world_height * 0.4]))

#     # plt.plot(time, list(map(lambda p: p[0], route)), label='Boat Velocity X')
#     # plt.plot(time, list(map(lambda p: p[1], route)), label='Boat Velocity Y')
#     if len(time) == len(rudder_angle):
#         plt.plot(time, rudder_angle, label='Rudder angle')
#         plt.plot(time, rudder_speed, label='Rudder speed')
    
#     if len(time) == len(wing_angle):
#         plt.plot(time, wing_angle, label='Wing angle')
#         plt.plot(time, wing_speed, label='Wing speed')
#     plt.legend()
#     plt.show()

#     while True:
#         pass

if __name__ == '__main__':
    unittest.main()

    # simulation_motor_only([np.array([50, 20]), np.array([100, -20]), np.array([150, 20]), np.array([200, -20])])
    # simulation_motor_only([np.array([50, 50]), np.array([100, -50]), np.array([150, 50]), np.array([200, -50])])
    # simulation_motor_only([np.array([50, 50]), np.array([-100, -50]), np.array([150, 50]), np.array([-200, -50])])
    
    # simulation_motor_wing([np.array([50, 20]), np.array([100, -20]), np.array([150, 20]), np.array([200, -20])])
    # simulation_motor_wing([np.array([50, 50]), np.array([100, -50]), np.array([150, 50]), np.array([200, -50])])
    # simulation_motor_wing([np.array([50, 50]), np.array([-100, -50]), np.array([150, 50]), np.array([-200, -50])])
    