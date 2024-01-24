import os
from datetime import datetime
from matplotlib import pyplot as plt
import numpy as np
from actuators.motor import Motor
from actuators.stepper import Stepper
from controllers.motor_controller import MotorController
from controllers.stepper_controller import StepperController
from ekf import EKF
from entities.boat import Boat
from entities.wind import Wind
from entities.world import World
from errors.absolute_error import AbsoluteError
from errors.mixed_error import MixedError
from errors.relative_error import RelativeError
from sensors.anemometer import Anemometer
from sensors.compass import Compass
from sensors.gnss import GNSS
from sensors.sonar import Sonar
from sensors.speedometer import Speedometer
from surfaces.rudder import Rudder
from surfaces.wing import Wing
from tools.disegnino import Drawer
from environment import SeabedMap
from controllers.pid import PID
from tools.logger import Logger
from environment import SeabedMap, SeabedBoatMap
from entities.fleet import Fleet
from tools.utils import *
from tools.metrics import Metrics, GlobalMetrics
from main import create_targets_from_map


def experiment(world_width, world_height, dt, time_experiment, boats_n, boats_per_group_n, prob_of_connection, prob_gnss, prob_compass, dt_gnss, dt_compass, dt_sonar, dt_sync, dt_ekf):
    
    np.set_printoptions(suppress=True)

    # seadbed initialization
    seabed = SeabedMap(int(-world_width*0.3),int(world_width*0.3),int(-world_height*0.3),int(world_height*0.3), resolution=15)
    seabed.create_seabed(20, 300, max_slope=2, prob_go_up=0.1, plot=False)

    # wind initialization
    wind = Wind(1.291)
    wind.velocity = np.array([10.0, -10.0])
    wind_derivative = np.array([0.0, 0.0])

    # world initialization
    world = World(9.81, wind, seabed)
    win_width = world_width * 3
    win_height = world_height * 3
    drawer = Drawer(win_width, win_height, world_width, world_height)
    drawer.debug = True
    drawer.draw_map(seabed)
    drawer.draw_axis()

    # boats initialization
    boats: list[Boat] = []

    for i in range(boats_n):

        ## sensor intialization
        anemometer = Anemometer(RelativeError(0.05), AbsoluteError(np.pi/180))
        speedometer_par = Speedometer(MixedError(0.01, 5), offset_angle=0)
        speedometer_per = Speedometer(MixedError(0.01, 5), offset_angle=-np.pi/2)
        compass = Compass(AbsoluteError(3*np.pi/180))
        gnss = GNSS(AbsoluteError(1.5), AbsoluteError(1.5))
        sonar = Sonar(RelativeError(0.01))

        # actuators initialization
        rudder_controller = StepperController(Stepper(100, 0.3), PID(0.5, 0, 0), np.pi * 0.25)
        wing_controller = StepperController(Stepper(100, 0.3), PID(0.5, 0, 0))
        motor_controller = MotorController(Motor(1000, 0.85, 1024))

        ## boat initialization
        boat = Boat(100, 5, SeabedBoatMap(seabed), Wing(15, wing_controller), Rudder(rudder_controller), motor_controller, gnss, compass, anemometer, speedometer_par, speedometer_per, sonar, EKF())
        boat.velocity = np.zeros(2)
        boat.heading = polar_to_cartesian(1, 0)

        # boat ekf setup
        ekf_constants = boat.mass, boat.length, boat.friction_mu, boat.drag_damping, boat.wing.area, wind.density, world.gravity_z, boat.motor_controller.motor.efficiency

        boat.ekf.set_constants(ekf_constants)
        
        boats.append(boat)

    # fleet initialization
    fleet = Fleet(boats, seabed, prob_of_connection=prob_of_connection)

    # boat as key, targets as value
    targets_dict = create_targets_from_map(seabed, boats, boats_per_group_n)
    targets_idx = {}
    for b in boats:
        uuid = str(b.uuid)
        # set the initial position of the boat to the first target 
        target = targets_dict[uuid][0]
        b.position = target
        b.ekf.set_initial_state(b.measure_state())
        b.ekf.set_initial_state_variance(b.get_state_variance())
        # and update the index of the current target to the next
        targets_idx[uuid] = 1

    update_gnss = False
    update_compass = False
    metrics = GlobalMetrics(boats)

    for i in range(int(time_experiment/dt)):
        time_elapsed = round(i * dt, 2)

        # update targets
        end=False
        for b in boats:
            uuid = str(b.uuid)
            boat_target = targets_dict[uuid][targets_idx[uuid]]

            if check_intersection_circle_circle(b.position, b.length * 0.5, boat_target, 5):
                targets_idx[uuid] += 1
                if targets_idx[uuid] >= len(targets_dict[uuid]):
                    #targets_idx[uuid] = 0
                    end=True
                    break
                b.set_target(targets_dict[uuid][targets_idx[uuid]])
        if end:
            break

        # update wind conditions
        wind_derivative_mag = value_from_gaussian(2*(1-compute_magnitude(wind.velocity)/15), 6)
        wind_derivative_angle = compute_angle(wind.velocity) + value_from_gaussian(0, 0.8)
        wind_derivative = polar_to_cartesian(wind_derivative_mag, wind_derivative_angle)
        wind.velocity += wind_derivative * dt

        # update sensors and sync
        update_gnss = is_multiple(time_elapsed, dt_gnss)
        update_compass = is_multiple(time_elapsed, dt_compass)
        if is_multiple(time_elapsed, dt_sonar):
            fleet.measure_sonars()
        if is_multiple(time_elapsed, dt_sync):
            fleet.sync_boat_measures(debug=False)

        # setup actuators
        fleet.follow_targets(world.wind, dt)

        # compute state estimation
        if is_multiple(time_elapsed, dt_ekf):
            fleet.update_filtered_states(world.wind.velocity, dt, update_gnss, update_compass, prob_gnss, prob_compass, time_elapsed, metrics)
        
        # update world
        world.update(boats, dt)

        # update metrics
        for b in boats:
            metrics.get_metrics(b.uuid).add_state(time_elapsed, filtered=b.get_filtered_state(), truth=b.get_state(), covariance=b.ekf.P)
            metrics.get_metrics(b.uuid).add_motor_on(time_elapsed, b.motor_controller.get_power() > 0)

        # update drawing
        drawer.clear()
        for b in boats:
            drawer.draw_boat(b)
            if b.target is not None:
                drawer.draw_target(b.target)
        drawer.draw_wind(world.wind, np.array([world_width * 0.3, world_height * 0.3]))

        # plt.pause(0.01)
    
    Logger.info('Experiment ended in ' + str(time_elapsed) + ' seconds')
    dir='../saved_metrics/test_'+datetime.now().strftime("%Y_%m_%d__%H_%M_%S")+'/'
    os.mkdir(dir)
    metrics.plot_metrics(save_path=dir)
    fleet.plot_boat_maps(save_path=dir, plot=False)

if __name__ == '__main__':
    world_width = 200
    world_height = 200
    dt = 0.1
    time_experiment = 1000
    boats_n = 3
    boats_per_group_n = 1
    prob_of_connection = 0.8
    prob_gnss = 0.9
    prob_compass = 0.9
    dt_gnss = 20
    dt_compass = 10
    dt_sonar = 1
    dt_sync = 10
    dt_ekf = 0.1

    experiment(world_width, world_height, dt, time_experiment, boats_n, boats_per_group_n, prob_of_connection, prob_gnss, prob_compass, dt_gnss, dt_compass, dt_sonar, dt_sync, dt_ekf)