import os
from datetime import datetime
from matplotlib import pyplot as plt
import numpy as np
from actuators.motor import Motor
from actuators.stepper import Stepper
from controllers.motor_controller import MotorController
from controllers.stepper_controller import StepperController
from estimation_algs.ekf import EKF
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
from components.rudder import Rudder
from components.wing import Wing
from tools.disegnino import Drawer
from entities.environment import SeabedMap
from controllers.pid import PID
from tools.logger import Logger
from entities.environment import SeabedMap, SeabedBoatMap
from estimation_algs.fleet import Fleet
from tools.utils import *
from tools.metrics import Metrics, GlobalMetrics
from main import create_targets_from_map

def create_random_targets_from_map(seabed, boats, time_experiment):
    targets_dict = {}
    for b in boats:
        key = str(b.uuid)
        targets_dict[key] = []
        for _ in range(time_experiment):
            x = 0.9 * (np.random.rand() * (seabed.max_x - seabed.min_x) + seabed.min_x)
            y = 0.9 * (np.random.rand() * (seabed.max_y - seabed.min_y) + seabed.min_y)
            target = np.array([x, y])
            targets_dict[key].append(np.copy(target))
    return targets_dict

def experiment(world_width, world_height, dt, time_experiment, boats_n, boats_per_group_n, prob_of_connection, prob_gnss, prob_compass, dt_gnss, dt_compass, dt_sonar, dt_sync, dt_ekf, rt, random_target=False):
    
    np.set_printoptions(suppress=True)

    # seadbed initialization
    world_width = int(world_width/2)
    world_height = int(world_height/2)
    seabed = SeabedMap(-world_width,world_width,-world_height,world_height, resolution=10)
    seabed.create_seabed(20, 300, max_slope=2, prob_go_up=0.1, plot=False)

    # wind initialization
    wind = Wind(1.291)
    wind_mag = np.random.random() * 15 + 5
    wind_ang = np.random.random() * 2 * np.pi
    wind.velocity = polar_to_cartesian(wind_mag, wind_ang)
    wind_derivative = np.array([0.0, 0.0])

    # world initialization
    world = World(9.81, wind, seabed)
    win_width = world_width * 10
    win_height = world_height * 10
    drawer = Drawer(win_width, win_height, world_width*3, world_height*3)
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
    if not random_target:
        targets_dict = create_targets_from_map(seabed, boats, boats_per_group_n)
    else:
        targets_dict = create_random_targets_from_map(seabed, boats, time_experiment)
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
    time_last=time_experiment
    end_boats={}
    for b in boats:
        end_boats[str(b.uuid)]=False
        uuid = str(b.uuid)
        b.set_target(targets_dict[uuid][targets_idx[uuid]])

    for i in range(int(time_experiment/dt)):
        time_elapsed = round(i * dt, 2)

        # update targets
        for b in boats:
            uuid = str(b.uuid)
            if check_intersection_circle_circle(b.position, b.length * 0.5, targets_dict[uuid][targets_idx[uuid]], 5):
                targets_idx[uuid] += 1
                if targets_idx[uuid] >= len(targets_dict[uuid]):
                    targets_idx[uuid] = 0
                    end_boats[uuid]=True
                b.set_target(targets_dict[uuid][targets_idx[uuid]])
        if [end_boats.get(uuid) for uuid in end_boats.keys()]==[True]*len(end_boats) and time_last==time_experiment: # if all boats have reached their targets
            time_last=time_elapsed
            Logger.info('All boats reached their targets at ' + str(time_elapsed) + ' seconds')
        if time_elapsed>time_last and is_multiple(time_elapsed, 1): # exchange data and collect last measures every seconds for 10 seconds after the end of the experiment
            fleet.measure_sonars()
            fleet.sync_boat_measures()
        if time_elapsed>time_last+10: # stop the experiment after 10 seconds
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
            fleet.sync_boat_measures()

        # setup actuators
        fleet.follow_targets(world.wind, dt, filtered_data=True)

        # compute state estimation
        if is_multiple(time_elapsed, dt_ekf):
            fleet.update_filtered_states(world.wind, dt, update_gnss, update_compass, prob_gnss, prob_compass, time_elapsed, metrics)
        
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

        # wait to simulate a real time execution
        if rt:
            plt.pause(dt)
    
    Logger.info('Experiment ended in ' + str(time_elapsed+dt) + ' seconds')
    dir='../saved_metrics/test_'+datetime.now().strftime("%Y_%m_%d__%H_%M_%S")+'/'
    os.mkdir(dir)
    metrics.write_metrics(save_path=dir)
    metrics.plot_metrics(save_path=None)
    # fleet.plot_boat_maps(save_path=dir, plot=False)

if __name__ == '__main__':
    world_width = 120
    world_height = 120
    time_experiment = 200
    boats_n = 1
    boats_per_group_n = 1
    prob_of_connection = 0.8
    dt = 0.1
    dt_gnss = 20
    dt_compass = 10
    dt_sonar = 1
    dt_sync = 10
    dt_ekf = 0.1
    prob_gnss = 0.9
    prob_compass = 0.9

    rt = False
    random_target = True

    experiment(world_width, world_height, dt, time_experiment, boats_n, boats_per_group_n, prob_of_connection, prob_gnss, prob_compass, dt_gnss, dt_compass, dt_sonar, dt_sync, dt_ekf, rt, random_target=random_target)