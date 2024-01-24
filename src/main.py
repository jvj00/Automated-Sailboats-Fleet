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
from pid import PID
from tools.logger import Logger
from environment import SeabedMap, SeabedBoatMap
from fleet import Fleet
from tools.utils import check_intersection_circle_circle, polar_to_cartesian

# takes a list of boats and creates, for each group of boats, its route
# each group goes to specific rows
def create_targets_from_map(map: SeabedMap, boats: list[Boat], boats_per_group_n: int):
    boats_per_group = [boats[n : n + boats_per_group_n] for n in range(0, len(boats), boats_per_group_n)]
    groups_n = len(boats_per_group)

    targets_dict = {}

    x_cells = int((map.max_x - map.min_x) / map.resolution)
    y_cells = int((map.max_y - map.min_y) / map.resolution)

    rows_idx = 0

    for row in np.arange(map.min_y, map.max_y, map.resolution):
        
        row_idx = rows_idx % groups_n
        
        for col in range(map.min_x, map.max_x, map.resolution):
            center_x = col + map.resolution / 2
            center_y = row + map.resolution / 2
            target = np.array([center_x, center_y])

            for b in boats_per_group[row_idx]:
                key = str(b.uuid)
                if key not in targets_dict:
                    targets_dict[key] = [np.copy(target)]
                else:
                    targets_dict[key].append(target)

        rows_idx += 1
    
    for key in targets_dict:
        targets = targets_dict[key]
        grouped_lists = [targets[i : i + x_cells] for i in range(0, len(targets), x_cells)]

        for i in range(1, len(grouped_lists), 2):
            grouped_lists[i] = grouped_lists[i][::-1]
        
        targets_dict[key] = [x for xs in grouped_lists for x in xs]
    
    return targets_dict

if __name__ == '__main__':

    np.set_printoptions(suppress=True)
    world_width = 400
    world_height = 200
    # seadbed initialization
    seabed = SeabedMap(int(-world_width*0.3),int(world_width*0.3),int(-world_height*0.3),int(world_height*0.3), resolution=15)
    seabed.create_seabed(20, 200, max_slope=2, prob_go_up=0.1, plot=False)

    ## wind initialization
    wind = Wind(1.291)
    wind.velocity = np.array([10.0, -10.0])
    
    # world initialization
    world = World(9.81, wind, seabed)
    win_width = world_width * 4
    win_height = world_height * 4
    drawer = Drawer(win_width, win_height, world_width, world_height)
    drawer.debug = False

    # boats initialization
    boats: list[Boat] = []
    boats_n = 4

    spawn_area_x_limits = (-120, -80)
    spawn_area_y_limits = (-90, -50)

    for i in range(boats_n):

        ## sensor intialization
        anemometer = Anemometer(RelativeError(0.05), AbsoluteError(np.pi/180))
        speedometer_par = Speedometer(MixedError(0.01, 5), offset_angle=0)
        speedometer_per = Speedometer(MixedError(0.01, 5), offset_angle=-np.pi/2)
        compass = Compass(AbsoluteError(3*np.pi/180))
        gnss = GNSS(AbsoluteError(1.5), AbsoluteError(1.5))
        sonar = Sonar(RelativeError(0.01))

        # actuators initialization
        rudder_controller = StepperController(Stepper(100, 0.3), PID(0.5, 0, 0), np.pi * 0.20)
        wing_controller = StepperController(Stepper(100, 0.3), PID(0.5, 0, 0))
        motor_controller = MotorController(Motor(1000, 0.85, 1024))

        ## boat initialization
        boat = Boat(100, 5, SeabedBoatMap(seabed), Wing(15, wing_controller), Rudder(rudder_controller), motor_controller, gnss, compass, anemometer, speedometer_par, speedometer_per, sonar, EKF())
        position_x = np.random.uniform(spawn_area_x_limits[0], spawn_area_x_limits[1])
        position_y = np.random.uniform(spawn_area_y_limits[0], spawn_area_y_limits[1])
        boat.position = np.array([position_x, position_y])
        boat.velocity = np.zeros(2)
        boat.heading = polar_to_cartesian(1, 0)

        # boat ekf setup
        ekf_constants = boat.mass, boat.length, boat.friction_mu, boat.drag_damping, boat.wing.area, wind.density, world.gravity_z, boat.motor_controller.motor.efficiency
        boat.ekf.set_initial_state(boat.measure_state())
        boat.ekf.set_initial_state_variance(boat.get_state_variance())
        boat.ekf.set_constants(ekf_constants)
        
        boats.append(boat)

    # fleet initialization
    fleet = Fleet(boats, seabed, prob_of_connection=0.8)

    # boat as key, targets as value
    targets_dict = create_targets_from_map(seabed, boats, 2)

    # boat as key, current target index as value
    targets_idx = {}
    
    for b in boats:
        uuid = str(b.uuid)
        targets_idx[uuid] = 0
        b.set_target(targets_dict[uuid][0])
    
    velocities = []
    wind_velocities = []
    positions = []
    anemo_meas = []
    anemo_truth = []
    times = []

    dt = 0.1
    
    update_gnss = False
    update_compass = False

    for i in range(10000):
        time_elapsed = i * dt

        for b in boats:
            uuid = str(b.uuid)
            boat_target = targets_dict[uuid][targets_idx[uuid]]

            if check_intersection_circle_circle(b.position, b.length * 0.5, boat_target, 2):
                targets_idx[uuid] += 1
                b.set_target(targets_dict[uuid][targets_idx[uuid]])

        if time_elapsed % 10 == 0:
            update_gnss = True
            update_compass = True
            fleet.sync_boat_measures(debug=True)
            # fleet.plot_boat_maps()
        else:
            update_gnss = False
            update_compass = False

        fleet.follow_targets(world.wind, dt)

        fleet.update_filtered_states(world.wind.velocity, dt, update_gnss, update_compass)
        
        world.update(boats, dt)

        fleet.measure_sonars()

        # for idx, b in enumerate(fleet.boats):
        #     print(b.get_filtered_state()-b.get_state())

        # update drawing
        drawer.clear()
        for b in boats:
            uuid = str(b.uuid)
            drawer.draw_boat(b)
            drawer.draw_target(targets_dict[uuid][targets_idx[uuid]])
            # drawer.draw_route(targets_dict[str(b.uuid)], 'red')
        drawer.draw_wind(world.wind, np.array([world_width * 0.3, world_height * 0.3]))
        
        # drawer.draw_axis()

        plt.pause(0.01)
    
    plt.show()