from matplotlib import pyplot as plt
import numpy as np
from ekf import EKF
from actuators import Motor, Stepper, StepperController, MotorController
from disegnino import Drawer
from entities import Boat, Wind, Wing, Rudder, World
from environment import SeabedMap
from pid import PID
from logger import Logger
from sensor import GNSS, AbsoluteError, Anemometer, Compass, MixedError, RelativeError, Speedometer, Sonar
from environment import SeabedMap, SeabedBoatMap
from fleet import Fleet
from utils import check_intersection_circle_circle, polar_to_cartesian

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
    win_width = world_width * 3
    win_height = world_height * 3
    drawer = Drawer(win_width, win_height, world_width, world_height)
    drawer.debug = True

    # boats initialization
    boats: list[Boat] = []
    boats_n = 2

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
        boat = Boat(100, 5, SeabedBoatMap(seabed), Wing(15, wing_controller), Rudder(rudder_controller), motor_controller, gnss, compass, anemometer, speedometer_par, speedometer_per, sonar, None, EKF())
        boat.velocity = np.zeros(2)
        boat.heading = polar_to_cartesian(1, 0)

        # boat ekf setup
        ekf_constants = boat.mass, boat.length, boat.friction_mu, boat.drag_damping, boat.wing.area, wind.density, world.gravity_z, boat.motor_controller.motor.efficiency

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
        # set the initial position of the boat to the first target 
        target = targets_dict[uuid][0]
        b.position = target
        b.ekf.set_initial_state(b.measure_state())
        b.ekf.set_initial_state_variance(b.get_state_variance())
        # and update the index of the current target to the next
        targets_idx[uuid] = 1
    
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

            if check_intersection_circle_circle(b.position, b.length * 0.5, boat_target, 5):
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
            drawer.draw_boat(b)
        drawer.draw_wind(world.wind, np.array([world_width * 0.3, world_height * 0.3]))
        if boats[0].target is not None:
            drawer.draw_target(boats[0].target)
        drawer.draw_axis()

        plt.pause(dt)
    
    plt.show()