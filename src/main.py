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
from utils import polar_to_cartesian

if __name__ == '__main__':
    
    world_width = 450
    world_height = 250
    # seadbed initialization
    seabed = SeabedMap(int(-world_height*0.5),int(world_width*0.5),int(-world_height*0.5),int(world_height*0.5), resolution=25)
    seabed.create_seabed(20, 200, max_slope=2, prob_go_up=0.1, plot=False)

    ## wind initialization
    wind = Wind(1.291)
    wind.velocity = np.array([10.0, -10.0])
    
    # world initialization
    world = World(9.81, wind, seabed)
    win_width = world_width * 2
    win_height = world_height * 2
    drawer = Drawer(win_width, win_height, world_width, world_height)
    drawer.debug = True

    # boats initialization
    boats: list[Boat] = []
    boats_n = 4

    boats_starting_point = np.array([0.0, 0.0])

    for i in range(boats_n):

        ## sensor intialization
        anemometer = Anemometer(RelativeError(0.05), AbsoluteError(np.pi/180))
        speedometer_par = Speedometer(MixedError(0.01, 5))
        speedometer_per = Speedometer(MixedError(0.01, 5))
        compass = Compass(AbsoluteError(3*np.pi/180))
        gnss = GNSS(AbsoluteError(1.5), AbsoluteError(1.5))
        sonar = Sonar(RelativeError(0.01))

        # actuators initialization
        rudder_controller = StepperController(Stepper(100, 0.3), PID(0.5, 0, 0), np.pi * 0.25)
        wing_controller = StepperController(Stepper(100, 0.3), PID(0.5, 0, 0))
        motor_controller = MotorController(Motor(200, 0.85, 1024))

        boat_position = boats_starting_point + np.array([i * 10, i * 10])
        ## boat initialization
        boat = Boat(40, 10, SeabedBoatMap(seabed), Wing(15, wing_controller), Rudder(rudder_controller), motor_controller, gnss, compass, anemometer, speedometer_par, speedometer_per, sonar, EKF())
        boat.position = np.array(boat_position)
        boat.velocity = np.array([0.0, 0.0])
        boat.heading = polar_to_cartesian(1, -np.pi/4)

        # boat ekf setup
        ekf_constants = boat.mass, boat.length, boat.friction_mu, boat.drag_damping, boat.wing.area, wind.density, world.gravity_z, boat.motor_controller.motor.efficiency

        # boat.ekf.set_initial_state(boat.get_state())
        # boat.ekf.set_initial_state_variance(boat.get_state_variance())
        # boat.ekf.set_constants(ekf_constants)
        
        boats.append(boat)

    # fleet initialization
    fleet = Fleet(boats, seabed, prob_of_connection=0.8)
    velocities = []
    wind_velocities = []
    positions = []
    anemo_meas = []
    anemo_truth = []
    times = []

    dt = 0.2
    
    update_gnss = False
    update_compass = False

    for time_elapsed in np.arange(0, 300, dt):
        # if time_elapsed == 5:
        #     world.wind.velocity = -world.wind.velocity

        if time_elapsed % 10 == 0:
            update_gnss = True
            update_compass = True
            fleet.sync_boat_measures(debug=True)
        else:
            update_gnss = False
            update_compass = False
        
        if time_elapsed % 30 == 0:
            x_pos = np.random.uniform(-0.5, 0.5)
            y_pos = np.random.uniform(-0.5, 0.5)
            for b in boats:
                b.set_target(np.array([world_width * x_pos, world_height * y_pos]))

        # velocities.append(boats[0].velocity.copy())
        # wind_velocities.append(world.wind.velocity.copy())

        # times.append(time_elapsed)
                
        for b in boats:
            try:
                x, P = b.update_filtered_state(world.wind.velocity, dt, update_gnss, update_compass)
                print(np.abs(b.get_state()-x))
            except:
                pass
                #print('ekf not available')

            b.measure_sonar(seabed, b.position) # WARNING: INSERT THEN FILTERED POSITION, NOT TRUTH
            b.follow_target(world.wind, dt, simulated_data=True)
        
        world.update(boats, dt)

        # update drawing
        drawer.clear()
        for b in boats:
            drawer.draw_boat(b)
        drawer.draw_wind(world.wind, np.array([world_width * 0.3, world_height * 0.3]))
        if boats[0].target is not None:
            drawer.draw_target(boats[0].target)
        drawer.draw_axis()

        # Logger.debug(f'Wind velocity: {world.wind.velocity}')
        # Logger.debug(f'Boat velocity: {world.boat.velocity}')
        # Logger.debug(f'Boat heading: {world.boat.heading}')

        #Plot anemometer measurements
        # plt.figure(1)
        # plt.cla()
        # plt.plot(times, list(map(lambda p: p[0], velocities)), label='Boat Velocity X')
        # plt.plot(times, list(map(lambda p: p[1], velocities)), label='Boat Velocity Y')
        # plt.plot(times, list(map(lambda p: p[0], wind_velocities)), label='Wind Velocity X')
        # plt.plot(times, list(map(lambda p: p[1], wind_velocities)), label='Wind Velocity Y')

        # plt.legend()
        plt.pause(dt)
    
    plt.show()