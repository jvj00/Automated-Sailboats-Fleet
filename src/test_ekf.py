from actuators.motor import Motor
from actuators.stepper import Stepper
from controllers.motor_controller import MotorController
from controllers.stepper_controller import StepperController
from estimation_algs.ekf import EKF
from entities.boat import Boat
from entities.wind import Wind
from entities.world import World
from entities.environment import SeabedMap
from errors.absolute_error import AbsoluteError
from errors.mixed_error import MixedError
from errors.relative_error import RelativeError
from controllers.pid import PID
from sensors.anemometer import Anemometer
from sensors.compass import Compass
from sensors.gnss import GNSS
from sensors.sonar import Sonar
from sensors.speedometer import Speedometer
from components.rudder import Rudder
from components.wing import Wing
import tools.logger as logger
import numpy as np
from tools.utils import *

from tools.utils import compute_angle, polar_to_cartesian

def test_ekf(dt=0.5, total_time=1000, gnss_every_sec=10, gnss_prob=1, compass_every_sec=5, compass_prob=1, print_debug=False, plot=True):
    
    steps_to_gnss = int(gnss_every_sec / dt)
    steps_to_compass = int(compass_every_sec / dt)

    ## sensor intialization
    anemometer = Anemometer(RelativeError(0.05), AbsoluteError(np.pi/180))
    speedometer_par = Speedometer(MixedError(0.01, 5), offset_angle=0)
    speedometer_perp = Speedometer(MixedError(0.01, 5), offset_angle=-np.pi/2)
    compass = Compass(AbsoluteError(3*np.pi/180))
    gnss = GNSS(AbsoluteError(1.5), AbsoluteError(1.5))
    sonar = Sonar(RelativeError(0.01))

    # actuators initialization
    rudder_controller = StepperController(Stepper(100, 0.1), PID(0.5, 0, 0), np.pi * 0.15)
    wing_controller = StepperController(Stepper(100, 0.3), PID(0.5, 0, 0))
    motor_controller = MotorController(Motor(1000, 0.85, 1024))

    # seadbed initialization
    seabed = SeabedMap(0,0,0,0)

    ## boat initialization    
    boat = Boat(100, 10, seabed, Wing(15, wing_controller), Rudder(rudder_controller), motor_controller, gnss, compass, anemometer, speedometer_par, speedometer_perp, sonar, EKF())
    boat.position = np.array([0.0, 0.0])
    boat.velocity = np.array([0.0, 0.0])
    boat.heading = polar_to_cartesian(1, 0)

    ## wind initialization
    wind = Wind(1.291)
    wind.velocity = np.array([10.0, -10.0])
    
    # world initialization
    world = World(9.81, wind, seabed)

    # boat ekf setup
    ekf_constants = boat.mass, boat.length, boat.drag_damping, boat.wing.area, wind.density, boat.motor_controller.motor.efficiency

    boat.ekf.set_initial_state(boat.measure_state())
    boat.ekf.set_initial_state_variance(boat.get_state_variance())
    boat.ekf.set_constants(ekf_constants)

    from tools.disegnino import Drawer
    drawer = Drawer(800, 800, 400, 400)

    # PLOT VARS
    err_x = []
    err_y = []
    err_theta = []
    cov_x = []
    cov_y = []
    cov_theta = []
    time = []
    updates_pos = []
    updates_dir = []
    motor_on = []

    np.set_printoptions(suppress=True)
    for i in range(int(total_time/dt)):
    
        update_gnss = np.random.rand() < gnss_prob and i % steps_to_gnss == 0
        update_compass = np.random.rand() < compass_prob and i % steps_to_compass == 0
        if (i*dt)%60 == 0:
            boat.set_target(np.array([np.random.randint(-200, 200), np.random.randint(-200, 200)]))
        boat.follow_target(world.wind, dt)
        try:
            x, P = boat.update_filtered_state(dt, update_gnss, update_compass)
        except Exception as e:
            print(e)
            continue
        
        world.update([boat], dt)
        
        t = boat.get_state()
        err_x.append(x[0] - t[0])
        err_y.append(x[1] - t[1])
        err_theta.append(modpi(x[2] - t[2]))
        cov_x.append(P[0,0])
        cov_y.append(P[1,1])
        cov_theta.append(P[2,2])
        time.append(i*dt)
        if boat.motor_controller.get_power() > 0:
            motor_on.append(i*dt)
        
        drawer.clear()
        drawer.draw_boat(boat)
        drawer.draw_wind(world.wind)
        if boat.target is not None:
            drawer.draw_target(boat.target)
        drawer.draw_axis()

        color = logger.colors.ORANGE
        prefix = 'PROCESS'
        if update_compass and update_gnss:
            color = logger.colors.OKGREEN
            prefix = 'UPDATE BOTH'
            updates_pos.append(i*dt)
            updates_dir.append(i*dt)
        elif update_compass:
            color = logger.colors.OKBLUE
            prefix = 'UPDATE COMPASS'
            updates_dir.append(i*dt)
        elif update_gnss: 
            color = logger.colors.OKCYAN
            prefix = 'UPDATE GNSS'
            updates_pos.append(i*dt)

        if print_debug:
            logger.custom_print(prefix, f'(X: {x[0]} -> {t[0]}\tY: {x[1]} -> {t[1]}\tTH: {x[2]} -> {t[2]})', color)




    if plot:
        import matplotlib.pyplot as plt
        dx = 1800
        dy = 900
        dpi = 100
        plt.figure(figsize=(dx/dpi, dy/dpi), dpi=dpi)
        mngr = plt.get_current_fig_manager()
        mngr.window.setGeometry(int((1920-dx)/2), int((1080-dy)/2), dx, dy)
        plt.subplot(2, 2, 1)
        plt.title('Errors Position')
        for up in updates_pos:
            plt.axvline(x=up, linestyle='--', linewidth=0.3, color='k')
        for mo in motor_on:
            plt.axvline(x=mo, linestyle='--', linewidth=0.1, color='g')
        plt.plot(time, err_x, label='Error X')
        plt.plot(time, err_y, label='Error Y')
        plt.legend()
        plt.subplot(2, 2, 2)
        plt.title('Variance Position')
        plt.plot(time, cov_x, label='Variance X')
        plt.plot(time, cov_y, label='Variance Y')
        plt.legend()
        plt.subplot(2, 2, 3)
        plt.title('Error Direction')
        for ud in updates_dir:
            plt.axvline(x=ud, linestyle='--', linewidth=0.3, color='k')
        plt.plot(time, err_theta, label='Error Theta')
        plt.legend()
        plt.subplot(2, 2, 4)
        plt.title('Variance Direction')
        plt.plot(time, cov_theta, label='Variance Theta')
        plt.legend()
        plt.tight_layout()
        plt.show()

if __name__ == '__main__':
    test_ekf(0.5, 1000, 60, 0.9, 10, 0.9, print_debug=False, plot=True)