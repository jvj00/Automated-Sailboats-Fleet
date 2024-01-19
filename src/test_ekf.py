from actuators import Motor, MotorController
from ekf import EKF
from entities import Boat, Wind, Wing, Rudder, Stepper, StepperController, World
from environment import SeabedMap
from pid import PID
import logger
from sensor import GNSS, AbsoluteError, Anemometer, Compass, MixedError, RelativeError, Speedometer
import numpy as np
from utils import *

from utils import compute_angle, polar_to_cartesian

def test_ekf(dt=0.5, total_time=1000, gnss_every_sec=10, gnss_prob=1, compass_every_sec=5, compass_prob=1, print_debug=False, plot=True):
    
    steps_to_gnss = int(gnss_every_sec / dt)
    steps_to_compass = int(compass_every_sec / dt)

    ## sensor intialization
    anemometer = Anemometer(RelativeError(0.05), AbsoluteError(np.pi/180))
    speedometer_par = Speedometer(MixedError(0.01, 5), offset_angle=0)
    speedometer_perp = Speedometer(MixedError(0.01, 5), offset_angle=-np.pi/2)
    compass = Compass(AbsoluteError(3*np.pi/180))
    gnss = GNSS(AbsoluteError(1.5), AbsoluteError(1.5))

    # actuators initialization
    rudder_controller = StepperController(Stepper(100, 1), PID(1, 0, 1), np.pi * 0.25)
    wing_controller = StepperController(Stepper(100, 1), PID(1, 0.1, 1))
    motor_controller = MotorController(Motor(1000, 0.85, 1024))

    # seadbed initialization
    seabed = SeabedMap(0,0,0,0)

    ## boat initialization    
    boat = Boat(100, 10, Wing(15, wing_controller), Rudder(rudder_controller), motor_controller, seabed, gnss, compass, anemometer, speedometer_par, speedometer_perp, None, EKF())
    boat.position = np.array([0.0, 0.0])
    boat.velocity = np.array([0.0, 0.0])
    boat.heading = polar_to_cartesian(1, np.pi)

    ## wind initialization
    wind = Wind(1.291)
    wind.velocity = np.array([10.0, -10.0])
    
    # world initialization
    world = World(9.81, wind, seabed)

    # boat ekf setup
    ekf_constants = boat.mass, boat.length, boat.friction_mu, boat.drag_damping, boat.wing.area, wind.density, world.gravity_z, boat.motor_controller.motor.efficiency

    boat.ekf.set_initial_state(boat.get_state())
    boat.ekf.set_initial_state_variance(boat.get_state_variance())
    boat.ekf.set_constants(ekf_constants)

    boat.set_target(np.array([-100, 0]))
    
    boats: list[Boat] = []
    boats.append(boat)

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
        boat.follow_target(world.wind, dt)
        x, P = boat.update_filtered_state(world.wind.velocity, dt, update_gnss, update_compass)
        world.update(boats, dt)
        
        t = np.array([*boat.position, compute_angle(boat.heading)])
        err_x.append(x[0] - t[0])
        err_y.append(x[1] - t[1])
        err_theta.append(modpi(x[2] - t[2]))
        cov_x.append(P[0,0])
        cov_y.append(P[1,1])
        cov_theta.append(P[2,2])
        time.append(i*dt)
        if boat.motor_controller.get_power() > 0:
            motor_on.append(i*dt)

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