import numpy as np
from entities import World, Boat, Wind, compute_acceleration, compute_drag_coeff, compute_friction_force, direction_local_to_world, project_wind_to_boat
from logger import Logger, colors, custom_print
from sensor import Anemometer, Speedometer, Compass, GNSS, AbsoluteError, RelativeError, MixedError
from utils import *

class EKF:
    def __init__(self, world: World, dt):
        self.world = world
        self.dt = dt
        self.x = np.array([*world.boat.measure_gnss(), world.boat.measure_compass()]).T
        self.P = np.diag(
            [
                world.boat.gnss.err_position_x.get_sigma(self.x[0])**2,
                world.boat.gnss.err_position_y.get_sigma(self.x[1])**2,
                world.boat.compass.err_angle.get_sigma(self.x[2])**2
            ]
        )

    def get_filtered_state(self, update=True):
        
        ## INIT MATRICES

        boat = self.world.boat
        wind = self.world.wind

        # Coefficient for speedometer, anemometer and rudder steps (Matrix A)
        k_friction = 1-compute_friction_force(self.world.gravity_z, boat.mass, boat.friction_mu)*self.dt
        if k_friction < 0:
            k_speed = 0
        else:
            k_speed = k_friction * self.dt

        k_acc = compute_drag_coeff(boat.drag_damping, wind.density, boat.wing.area) / boat.mass
        
        k_angspeed = self.dt / boat.length

        A = np.array(
            [
                [k_speed, 0, 0],
                [0, k_acc, 0],
                [0, 0, k_angspeed]
            ]
        )

        ## PROPRIOCEPTIVE MEASUREMENTS

        # Get proprioceptive measurements and compute dynamics (u_dt)
        boat_speed = self.world.boat.measure_speedometer()
        wind_speed, wind_angle = self.world.boat.measure_anemometer(self.world.wind)
        rudder_angle = self.world.boat.measure_rudder()
        wing_angle = self.world.boat.measure_wing()

        sensor_meas = np.array(
            [
                boat_speed,
                wind_speed ** 2 * np.cos(wing_angle - wind_angle) * np.cos(wind_angle),
                boat_speed * np.tan(rudder_angle)
            ]
        ).T
        
        u_dt = A @ sensor_meas


        # State transition matrix (F_q)
        F_q = np.array([[np.cos(self.x[2]), np.cos(self.x[2]), 0],
                        [np.sin(self.x[2]), np.sin(self.x[2]), 0],
                        [0, 0, 1]])
        
        # Process noise covariance matrix (velocity, acceleration, angular velocity) (It varies for each measurement!) (Matrix Q)
        speedometer_var = self.world.boat.speedometer.err_speed.get_sigma(boat_speed)**2
        anemometer_var = self.world.boat.anemometer.err_speed.get_sigma(wind_speed)**2
        anemometerdir_var = self.world.boat.anemometer.err_angle.get_sigma(wind_angle)**2
        rudder_var = self.world.boat.rudder.controller.stepper.get_sigma()**2
        wing_var = self.world.boat.wing.controller.stepper.get_sigma()**2
        Q = np.diag(
            [
                speedometer_var,
                (2*wind_speed*np.cos(wing_angle-wind_angle)*np.cos(wing_angle))**2 * anemometer_var  +  (wind_speed**2*(np.sin(wing_angle)*np.cos(wing_angle-wind_angle)+np.cos(wing_angle)*np.sin(wing_angle-wind_angle)))**2 * wing_var  +  (wind_speed**2*np.cos(wing_angle)*np.sin(wing_angle-wind_angle))**2 * anemometerdir_var,
                (boat_speed**2) / (np.cos(rudder_angle)**4) * rudder_var + np.tan(rudder_angle)**2 * speedometer_var
            ]
        )

        # Jacobian of partial derivatives of the state transition matrix
        Ad = np.array([[1, 0, -np.sin(self.x[2])*(u_dt[0]+u_dt[1])],
                       [0, 1, np.cos(self.x[2])*(u_dt[0]+u_dt[1])],
                       [0, 0, 1]])

        ## PREDICTION STEP
        gain = 10
        x_pred = self.x + F_q @ u_dt
        P_pred = Ad @ self.P @ Ad.T + gain * (F_q @ (A @ Q @ A.T) @ F_q.T)

        ## UPDATE STEP

        if update:

            ## EXTEROCEPTIVE MEASUREMENTS

            # Measurement matrix
            H = np.eye(3)
            # Measurement vector
            z = H @ np.append(self.world.boat.measure_gnss(), self.world.boat.measure_compass()).T
            # Measurement noise covariance matrix
            R = np.diag(
                [
                    self.world.boat.gnss.err_position_x.get_sigma(z[0])**2,
                    self.world.boat.gnss.err_position_y.get_sigma(z[1])**2,
                    self.world.boat.compass.err_angle.get_sigma(z[2])**2
                ]
            )
            # covariance of residuals
            S = H @ P_pred @ H.T + R
            # gain matrix
            W = P_pred @ H.T @ np.linalg.inv(S)
            self.x = x_pred + W @ (z - H @ x_pred)
            self.P = (np.eye(3) - W @ H) @ P_pred
        else:
            self.x = x_pred
            self.P = P_pred

        return self.x, self.P
    

def test_ekf(probability_of_update=1.0):
    from entities import Wing, Rudder, Stepper, StepperController
    from pid import PID
    import logger
    ## SENSORS INITIALIZATION
    dt = 0.5 # time step in seconds
    total_time = 1000 # total time in seconds
    err_speed = RelativeError(0.05)
    err_angle = AbsoluteError(np.pi/180)
    anemometer = Anemometer(err_speed, err_angle)
    err_speed = MixedError(0.01, 5)
    speedometer = Speedometer(err_speed)
    err_angle = AbsoluteError(3*np.pi/180)
    compass = Compass(err_angle)
    err_position_x = AbsoluteError(1.5)
    err_position_y = AbsoluteError(1.5)
    gnss = GNSS(err_position_x=err_position_x, err_position_y=err_position_y)

    ## BOAT WORLD INITIALIZATION
    wind = Wind(1.291)
    rudder_controller = StepperController(Stepper(100, 1), PID(1, 0, 1), limits = (-np.pi * 0.25, np.pi * 0.25))
    wing_controller = StepperController(Stepper(100, 1), PID(1, 0.1, 1))
    boat = Boat(100, 10, Wing(15, wing_controller), Rudder(rudder_controller), gnss, compass, anemometer, speedometer)
    world = World(9.81, wind, boat)
    world.boat.position = np.array([0.0, 0.0])
    world.boat.velocity = np.array([0.0, 0.0])
    world.boat.heading = polar_to_cartesian(1, -np.pi/4)
    world.wind.velocity = np.array([10.0, -10.0])

    ekf = EKF(world, dt)

    # PLOT VARS
    err_x = []
    err_y = []
    err_theta = []
    cov_x = []
    cov_y = []
    cov_theta = []
    time = []

    np.set_printoptions(suppress=True)
    for i in range(int(total_time/dt)):
        world.update(dt)
        update = np.random.rand() < probability_of_update
        x, P = ekf.get_filtered_state(update)
        t = np.array([*boat.position, compute_angle(boat.heading)])
        err_x.append(x[0] - t[0])
        err_y.append(x[1] - t[1])
        err_theta.append(x[2] - t[2])
        cov_x.append(P[0,0])
        cov_y.append(P[1,1])
        cov_theta.append(P[2,2])
        time.append(i*dt)
        color = colors.ORANGE
        prefix = 'PROCESS'
        if update:
            color = colors.OKGREEN
            prefix = 'UPDATE'    

        custom_print(prefix, f'(X: {x[0]} -> {t[0]}\tY: {x[1]} -> {t[1]}\tTH: {x[2]} -> {t[2]})', color)

    import matplotlib.pyplot as plt
    plt.figure(1)
    plt.title('Errors')
    plt.plot(time, err_x, label='Error X')
    plt.plot(time, err_y, label='Error Y')
    plt.plot(time, err_theta, label='Error Theta')
    plt.legend()
    plt.figure(2)
    plt.title('Variance')
    plt.plot(time, cov_x, label='Variance X')
    plt.plot(time, cov_y, label='Variance Y')
    plt.plot(time, cov_theta, label='Variance Theta')
    plt.legend()
    plt.show()

if __name__ == '__main__':
    test_ekf(0.05)