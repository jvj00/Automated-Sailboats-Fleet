import numpy as np
from entities import World, Boat, Wind, compute_acceleration, compute_drag_coeff, compute_friction_force, direction_local_to_world, project_wind_to_boat
from logger import Logger, colors, custom_print
from sensor import Anemometer, Speedometer, Compass, GNSS, AbsoluteError, RelativeError, MixedError
from utils import *

class EKF:
    def __init__(self, boat: Boat, world: World):
        self.boat = boat
        self.wind = world.wind
        self.gravity = world.gravity_z
        self.x = self.boat.get_state()
        self.P = self.boat.get_state_variance()

    def get_filtered_state(self,  dt, update_gnss=True, update_compass=True):
        
        ## INIT MATRICES
        a_dt = compute_a(
            self.gravity,
            self.boat.mass,
            self.boat.friction_mu,
            self.boat.drag_damping,
            self.boat.wing.area,
            self.wind.density,
            dt
        )
        
        ## PROPRIOCEPTIVE MEASUREMENTS
        boat_speed = self.boat.measure_speedometer()
        wind_speed, wind_angle = self.boat.measure_anemometer(self.wind)
        rudder_angle = self.boat.measure_rudder()
        wing_angle = self.boat.measure_wing()

        sensor_meas = np.array(
            [
                boat_speed,
                wind_speed ** 2 * np.cos(wing_angle - wind_angle) * np.cos(wing_angle) / self.boat.mass,
                boat_speed * np.tan(rudder_angle) / self.boat.length
            ]
        ).T
        
        u_dt = a_dt @ sensor_meas

        # State transition matrix (F_q)
        F_q = np.array([[np.cos(self.x[2]), np.cos(self.x[2]), 0],
                        [np.sin(self.x[2]), np.sin(self.x[2]), 0],
                        [0, 0, 1]])
        
        # Process noise covariance matrix (velocity, acceleration, angular velocity) (It varies for each measurement!) (Matrix Q)
        speedometer_var = self.boat.speedometer.err_speed.get_sigma(boat_speed)**2
        anemometer_var = self.boat.anemometer.err_speed.get_sigma(wind_speed)**2
        anemometerdir_var = self.boat.anemometer.err_angle.get_sigma(wind_angle)**2
        rudder_var = self.boat.rudder.controller.stepper.get_sigma()**2
        wing_var = self.boat.wing.controller.stepper.get_sigma()**2
        
        Q = np.diag(
            [
                speedometer_var,
                (1/self.boat.mass**2) * ((2*wind_speed*np.cos(wind_angle-wing_angle)*np.cos(wind_angle))**2 * anemometer_var  +  (wind_speed**2*(np.sin(wind_angle)*np.cos(wind_angle-wing_angle)+np.cos(wind_angle)*np.sin(wind_angle-wing_angle)))**2 * wing_var  +  (wind_speed**2*np.cos(wind_angle)*np.sin(wind_angle-wing_angle))**2 * anemometerdir_var),
                (1/self.boat.length**2) * ((boat_speed**2) / (np.cos(rudder_angle)**4) * rudder_var + np.tan(rudder_angle)**2 * speedometer_var)
            ]
        )

        # Jacobian of partial derivatives of the state transition matrix
        Ad = np.array([[1, 0, -np.sin(self.x[2])*(u_dt[0]+u_dt[1])],
                       [0, 1, np.cos(self.x[2])*(u_dt[0]+u_dt[1])],
                       [0, 0, 1]])

        ## PREDICTION STEP
        x_pred = self.x + F_q @ u_dt
        P_pred = Ad @ self.P @ Ad.T + (F_q @ (a_dt @ Q @ a_dt.T) @ F_q.T)

        ## UPDATE STEP

        if update_gnss or update_compass:

            ## EXTEROCEPTIVE MEASUREMENTS

            # Measurement matrix
            if update_gnss and update_compass:
                H = np.eye(3)
            elif update_gnss:
                H = np.array([[1, 0, 0],
                              [0, 1, 0],
                              [0, 0, 0]])
            elif update_compass:
                H = np.array([[0, 0, 0],
                              [0, 0, 0],
                              [0, 0, 1]])
            # Measurement vector
            z = H @ np.append(
                self.boat.measure_gnss(),
                self.boat.measure_compass()
            ).T
            # Measurement noise covariance matrix
            R = np.diag(
                [
                    self.boat.gnss.err_position_x.get_sigma(z[0])**2,
                    self.boat.gnss.err_position_y.get_sigma(z[1])**2,
                    self.boat.compass.err_angle.get_sigma(z[2])**2
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
    

def test_ekf(dt=0.5, total_time=1000, gnss_every_sec=10, gnss_prob=1, compass_every_sec=5, compass_prob=1, print_debug=False, plot=True):
    from entities import Wing, Rudder, Stepper, StepperController
    from pid import PID
    import logger
    ## SENSORS INITIALIZATION
    steps_to_gnss = int(gnss_every_sec / dt)
    steps_to_compass = int(compass_every_sec / dt)
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
    wind.velocity = np.array([10.0, -10.0])
    rudder_controller = StepperController(Stepper(100, 1), PID(1, 0, 1), limits = (-np.pi * 0.25, np.pi * 0.25))
    wing_controller = StepperController(Stepper(100, 1), PID(1, 0.1, 1))
    boat = Boat(100, 10, Wing(15, wing_controller), Rudder(rudder_controller), gnss, compass, anemometer, speedometer)
    boat.position = np.array([0.0, 0.0])
    boat.velocity = np.array([0.0, 0.0])
    boat.heading = polar_to_cartesian(1, -np.pi/4)
    
    world = World(9.81, wind)

    boats = []
    boats.append(boat)
    
    ekf = EKF(boats[0], world)

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

    np.set_printoptions(suppress=True)
    for i in range(int(total_time/dt)):
        world.update(boats, dt)
        update_gnss = np.random.rand() < gnss_prob and i % steps_to_gnss == 0
        update_compass = np.random.rand() < compass_prob and i % steps_to_compass == 0
        x, P = ekf.get_filtered_state(dt, update_gnss, update_compass)
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
        if update_compass and update_gnss:
            color = colors.OKGREEN
            prefix = 'UPDATE BOTH'
            updates_pos.append(i*dt)
            updates_dir.append(i*dt)
        elif update_compass:
            color = colors.OKBLUE
            prefix = 'UPDATE COMPASS'
            updates_dir.append(i*dt)
        elif update_gnss: 
            color = colors.OKCYAN
            prefix = 'UPDATE GNSS'
            updates_pos.append(i*dt)

        if print_debug:
            custom_print(prefix, f'(X: {x[0]} -> {t[0]}\tY: {x[1]} -> {t[1]}\tTH: {x[2]} -> {t[2]})', color)

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
    test_ekf(0.5, 1000, 60, 0.9, 10, 0.9)