import numpy as np
from logger import Logger, colors, custom_print
from utils import *

class EKF:
    def __init__(self):
        self.x = None
        self.P = None
        self.constants = None
    
    def set_initial_state(self, state):
        self.x = state
    
    def set_initial_state_variance(self, variance):
        self.P = variance
    
    def set_constants(self, constants):
        self.constants = constants

    def get_filtered_state(self, boat_sensors, true_boat_data, true_wind_data, dt, update_gnss=True, update_compass=True):

        if self.x is None or self.P is None or self.constants is None:
            raise Exception('Initial state, variance or constants not provided')

        boat_mass, boat_length, boat_friction_mu, boat_drag_damping, wing_area, wind_density, gravity = self.constants
        
        ## INIT MATRICES
        # coefficient matrix
        a_dt = compute_a(
            gravity,
            boat_mass,
            boat_friction_mu,
            boat_drag_damping,
            wing_area,
            wind_density,
            dt
        )
        
        ## PROPRIOCEPTIVE MEASUREMENTS
        # retrieve boat sensors
        speedometer, anemometer, rudder, wing, gnss, compass = boat_sensors
        # retrieve true boat data (from simulation)
        true_boat_velocity, true_boat_heading, true_boat_position = true_boat_data
        true_wind_velocity = true_wind_data

        # retrieve measurements from boat sensors
        boat_speed = speedometer.measure(true_boat_velocity)
        wind_speed, wind_angle = anemometer.measure(true_wind_velocity, true_boat_velocity, true_boat_heading)
        boat_position = gnss.measure(true_boat_position)
        boat_angle = compass.measure(compute_angle(true_boat_heading))
        rudder_angle = rudder.controller.measure_angle()
        wing_angle = wing.controller.measure_angle()

        sensor_meas = np.array(
            [
                boat_speed,
                wind_speed ** 2 * np.cos(wing_angle - wind_angle) * np.cos(wing_angle) / boat_mass,
                boat_speed * np.tan(rudder_angle) / boat_length
            ]
        ).T
        
        u_dt = a_dt @ sensor_meas

        # compute variance for each sensor reading
        speedometer_var = speedometer.err_speed.get_variance(boat_speed)
        anemometer_var = anemometer.err_speed.get_variance(wind_speed)
        anemometerdir_var = anemometer.err_angle.get_variance(wind_angle)
        rudder_var = rudder.controller.stepper.get_variance()
        wing_var = wing.controller.stepper.get_variance()

        # Process noise covariance matrix (velocity, acceleration, angular velocity) (It varies for each measurement!) (Matrix Q)
        Q = np.diag(
            [
                speedometer_var,
                (1/boat_mass**2) * ((2*wind_speed*np.cos(wind_angle-wing_angle)*np.cos(wind_angle))**2 * anemometer_var  +  (wind_speed**2*(np.sin(wind_angle)*np.cos(wind_angle-wing_angle)+np.cos(wind_angle)*np.sin(wind_angle-wing_angle)))**2 * wing_var  +  (wind_speed**2*np.cos(wind_angle)*np.sin(wind_angle-wing_angle))**2 * anemometerdir_var),
                (1/boat_length**2) * ((boat_speed**2) / (np.cos(rudder_angle)**4) * rudder_var + np.tan(rudder_angle)**2 * speedometer_var)
            ]
        )

        # State transition matrix (F_q)
        F_q = np.array([[np.cos(self.x[2]), np.cos(self.x[2]), 0],
                        [np.sin(self.x[2]), np.sin(self.x[2]), 0],
                        [0, 0, 1]])
        

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
            z = H @ np.array([*boat_position, boat_angle]).T
            # Measurement noise covariance matrix
            R = np.diag(
                [
                    gnss.err_position_x.get_variance(z[0]),
                    gnss.err_position_y.get_variance(z[1]),
                    compass.err_angle.get_variance(z[2])
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
    