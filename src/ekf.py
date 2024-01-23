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

        # retrieve coefficients
        boat_mass, boat_length, boat_friction_mu, boat_drag_damping, wing_area, wind_density, gravity, motor_efficiency = self.constants
        
        ## INIT MATRICES
        # coefficient matrix
        a_dt = compute_a(
            gravity,
            boat_mass,
            boat_friction_mu,
            boat_drag_damping,
            wing_area,
            wind_density,
            motor_efficiency,
            dt
        )
        
        ## PROPRIOCEPTIVE MEASUREMENTS

        # retrieve boat sensors
        speedometer_par, speedometer_perp, anemometer, rudder, wing, motor_controller, gnss, compass = boat_sensors
        # retrieve true boat data (from simulation)
        true_boat_velocity, true_boat_heading, true_boat_position = true_boat_data
        true_wind_velocity = true_wind_data
        # retrieve measurements from boat sensors
        boat_speed_par = speedometer_par.measure(true_boat_velocity, true_boat_heading)
        boat_speed_perp = speedometer_perp.measure(true_boat_velocity, true_boat_heading)
        wind_speed, wind_angle = anemometer.measure(true_wind_velocity, true_boat_velocity, true_boat_heading)
        rudder_angle = rudder.controller.measure_angle()
        wing_angle = wing.controller.measure_angle()
        motor_power = motor_controller.measure_power()

        sensor_meas = np.array(
            [
                boat_speed_par,
                boat_speed_perp,
                wind_speed ** 2 * np.cos(wing_angle - wind_angle) * np.cos(wing_angle) / boat_mass,
                motor_power / ((np.abs(boat_speed_par)+1) * boat_mass),
                boat_speed_par * np.tan(rudder_angle) / boat_length
            ]
        ).T
        
        u_dt = a_dt @ sensor_meas

        # compute variance for each sensor reading
        speedometer_par_var = speedometer_par.err_speed.get_variance(boat_speed_par)
        speedometer_perp_var = speedometer_perp.err_speed.get_variance(boat_speed_perp)
        anemometer_var = anemometer.err_speed.get_variance(wind_speed)
        anemometerdir_var = anemometer.err_angle.get_variance(wind_angle)
        rudder_var = rudder.controller.stepper.get_variance()
        wing_var = wing.controller.stepper.get_variance()
        motor_var = motor_controller.motor.get_variance()

        # Process noise covariance matrix (velocity, acceleration, angular velocity) (It varies for each measurement!) (Matrix Q)
        Q = np.diag(
            [
                speedometer_par_var,
                speedometer_perp_var,
                (1/boat_mass**2) * ((2*wind_speed*np.cos(wind_angle-wing_angle)*np.cos(wind_angle))**2 * anemometer_var  +  (wind_speed**2*(np.sin(wind_angle)*np.cos(wind_angle-wing_angle)+np.cos(wind_angle)*np.sin(wind_angle-wing_angle)))**2 * wing_var  +  (wind_speed**2*np.cos(wind_angle)*np.sin(wind_angle-wing_angle))**2 * anemometerdir_var),
                (1/boat_mass**2) * ((1/(boat_speed_par+1)**2) * motor_var + (motor_power**2/(boat_speed_par+1)**4) * speedometer_par_var),
                (1/boat_length**2) * ((boat_speed_par**2) / (np.cos(rudder_angle)**4) * rudder_var + np.tan(rudder_angle)**2 * speedometer_par_var)
            ]
        )

        # State transition matrix (F_q)
        F_q = np.array([[np.cos(self.x[2]), np.sin(self.x[2]), 0],
                        [np.sin(self.x[2]), -np.cos(self.x[2]), 0],
                        [0, 0, 1]])
        

        # Jacobian of partial derivatives of the state transition matrix
        J = np.array([[1, 0, -np.sin(self.x[2])*u_dt[0] + np.cos(self.x[2])*u_dt[1]],
                       [0, 1, np.cos(self.x[2])*u_dt[0] + np.sin(self.x[2])*u_dt[1]],
                       [0, 0, 1]])

        ## PREDICTION STEP
        x_pred = self.x + F_q @ u_dt
        P_pred = J @ self.P @ J.T + (F_q @ (a_dt @ Q @ a_dt.T) @ F_q.T)
        x_pred[2] = mod2pi(x_pred[2])
        

        ## UPDATE STEP

        if update_gnss or update_compass:

            ## EXTEROCEPTIVE MEASUREMENTS
            boat_angle = compass.measure(true_boat_heading)
            boat_position = gnss.measure(true_boat_position)

            if boat_angle-x_pred[2] > np.pi: # because Kalman Filter's update gain doesn't matter to angles (6.27 is much bigger than 0.01) 
                x_pred[2]+=2*np.pi
            elif boat_angle-x_pred[2] < -np.pi:
                x_pred[2]-=2*np.pi

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
    