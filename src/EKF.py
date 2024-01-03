import numpy as np
from entities import World, Boat, Wind
from sensor import Anemometer, Speedometer, Compass, GNSS, AbsoluteError, RelativeError, MixedError
from utils import *

class EKF:
    def __init__(self, world: World, dt):
        self.world = world
        self.dt = dt
        self.x = np.array([world.boat.position[0], world.boat.position[1], compute_angle(world.boat.heading)]).T
        self.P = np.diag([world.boat.gnss.err_position_x.get_sigma(self.x[0])**2, world.boat.gnss.err_position_y.get_sigma(self.x[1])**2, world.boat.compass.err_direction.get_sigma(self.x[2])**2])

    def get_filtered_state(self, update=True):
        
        ## INIT MATRICES
        
        # Measurement matrix
        H = np.array([[1, 0, 0],
                      [0, 1, 0],
                      [0, 0, 1]])
        # Map from speedometer, anemometer and rudder steps to distance due to velocity, acceleration and new angle (A)
        k_acc = 0.5 * self.world.boat.drag_coeff * self.world.wind.density * self.world.boat.wing.area / self.world.boat.mass
        k_dir = self.world.boat.angular_damping
        A = np.array([[self.dt, 0, 0],
                    [0, k_acc*0.5*(self.dt**2), 0],
                    [0, 0, k_dir*self.dt]])
        

        ## PROPRIOCEPTIVE MEASUREMENTS

        # Get proprioceptive measurements and compute dynamics (u_dt)
        boat_vel = self.world.boat.measure_speedometer()
        wind_vel, wind_dir = self.world.boat.measure_anemometer(self.world.wind)
        rudder_dir = self.world.boat.measure_rudder()
        sensor_meas = np.array([boat_vel, (wind_vel*np.cos(wind_dir))**2, rudder_dir * boat_vel]).T
        u_dt = A @ sensor_meas

        # State transition matrix
        F_q = np.array([[np.cos(self.x[2]), np.cos(self.x[2]), 0],
                        [np.sin(self.x[2]), np.sin(self.x[2]), 0],
                        [0, 0, 1]])
        
        # Process noise covariance matrix (velocity, acceleration, rudder steps) (It varies for each measurement!)
        speedometer_var = self.world.boat.speedometer.err_velocity.get_sigma(boat_vel)**2
        anemometer_var = self.world.boat.anemometer.err_velocity.get_sigma(wind_vel)**2
        anemometerdir_var = self.world.boat.anemometer.err_direction.get_sigma(wind_dir)**2
        rudder_var = self.world.boat.rudder.controller.stepper.get_sigma()**2
        #Q = np.diag([speedometer_var, anemometer_var, rudder_var])
        Q = np.diag([speedometer_var, 4*(wind_vel**2)*(np.cos(wind_dir)**2)*((np.cos(wind_dir)**2)*anemometer_var-(wind_vel**2)*(np.sin(wind_dir)**2)*anemometerdir_var), (boat_vel**2) * rudder_var + (rudder_dir**2) * speedometer_var])


        # Jacobian of partial derivatives of the state transition matrix
        Ad = np.array([[1, 0, -np.sin(self.x[2])*(u_dt[0]+u_dt[1])],
                       [0, 1, np.cos(self.x[2])*(u_dt[0]+u_dt[1])],
                       [0, 0, 1]])

        ## PREDICTION STEP
        x_pred = self.x + F_q @ u_dt
        P_pred = Ad @ self.P @ Ad.T + F_q @ (A @ Q @ A.T) @ F_q.T


        ## UPDATE STEP

        if update:

            ## EXTEROCEPTIVE MEASUREMENTS

            # Measurement vector
            z = np.append(self.world.boat.measure_gnss(), self.world.boat.measure_compass())
            # Measurement noise covariance matrix
            R = np.diag([self.world.boat.gnss.err_position_x.get_sigma(z[0])**2, self.world.boat.gnss.err_position_y.get_sigma(z[1])**2, self.world.boat.compass.err_direction.get_sigma(z[2])**2])

            z_pred = x_pred.T
            K = P_pred @ H.T @ np.linalg.inv(H @ P_pred @ H.T + R)
            self.x = x_pred + K @ (z - z_pred)
            self.P = (np.eye(3) - K @ H) @ P_pred
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
    err_velocity = RelativeError(0.05)
    err_direction = AbsoluteError(np.pi/180)
    anemometer = Anemometer(err_velocity=err_velocity, err_direction=err_direction)
    err_velocity = MixedError(0.01, 5)
    speedometer = Speedometer(err_velocity=err_velocity)
    err_direction = AbsoluteError(3*np.pi/180)
    compass = Compass(err_direction=err_direction)
    err_position_x = AbsoluteError(1.5)
    err_position_y = AbsoluteError(1.5)
    gnss = GNSS(err_position_x=err_position_x, err_position_y=err_position_y)

    ## BOAT WORLD INITIALIZATION
    wind = Wind(1.291)
    rudder_controller = StepperController(Stepper(100, 1), PID(1, 0, 1), limits = (-np.pi * 0.25, np.pi * 0.25))
    wing_controller = StepperController(Stepper(100, 1), PID(1, 0.1, 1))
    boat = Boat(100, Wing(15, wing_controller), Rudder(rudder_controller), gnss, compass, anemometer, speedometer)
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
        update=np.random.rand() < probability_of_update
        x, P = ekf.get_filtered_state(update)
        t = boat.position
        t = np.append(t, compute_angle(boat.heading))
        err_x.append(x[0] - t[0])
        err_y.append(x[1] - t[1])
        err_theta.append(x[2] - t[2])
        cov_x.append(P[0,0])
        cov_y.append(P[1,1])
        cov_theta.append(P[2,2])
        time.append(i*dt)
        if update:
            print(logger.colors.OKGREEN, "Estimated:", x[0], "(", P.diagonal(), ")", " - Truth:", t[0], logger.colors.ENDC)
        else:
            print(logger.colors.ORANGE, "Estimated:", x[0], "(", P.diagonal(), ")", " - Truth:", t[0], logger.colors.ENDC)

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
    test_ekf(0.01)