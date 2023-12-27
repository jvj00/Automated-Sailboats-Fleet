import numpy as np
from entities import World, Boat, Wind
from sensor import Anemometer, Speedometer, Compass, GNSS, AbsoluteError, RelativeError, MixedError
from utils import *

class EKF:
    def __init__(self, world: World, dt):
        self.world = world
        self.dt = dt
        self.x = np.array([world.boat.position[0], world.boat.position[1], compute_angle(world.boat.heading)]).T
        self.P = np.diag([world.boat.gnss.err_position_x.get_sigma(self.x[0]), world.boat.gnss.err_position_y.get_sigma(self.x[1]), world.boat.compass.err_direction.get_sigma(self.x[2])**2])

    def get_filtered_state(self, update=True):
        
        ## INIT MATRICES
        
        # Measurement matrix
        H = np.array([[1, 0, 0],
                      [0, 1, 0]])
        # Map from speedometer, anemometer to velocity, acceleration (A)
        drag_coeff = 0.5
        k_acc = 0.5 * drag_coeff * self.world.wind.density * self.world.boat.wing.area / self.world.boat.mass
        A = np.array([[self.dt, 0],
                    [0, k_acc*0.5*self.dt**2]])
        

        ## PROPRIOCEPTIVE MEASUREMENTS

        # Get proprioceptive measurements and compute dynamics (u_dt)
        boat_vel = self.world.boat.measure_speedometer()
        wind_vel, wind_dir = self.world.boat.measure_anemometer(self.world.wind)
        sensor_meas = np.array([boat_vel, wind_vel*np.cos(wind_dir)**2]).T
        self.x[2] = self.world.boat.measure_compass()
        u_dt = A @ sensor_meas

        # State transition matrix
        F_q = np.array([[np.cos(self.x[2]), np.cos(self.x[2])],
                        [np.sin(self.x[2]), np.sin(self.x[2])]])
        
        # Process noise covariance matrix (velocity, acceleration) (It varies for each measurement!)
        Q = np.diag([self.world.boat.speedometer.err_velocity.get_sigma(sensor_meas[0])**2, (2*self.world.boat.anemometer.err_velocity.get_sigma(sensor_meas[1]))**2])

        # Jacobian of partial derivatives of the state transition matrix
        Ad = np.array([[1, 0, -np.sin(self.x[2])*(u_dt[0]+u_dt[1])],
                       [0, 1, np.cos(self.x[2])*(u_dt[0]+u_dt[1])],
                       [0, 0, 1]])
        

        ## PREDICTION STEP
        x_pred = self.x + np.concatenate([F_q @ u_dt, np.zeros(1)])
        P_pred = Ad @ self.P @ Ad.T + np.pad(F_q @ Q @ F_q.T, ((0, 1), (0, 1)), mode='constant', constant_values=0)


        ## UPDATE STEP

        if update:

            ## EXTEROCEPTIVE MEASUREMENTS

            # Measurement vector
            z = self.world.boat.measure_gnss()
            # Measurement noise covariance matrix
            R = np.diag([self.world.boat.gnss.err_position_x.get_sigma(z[0])**2, self.world.boat.gnss.err_position_y.get_sigma(z[1])**2])

            z_pred = np.array([x_pred[0], x_pred[1]]).T
            K = P_pred @ H.T @ np.linalg.inv(H @ P_pred @ H.T + R)
            self.x = x_pred + K @ (z - z_pred)
            self.P = (np.eye(3) - K @ H) @ P_pred
        else:
            self.x = x_pred
            self.P = P_pred

        return self.x, self.P
    

def test_ekf(probability_of_update=1.0):
    from entities import Wing, Rudder, Stepper
    from pid import PID
    import logger
    ## SENSORS INITIALIZATION
    dt = 1 # time step in seconds
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
    rudder_pid = PID(1, 0.1, 0.1, limits=(-np.pi * 0.25, np.pi * 0.25))
    wing_pid = PID(1, 0.1, 0.1, limits=(0, np.pi * 2))
    boat = Boat(100, Wing(15, Stepper(100, 1)), Rudder(Stepper(100, 1)), rudder_pid, wing_pid, gnss, compass, anemometer, speedometer)
    world = World(9.81, wind, boat)
    world.boat.position = np.array([0.0, 0.0])
    world.boat.velocity = np.array([0.0, 0.0])
    world.boat.heading = polar_to_cartesian(1, np.pi)
    world.wind.velocity = np.array([-15.0, 8.0])

    ekf = EKF(world, dt)

    np.set_printoptions(suppress=True)
    for i in range(100):
        world.update(dt)
        update=np.random.rand() < probability_of_update
        x, P = ekf.get_filtered_state(update)
        if update:
            print(logger.colors.OKGREEN, "Estimated:", x[0], "(", P.diagonal(), ")", " - Truth:", boat.position[0], logger.colors.ENDC)
        else:
            print(logger.colors.ORANGE, "Estimated:", x[0], "(", P.diagonal(), ")", " - Truth:", boat.position[0], logger.colors.ENDC)

if __name__ == '__main__':
    test_ekf(0.05)