from typing import Optional
from controllers.motor_controller import MotorController
from estimation_algs.ekf import EKF
from entities.rigid_body import RigidBody
from entities.wind import Wind

from entities.environment import SeabedBoatMap, SeabedMap
from sensors.anemometer import Anemometer
from sensors.compass import Compass
from sensors.gnss import GNSS
from sensors.sonar import Sonar
from sensors.speedometer import Speedometer
from components.rudder import Rudder
from components.wing import Wing
from tools.logger import Logger

import numpy as np

from tools.utils import cartesian_to_polar, compute_acceleration, compute_angle, compute_angle_between, compute_magnitude, compute_motor_thrust, compute_turning_radius, compute_wind_force, is_angle_between, mod2pi, polar_to_cartesian

class Boat(RigidBody):
    def __init__(
            self,
            mass,
            length,
            map: Optional[SeabedBoatMap] = None,
            wing: Optional[Wing] = None,
            rudder: Optional[Rudder] = None,
            motor_controller: Optional[MotorController] = None,
            gnss: Optional[GNSS] = None,
            compass: Optional[Compass] = None,
            anemometer: Optional[Anemometer] = None,
            speedometer_par: Optional[Speedometer] = None,
            speedometer_perp: Optional[Speedometer] = None,
            sonar: Optional[Sonar] = None,
            ekf: Optional[EKF] = None):

        super().__init__(mass)
        self.length = length
        self.drag_damping = 0.2
        self.target = None
        self.map = map
        self.motor_controller = motor_controller
        self.trigger_motor = False

        if gnss is None:
            Logger.warning('No GNSS sensor provided')
        self.gnss = gnss

        if compass is None:
            Logger.warning('No compass sensor provided')
        self.compass = compass

        if anemometer is None:
            Logger.warning('No anemometer sensor provided')
        self.anemometer = anemometer

        if speedometer_par is None:
            Logger.warning('No parallel speedometer sensor provided')
        self.speedometer_par = speedometer_par

        if speedometer_perp is None:
            Logger.warning('No perpendicular speedometer sensor provided')
        self.speedometer_perp = speedometer_perp

        if sonar is None:
            Logger.warning('No sonar sensor provided')
        self.sonar = sonar

        if wing is None:
            Logger.warning('No wing provided')
        self.wing = wing        

        if rudder is None:
            Logger.warning('No rudder provided')
        self.rudder = rudder

        if ekf is None:
            Logger.warning('No EKF provided')
        
        if map is None:
            Logger.warning('No map provided')
        
        self.ekf = ekf

    def update_filtered_state(self, dt, update_gnss, update_compass):
        boat_sensors = None
        
        if self.speedometer_par is None or \
            self.speedometer_perp is None or \
            self.anemometer is None or \
            self.rudder is None or \
            self.wing is None or \
            self.motor_controller is None or \
            self.gnss is None or \
            self.compass is None:
                
                raise Exception('Not enough sensors to update the ekf')
        
        boat_sensors = self.speedometer_par, self.speedometer_perp, self.anemometer, self.gnss, self.compass, self.rudder, self.wing, self.motor_controller

        return self.ekf.compute_filtered_state(boat_sensors, dt, update_gnss, update_compass)
    
    def get_filtered_state(self):
        return self.ekf.x

    def get_state(self):
        return np.array(
            [
                *self.position,
                compute_angle(self.heading)
            ]
        ).T
    
    def measure_state(self):
        return np.array(
            [
                *self.measure_gnss(),
                self.measure_compass()
            ]
        ).T

    def get_state_variance(self):
        state = self.measure_state()
        return np.diag(
            [
                self.gnss.err_position_x.get_variance(state[0]),
                self.gnss.err_position_y.get_variance(state[1]),
                self.compass.err_angle.get_variance(state[2])
            ]
        )
    
    # compute the rotation rate
    # the rotation rate is directly proportional to the rudder angle and the boat velocity.
    # the higher is the rudder angle and the boat velocity, the higher will be the rotation rate of the boat
    # the result is scaled is using an angular damping
    # angular_speed = boat_speed / (boat_length / np.tan(rudder_angle)) = (boat_speed * np.tan(rudder_angle)) / boat_length
    def rotate(self, dt):
        if self.rudder is not None:
            turning_radius = compute_turning_radius(self.length, self.rudder.controller.get_angle())
            self.angular_speed = 0 if turning_radius == 0 else np.dot(self.velocity, self.heading) / turning_radius
        super().rotate(dt)
    
    def translate(self, dt):
        super().translate(dt)
    
    def move(self, dt):
        self.translate(dt)
        self.rotate(dt)
    
    def apply_forces(self, wind, dt):
        motor_force = 0
        wind_force = 0
        
        if self.motor_controller is not None:
            motor_force = compute_motor_thrust(self.motor_controller.power, self.motor_controller.motor.efficiency, self.velocity, self.heading)
        
        if self.wing is not None:
            wind_force = compute_wind_force(
                wind.velocity,
                wind.density,
                self.velocity,
                self.heading,
                polar_to_cartesian(1, self.wing.controller.get_angle()),
                self.wing.area,
                self.drag_damping
            )
        
        tot_force = motor_force + wind_force
        self.acceleration = compute_acceleration(tot_force, self.mass)


    def set_target(self, target):
        self.target = target
    
    def get_data(self, wind: Wind, simulated: False, measured: False, filtered: False):
        boat_position = None
        boat_velocity = None
        boat_heading = None
        wind_velocity = None

        if simulated:
            boat_position = self.position
            boat_heading = self.heading
            boat_velocity = self.velocity
            # convert wind data (absolute) to relative to the boat
            wind_velocity = wind.velocity - self.velocity
            wind_speed, wind_angle = cartesian_to_polar(wind_velocity)
            wind_angle = mod2pi(wind_angle - compute_angle(self.heading))
            wind_velocity = polar_to_cartesian(wind_speed, wind_angle)
        
        elif measured:
            boat_position = self.gnss.get_value()
            
            if self.compass.get_value() is not None:
                boat_heading = polar_to_cartesian(1, self.compass.get_value())
            
            if self.speedometer_par.get_value() is not None and self.speedometer_perp.get_value() is not None:
                boat_velocity = np.array([self.speedometer_par.get_value(), self.speedometer_perp.get_value()])
            
            if self.anemometer.get_value() is not None:
                wind_speed, wind_angle = self.anemometer.get_value()
                wind_velocity = polar_to_cartesian(wind_speed, wind_angle)
        
        elif filtered:
            if self.get_filtered_state() is not None:
                state = self.get_filtered_state()
                boat_position = np.array([state[0], state[1]])
                boat_heading = polar_to_cartesian(1, state[2])
                
            else:
                boat_position = self.gnss.get_value()
                if self.compass.get_value() is not None:
                    boat_heading = polar_to_cartesian(1, self.compass.get_value())
            
            if self.speedometer_par.get_value() is not None and self.speedometer_perp.get_value() is not None:
                boat_velocity = np.array([self.speedometer_par.get_value(), self.speedometer_perp.get_value()])
        
            if self.anemometer.get_value() is not None:
                wind_speed, wind_angle = self.anemometer.get_value()
                wind_velocity = polar_to_cartesian(wind_speed, wind_angle)
        
        if boat_position is None or boat_velocity is None or boat_heading is None or wind_velocity is None:
            return None
        
        return boat_position, boat_velocity, boat_heading, wind_velocity


    # enable simulation data to use boat and wind data from the simulation
    # enable measured_data to use boat and wind measured data
    # set both to False to use filtered data coming from the kalman filter, if available
    def follow_target(self, wind: Wind, dt, simulated_data = False, measured_data = False, filtered_data = False, motor_only = False):
        if self.target is None:
            return
        
        if self.get_data(wind, simulated_data, measured_data, filtered_data) is None:
            Logger.error('Cannot follow target due to missing values')
            return

        boat_position, boat_velocity, boat_heading, wind_velocity = self.get_data(wind, simulated_data, measured_data, filtered_data)

        boat_angle = compute_angle(boat_heading)
        wind_angle = compute_angle(wind_velocity)

        if self.rudder is not None:
            # set the angle of rudder equal to the angle between the direction of the boat and
            # the target point
            filtered_heading = polar_to_cartesian(1, boat_angle)
            target_direction = self.target - boat_position
            angle_from_target = mod2pi(-compute_angle_between(filtered_heading, target_direction))
            self.rudder.controller.set_target(angle_from_target)
            self.rudder.controller.move(dt)
        
        # if the boat is upwind (controvento), switch to motor mode
        # in this case, in order to reduce the wing thrust as much as possible,
        # the wing must be placed parallel to the wind
        if is_angle_between(wind_angle, np.pi * 1/2, np.pi * 3/2):
            self.trigger_motor = True
        elif is_angle_between(wind_angle, 0, np.pi * 1/3) or is_angle_between(wind_angle, np.pi * 5/3, np.pi * 2):
            self.trigger_motor = False
        
        not_enough_wind = compute_magnitude(boat_velocity + wind.velocity) < 1
        
        self.trigger_motor |= not_enough_wind

        if self.trigger_motor or motor_only:

            if self.wing is not None:
                wing_angle = self.wing.controller.get_angle()
                if np.abs(wing_angle - mod2pi(wind_angle + np.pi * 0.5)) > np.abs(wing_angle - mod2pi(wind_angle - np.pi * 0.5)):
                    wing_angle = mod2pi(wind_angle - np.pi * 0.5)
                    # print("POS")
                else:
                    wing_angle = mod2pi(wind_angle + np.pi * 0.5)
                    # print("NEG")
                self.wing.controller.set_target(wing_angle)
            
            if self.motor_controller is not None:
                self.motor_controller.set_power(self.motor_controller.motor.max_power)
        else:
            if self.wing is not None:
                boat_w = 0.5
                wing_angle = mod2pi(wind_angle * boat_w)
                self.wing.controller.set_target(wing_angle)
            if self.motor_controller is not None:
                self.motor_controller.set_power(0)
    
        if self.wing is not None:
            self.wing.controller.move(dt)

    def measure_anemometer(self, wind: Wind, mult_var_speed: float = 1.0, mult_var_dir: float = 1.0) -> tuple[float, float]:
        return self.anemometer.measure(wind.velocity, self.velocity, self.heading, mult_var_speed, mult_var_dir)
    
    def measure_speedometer_par(self, mult_var: float = 1.0) -> float:
        return self.speedometer_par.measure(self.velocity, self.heading, mult_var)
    
    def measure_speedometer_perp(self, mult_var: float = 1.0) -> float:
        return self.speedometer_perp.measure(self.velocity, self.heading, mult_var)
    
    def measure_compass(self, mult_var: float = 1.0) -> float:
        return self.compass.measure(self.heading, mult_var)
    
    def measure_gnss(self, mult_var_x: float = 1.0, mult_var_y: float = 1.0) -> np.ndarray:
        return self.gnss.measure(self.position, mult_var_x, mult_var_y)

    def measure_sonar(self, seabed: SeabedMap) -> float:
        try:
            value = self.sonar.measure(seabed.get_seabed_height(self.position[0], self.position[1]))
            self.map.insert_measure(self.get_filtered_state()[0], self.get_filtered_state()[1], value)
            return value
        except Exception as e:
            Logger.warning(e)
            meas = 0
        return meas
    
    def reset(self):
        super().reset()
        self.rudder.controller.set_angle(0)
        self.wing.controller.set_angle(0)
        self.motor_controller.set_power(0)
        self.ekf.set_initial_state(self.measure_state())
        self.ekf.set_initial_state_variance(self.get_state_variance())