import json
import os
from tools.logger import Logger
from errors.error import Error
class SimulationConfig:
    def __init__(
            self, 
            world_width = 500,
            world_height = 500,
            duration = 1000,
            update_period = 0.1,
            boats = 2,
            groups = 1,
            random_target_enabled = False,
            real_time_charts_enabled = False,
            real_time_drawings_enabled = False,
            save_folder = ""
    ):
        
        self.world_width = world_width
        self.world_height = world_height
        self.duration = duration
        self.update_period = update_period
        self.boats = boats
        self.groups = groups
        self.random_target_enabled = random_target_enabled
        self.real_time_charts_enabled = real_time_charts_enabled
        self.real_time_drawings_enabled = real_time_drawings_enabled
        self.save_folder = save_folder

class FiltersConfig:
    def __init__(self, ekf_update_period = 0.1, fleet_update_period = 1, fleet_update_probability = 1):
        self.ekf_update_period = ekf_update_period
        self.fleet_update_period = fleet_update_period
        self.fleet_update_probability = fleet_update_probability

class SensorConfig:
    def __init__(self, error = Error(), update_period = 1, update_probability = 1):
        self.error = error
        self.update_period = update_period
        self.update_probability = update_probability

class SensorsConfig:
    def __init__(self, gnss: SensorConfig, compass: SensorConfig, anemometer: SensorConfig, speedometer: SensorConfig, sonar: SensorConfig):
        self.gnss = gnss
        self.compass = compass
        self.anemometer = anemometer
        self.speedometer =  speedometer
        self.sonar =  sonar

class ActuatorConfig:
    def __init__(self, resolution = 1):
        self.resolution = resolution

class ActuatorsConfig:
    def __init__(self, wing: ActuatorConfig, rudder: ActuatorConfig, motor: ActuatorConfig):
        self.wing = wing
        self.rudder = rudder
        self.motor = motor

class Config:
    def __init__(
            self,
            simulation: SimulationConfig = None,
            filters: FiltersConfig = None,
            sensors: SensorsConfig = None,
            actuators: ActuatorsConfig = None
    ):
        self.simulation = simulation if simulation is not None else SimulationConfig()
        self.filters = filters if filters is not None else FiltersConfig()
        self.sensors = sensors if sensors is not None else SensorsConfig(SensorConfig(), SensorConfig(), SensorConfig(), SensorConfig(), SensorConfig())
        self.actuators = actuators if actuators is not None else ActuatorsConfig(ActuatorConfig(), ActuatorConfig(), ActuatorConfig())

    def load(self, path):
        with open(path) as json_file:
            data = json.load(json_file)
            simulation = data['simulation']
            
            self.simulation.world_height = simulation['world']['height']
            self.simulation.world_width = simulation['world']['width']

            self.simulation.duration = simulation['duration']
            self.simulation.update_period = simulation['update_period']
            self.simulation.boats = simulation['boats']
            self.simulation.groups = simulation['groups']
            self.simulation.real_time_charts_enabled = simulation['real_time_charts_enabled']
            self.simulation.real_time_drawings_enabled = simulation['real_time_drawings_enabled']
            self.simulation.random_target_enabled = simulation['random_target_enabled']
            self.simulation.save_folder = simulation['save_folder']

            sensors = data['sensors']
            self.sensors.anemometer.error = (sensors['anemometer']['error']['speed']['relative'], sensors['anemometer']['error']['angle']['absolute'])
            self.sensors.anemometer.update_period = sensors['anemometer']['update_period']
            self.sensors.anemometer.update_probability = sensors['anemometer']['update_probability']

            self.sensors.gnss.error = (sensors['gnss']['error']['x']['absolute'], sensors['gnss']['error']['y']['absolute'])
            self.sensors.gnss.update_period = sensors['gnss']['update_period']
            self.sensors.gnss.update_probability = sensors['gnss']['update_probability']

            self.sensors.compass.error = sensors['compass']['error']['absolute']
            self.sensors.compass.update_period = sensors['compass']['update_period']
            self.sensors.compass.update_probability = sensors['compass']['update_probability']

            self.sensors.speedometer.error = (sensors['speedometer']['error']['relative'], sensors['speedometer']['error']['threshold'])
            self.sensors.speedometer.update_period = sensors['speedometer']['update_period']
            self.sensors.speedometer.update_probability = sensors['speedometer']['update_probability']
            
            self.sensors.sonar.error = sensors['sonar']['error']['relative']
            self.sensors.sonar.update_period = sensors['sonar']['update_period']
            self.sensors.sonar.update_probability = sensors['sonar']['update_probability']

            actuators = data['actuators']
            self.actuators.motor.resolution = actuators['motor']['resolution']
            self.actuators.wing.resolution = actuators['wing']['resolution']
            self.actuators.rudder.resolution = actuators['rudder']['resolution']

            filters = data['filters']
            self.filters.ekf_update_period = filters['ekf']['update_period']
            self.filters.fleet_update_period = filters['fleet']['update_period']
            self.filters.fleet_update_probability = filters['fleet']['update_probability']