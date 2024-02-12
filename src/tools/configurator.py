import json
import os
from tools.logger import Logger

class Config:
    def __init__(self) -> None:
        self.config = False

    def load(self, path):
        #import from config file if exist
        if os.path.isfile(path):
            try:
                with open(path) as json_file:
                    data = json.load(json_file)
                    self.file = data

                    self.world_width = data['world']['width']
                    self.world_height = data['world']['height']

                    self.duration = data['experiment']['duration']
                    self.dt = data['experiment']['dt']
                    self.boats = data['experiment']['boats']
                    self.boats_per_group = data['experiment']['boats_per_group']
                    self.real_time_graphs = data['experiment']['real_time_graphs']
                    self.real_time_drawings = data['experiment']['real_time_drawings']
                    self.random_target = data['experiment']['random_target']
                    self.save_folder = data['experiment']['save_folder']

                    self.dt_gnss = data['components']['dt_gnss']
                    self.dt_compass = data['components']['dt_compass']
                    self.dt_prediction_sensors = data['components']['dt_prediction_sensors']
                    self.dt_sonar = data['components']['dt_sonar']
                    self.prob_gnss = data['components']['prob_gnss']
                    self.prob_compass = data['components']['prob_compass']
                    self.prob_of_radio_connection = data['components']['prob_of_radio_connection']

                    self.dt_sync = data['algorithms']['dt_sync']
                    self.dt_ekf = data['algorithms']['dt_ekf']

                    self.err_anemo_speed = data['error']['anemometer']['speed']['relative']
                    self.err_anemo_direction = data['error']['anemometer']['direction']['absolute']
                    self.err_speed_thresh = data['error']['speedometer']['threshold']
                    self.err_speed_rel = data['error']['speedometer']['relative']
                    self.err_compass = data['error']['compass']['absolute']
                    self.err_gnss_x = data['error']['gnss']['x']['absolute']
                    self.err_gnss_y = data['error']['gnss']['y']['absolute']
                    self.err_sonar = data['error']['sonar']['relative']
                    self.res_rudder = data['resolution']['rudder']['steps_per_revolution']
                    self.res_wing = data['resolution']['wing']['steps_per_revolution']
                    self.res_pwm_engine = data['resolution']['engine']['pwm']

                    self.config = True
            except Exception as e:
                Logger.error('Error while reading config file with error:\n'+str(e)+'\nExiting...')
                exit()
        else:
            Logger.error('No config file found at path "'+path+'".\nExiting...')
            exit()