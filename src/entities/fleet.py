from entities.boat import Boat
from environment import SeabedMap, SeabedBoatMap
from typing import List
import numpy as np
import matplotlib.pyplot as plt
from tools.metrics import Metrics, GlobalMetrics
from typing import Optional

class Fleet:
    def __init__(self, boats: List[Boat], seabed: SeabedMap, prob_of_connection=1):
        self.boats = boats
        self.prob_of_connection = prob_of_connection
        self.seabed = seabed

    def follow_targets(self, wind, dt):
        for boat in self.boats:
            boat.follow_target(wind, dt)
    
    def update_filtered_states(self, wind, dt, update_gnss, update_compass, prob_gnss=1, prob_compass=1, time=None, metrics=Optional[GlobalMetrics]):
        for boat in self.boats:
            try:
                if np.random.rand() >= prob_gnss:
                    update_gnss = False
                if np.random.rand() >= prob_compass:
                    update_compass = False
                if metrics is not None and time is not None:
                    metrics.get_metrics(boat.uuid).add_update(time, update_gnss, update_compass)
                boat.update_filtered_state(wind, dt, update_gnss, update_compass)
            except:
                print('ekf not available')
    
    def measure_sonars(self):
        for boat in self.boats:
            boat.measure_sonar(self.seabed, boat.ekf.x)
    
    def sync_boat_measures(self, debug=False):
        # EXCHANGE MESSAGES
        for rcv in self.boats: #receiver
            rcv.map.empty_others()
            for snd in self.boats: #sender
                if rcv != snd and np.random.rand() < self.prob_of_connection:
                    for row in range(self.seabed.len_x):
                        for col in range(self.seabed.len_y):
                            if snd.map.partial_map[row][col] != 0:
                                rcv.map.count_others[row][col] += 1
                                rcv.map.sum_others[row][col] += snd.map.partial_map[row][col]
        
        # COMPUTE RESULT
        debug_index=0
        for boat in self.boats:
            for row in range(self.seabed.len_x):
                for col in range(self.seabed.len_y):
                    # CASE ALL: data from old value, data from measurement, (possible) data from others
                    if boat.map.partial_map[row][col] !=0 and boat.map.get_measure(row, col) != 0:
                        boat.map.partial_map[row][col] = 0.5 * (1 - boat.map.count_others[row][col]/len(self.boats)) * (boat.map.partial_map[row][col] + boat.map.get_measure(row, col))
                        boat.map.partial_map[row][col] += boat.map.sum_others[row][col] / len(self.boats)
                    # CASE OLD VALUE AND OTHERS: data from old value, (possible) data from others
                    elif boat.map.partial_map[row][col] !=0:
                        boat.map.partial_map[row][col] = (1 - boat.map.count_others[row][col]/len(self.boats)) * boat.map.partial_map[row][col]
                        boat.map.partial_map[row][col] += boat.map.sum_others[row][col] / len(self.boats)
                    # CASE MEASUREMENT AND OTHERS: data from measurement, (possible) data from others
                    elif boat.map.get_measure(row, col) != 0:
                        boat.map.partial_map[row][col] = (1 - boat.map.count_others[row][col]/len(self.boats)) * boat.map.get_measure(row, col)
                        boat.map.partial_map[row][col] += boat.map.sum_others[row][col] / len(self.boats)
                    # CASE OTHERS: data only from others
                    elif boat.map.count_others[row][col] != 0:
                        boat.map.partial_map[row][col] = boat.map.sum_others[row][col] / boat.map.count_others[row][col]
            # Empty measures
            boat.map.empty_measures()
            if debug:
                error=0
                count=0
                for row in range(self.seabed.len_x):
                    for col in range(self.seabed.len_y):
                        if boat.map.partial_map[row][col] != 0:
                            error += np.abs(self.seabed.seabed[row][col] - boat.map.partial_map[row][col])
                            count += 1
                if count != 0:
                    error = round(error / count *100)/100
                print(f'Boat {debug_index}: {round(np.count_nonzero(boat.map.partial_map)*1000/boat.map.partial_map.size)/10}% of map with average error of {error} m')
                debug_index+=1
    
    def plot_boat_maps(self, save_path=None, plot=True):
        dx = 1800
        dy = 900
        dpi = 100
        if np.any([np.count_nonzero(boat.map.partial_map) for boat in self.boats]):
            fig = plt.figure(figsize=(dx/dpi, dy/dpi), dpi=dpi)
            fig.suptitle('Boat maps')
            mngr = plt.get_current_fig_manager()
            mngr.window.setGeometry(int((1920-dx)/2), int((1080-dy)/2), dx, dy)
            fig_err = plt.figure(figsize=(dx/dpi, dy/dpi), dpi=dpi)
            fig_err.suptitle('Boat maps error')
            mngr_err = plt.get_current_fig_manager()
            mngr_err.window.setGeometry(int((1920-dx)/2), int((1080-dy)/2), dx, dy)
            fig_col = int(np.ceil(np.sqrt(len(self.boats))))
            fig_row = int(np.ceil(len(self.boats)/fig_col))
            for idx, boat in enumerate(self.boats):
                if np.count_nonzero(boat.map.partial_map) != 0:
                    error=0
                    count=0
                    for row in range(self.seabed.len_x):
                        for col in range(self.seabed.len_y):
                            if boat.map.partial_map[row][col] != 0:
                                error += np.abs(self.seabed.seabed[row][col] - boat.map.partial_map[row][col])
                                count += 1
                    if count != 0:
                        error = round(error / count *100)/100
                    ax = fig.add_subplot(fig_row, fig_col, idx+1, projection='3d')
                    ax_err = fig_err.add_subplot(fig_row, fig_col, idx+1, projection='3d')
                    ax.set_title(f'Boat {idx+1}: {round(np.count_nonzero(boat.map.partial_map)*1000/boat.map.partial_map.size)/10}% of map with avg error of {error} m')
                    ax_err.set_title(f'Boat {idx+1}')
                    x, y = np.meshgrid(range(boat.map.min_x, boat.map.max_x, boat.map.resolution), range(boat.map.min_y, boat.map.max_y, boat.map.resolution))
                    ax.plot_surface(x, y, -boat.map.partial_map.T, color=(0.7,0.0,0.0,0.5), edgecolor='red', lw=0.5, rstride=1, cstride=1)
                    ax.plot_surface(x, y, -self.seabed.seabed.T, color=(0.5,0.5,0.5,0.5) ,edgecolor='grey', lw=0.5, rstride=1, cstride=1)
                    ax_err.plot_surface(x, y, np.abs(boat.map.partial_map - self.seabed.seabed).T, color=(0.0,0.7,0.0,0.5), edgecolor='green', lw=0.5, rstride=1, cstride=1)
                    ax.set_aspect('equal', adjustable='box')
                    ax_err.set_aspect('equal', adjustable='box')
            if save_path is not None:
                fig.savefig(save_path + 'boat_maps.png', dpi=dpi)
                fig_err.savefig(save_path + 'boat_maps_error.png', dpi=dpi)
            if plot:
                plt.show()
            plt.close()