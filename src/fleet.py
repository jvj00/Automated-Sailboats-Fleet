from entities import Boat
from environment import SeabedMap, SeabedBoatMap
from typing import List
import numpy as np

class Fleet:
    def __init__(self, boats: List[Boat], seabed: SeabedMap, prob_of_connection=1):
        self.boats = boats
        self.prob_of_connection = prob_of_connection
        self.seabed = seabed
    
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
                print(f'Boat {debug_index}: {round(np.count_nonzero(boat.map.partial_map)*1000/boat.map.partial_map.size)/10}% of map with avgerage error of {error} m')
                debug_index+=1