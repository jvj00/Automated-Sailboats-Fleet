import numpy as np
from entities.boat import Boat

from entities.environment import SeabedMap

# takes a list of boats and creates, for each group of boats, its route
# each group goes to specific rows
def create_targets_from_map(map: SeabedMap, boats: list[Boat], boats_per_group_n: int):
    boats_per_group = [boats[n : n + boats_per_group_n] for n in range(0, len(boats), boats_per_group_n)]
    groups_n = len(boats_per_group)

    targets_dict = {}

    x_cells = int((map.max_x - map.min_x) / map.resolution)

    rows_idx = 0

    for row in np.arange(map.min_y, map.max_y, map.resolution):
        
        row_idx = rows_idx % groups_n
        
        for col in range(map.min_x, map.max_x, map.resolution):
            center_x = col + map.resolution / 2
            center_y = row + map.resolution / 2
            target = np.array([center_x, center_y])

            for b in boats_per_group[row_idx]:
                key = str(b.uuid)
                if key not in targets_dict:
                    targets_dict[key] = [np.copy(target)]
                else:
                    targets_dict[key].append(target)

        rows_idx += 1
    
    for key in targets_dict:
        targets = targets_dict[key]
        grouped_lists = [targets[i : i + x_cells] for i in range(0, len(targets), x_cells)]

        for i in range(1, len(grouped_lists), 2):
            grouped_lists[i] = grouped_lists[i][::-1]
        
        targets_dict[key] = [x for xs in grouped_lists for x in xs]
    
    return targets_dict

def create_random_targets_from_map(seabed, boats, time_experiment):
    targets_dict = {}
    for b in boats:
        key = str(b.uuid)
        targets_dict[key] = []
        for _ in range(time_experiment):
            x = 0.9 * (np.random.rand() * (seabed.max_x - seabed.min_x) + seabed.min_x)
            y = 0.9 * (np.random.rand() * (seabed.max_y - seabed.min_y) + seabed.min_y)
            target = np.array([x, y])
            targets_dict[key].append(np.copy(target))
    return targets_dict
