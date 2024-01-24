import numpy as np
import matplotlib.pyplot as plt

class SeabedMap:
    def __init__(self, min_x, max_x, min_y, max_y, resolution=5) -> None:
        self.resolution = resolution
        self.min_x = min_x
        self.max_x = max_x
        self.min_y = min_y
        self.max_y = max_y
        self.len_x = int((self.max_x - self.min_x)/self.resolution)
        self.len_y = int((self.max_y - self.min_y)/self.resolution)
        self.seabed = []

    def create_seabed(self, min_z, max_z, max_slope=1, prob_go_up=0.2, plot=False):
        max_diff_up = max_slope * self.resolution / ((1-prob_go_up) * 2)
        max_diff_down = max_diff_up * prob_go_up
        self.seabed = np.zeros((self.len_x, self.len_y))
        reference = []
        reference.append(np.random.random() * 0.1 * (max_z - min_z) + min_z)
        for r in range(1, int(min(self.len_x, self.len_y)/2)):
            reference.append(reference[r-1] + (np.random.random() * max_diff_up -  max_diff_down)*np.sqrt(2))
        for i in range(self.len_x):
            for j in range(self.len_y):
                min_dist = np.min([i, j, self.len_y-j-1, self.len_x-i-1])
                self.seabed[i][j] = reference[min_dist] + np.random.random() * max_diff_up - max_diff_down

        np.clip(self.seabed, min_z, max_z, out=self.seabed)
        
        if plot:
            fig = plt.figure()
            ax = fig.add_subplot(1, 1, 1, projection='3d')
            x, y = np.meshgrid(range(self.min_x, self.max_x, self.resolution), range(self.min_y, self.max_y, self.resolution))
            z = - self.seabed.T
            ax.plot_surface(x, y, z, edgecolor='royalblue', lw=0.5, rstride=1, cstride=1, alpha=0.3)
            plt.axis('equal')
            plt.show()
    
    def get_seabed_height(self, x, y):
        if len(self.seabed) == 0:
            raise Exception('No seabed defined')
        else:
            if x < self.min_x or x > self.max_x or y < self.min_y or y > self.max_y:
                raise Exception('Boat out of the mapped seabed')
            else:
                return self.seabed[int((x - self.min_x)/self.resolution)][int((y - self.min_y)/self.resolution)]
    
    def set_seabed_height(self, x, y, z):
        if len(self.seabed) == 0:
            raise Exception('No seabed defined')
        else:
            if x < self.min_x or x > self.max_x or y < self.min_y or y > self.max_y:
                raise Exception('Boat out of the mapped seabed')
            else:
                self.seabed[int((x - self.min_x)/self.resolution)][int((y - self.min_y)/self.resolution)] = z


class SeabedBoatMap:
    def __init__(self, map_like: SeabedMap) -> None:
        self.resolution = map_like.resolution
        self.min_x = map_like.min_x
        self.max_x = map_like.max_x
        self.min_y = map_like.min_y
        self.max_y = map_like.max_y
        self.len_x = map_like.len_x
        self.len_y = map_like.len_y

        self.partial_map = np.zeros((self.len_x, self.len_y))
        self.count_others = np.zeros((self.len_x, self.len_y))
        self.sum_others = np.zeros((self.len_x, self.len_y))
        self.meas = []
    
    def insert_measure(self, x, y, meas):
        if x < self.min_x or x > self.max_x or y < self.min_y or y > self.max_y:
            raise Exception('Boat out of the mapped seabed')
        else:
            row = int((x - self.min_x)/self.resolution)
            col = int((y - self.min_y)/self.resolution)
            if self.get_measure(row, col) == 0:
                self.meas.append((row, col, meas))
                return True
            else:
                return False

    def get_measure(self, row, col):
        for m in self.meas:
            if m[0] == row and m[1] == col:
                return m[2]
        return 0
    
    def empty_measures(self):
        self.meas = []
    
    def empty_others(self):
        self.count_others = np.zeros((self.len_x, self.len_y))
        self.sum_others = np.zeros((self.len_x, self.len_y))