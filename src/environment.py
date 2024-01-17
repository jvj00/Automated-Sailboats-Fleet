import numpy as np

class SeabedMap:
    def __init__(self, min_x, max_x, min_y, max_y, resolution=5) -> None:
        self.resolution = resolution
        self.min_x = min_x
        self.max_x = max_x
        self.min_y = min_y
        self.max_y = max_y
        self.seabed = None

    def create_empty_seabed(self):
        len_x = int((self.max_x - self.min_x)/self.resolution)
        len_y = int((self.max_y - self.min_y)/self.resolution)
        self.seabed = np.zeros((len_y, len_x))

    def create_seabed(self, min_z, max_z, max_slope=1, prob_go_up=0.2, plot=False):
        len_x = int((self.max_x - self.min_x)/self.resolution)
        len_y = int((self.max_y - self.min_y)/self.resolution)
        max_diff_up = max_slope * self.resolution / ((1-prob_go_up) * 2)
        max_diff_down = max_diff_up * prob_go_up
        self.seabed = np.zeros((len_y, len_x))
        reference = []
        reference.append(np.random.random() * 0.1 * (max_z - min_z) + min_z)
        for r in range(1, int(min(len_x, len_y)/2)):
            reference.append(reference[r-1] + np.random.random() * max_diff_up -  max_diff_down)
        
        for i in range(len_y):
            for j in range(len_x):
                min_dist = np.min([i, j, len_x-j-1, len_y-i-1])
                self.seabed[i][j] = reference[min_dist] + np.random.random() * max_diff_up - max_diff_down

        np.clip(self.seabed, min_z, max_z, out=self.seabed)
        
        if plot:
            import matplotlib.pyplot as plt
            fig = plt.figure()
            ax = fig.add_subplot(1, 1, 1, projection='3d')
            x, y = np.meshgrid(range(self.min_x, self.max_x, self.resolution), range(self.min_y, self.max_y, self.resolution))
            z = - self.seabed
            ax.plot_surface(x, y, z, edgecolor='royalblue', lw=0.5, rstride=8, cstride=8, alpha=0.3)
            plt.show()
    
    def get_seabed_height(self, x, y):
        if self.seabed == None:
            raise Exception('No seabed defined')
        else:
            if x < self.seabed_min_x or x > self.seabed_max_x or y < self.seabed_min_y or y > self.seabed_max_y:
                raise Exception('Boat out of the mapped seabed')
            else:
                return self.seabed[int((y - self.seabed_min_y)/self.seabed_resolution)][int((x - self.seabed_min_x)/self.seabed_resolution)]
    
    def set_seabed_height(self, x, y, z):
        if self.seabed == None:
            raise Exception('No seabed defined')
        else:
            if x < self.seabed_min_x or x > self.seabed_max_x or y < self.seabed_min_y or y > self.seabed_max_y:
                raise Exception('Boat out of the mapped seabed')
            else:
                self.seabed[int((y - self.seabed_min_y)/self.seabed_resolution)][int((x - self.seabed_min_x)/self.seabed_resolution)] = z
