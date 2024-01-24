import numpy as np
import matplotlib.pyplot as plt
from tools.utils import *
from entities import Boat

class Metrics:
    def __init__(self) -> None:
        # TIME
        self.time = []
        # STATES
        self.state_x = []
        self.state_y = []
        self.state_theta = []
        self.filtered_x = []
        self.filtered_y = []
        self.filtered_theta = []
        self.error_x = []
        self.error_y = []
        self.error_theta = []
        self.cov_x = []
        self.cov_y = []
        self.cov_theta = []
        # UPDATES
        self.updates_gnss = []
        self.updates_compass = []
        # ACTUATORS
        self.motor_on = []

    def add_state(self, time, filtered: np.ndarray, truth: np.ndarray, covariance: np.ndarray) -> None:
        if filtered.shape != (3,) or truth.shape != (3,) or covariance.shape != (3,3):
            raise Exception('Error input parameters')
        self.time.append(time)
        self.filtered_x.append(filtered[0])
        self.filtered_y.append(filtered[1])
        self.filtered_theta.append(filtered[2])
        self.state_x.append(truth[0])
        self.state_y.append(truth[1])
        self.state_theta.append(truth[2])
        self.error_x.append(filtered[0] - truth[0])
        self.error_y.append(filtered[1] - truth[1])
        self.error_theta.append(modpi(filtered[2] - truth[2]))
        self.cov_x.append(covariance[0,0])
        self.cov_y.append(covariance[1,1])
        self.cov_theta.append(covariance[2,2])
    
    def add_update(self, time, update_gnss: bool, update_compass: bool) -> None:
        if update_gnss:
            self.updates_gnss.append(time)
        if update_compass:
            self.updates_compass.append(time)
    
    def add_motor_on(self, time, motor_on=True) -> None:
        if motor_on:
            self.motor_on.append(time)

    def plot_metrics(self, dx=1800, dy=900, dpi=100, save_name=None, plot=True) -> None:
        plt.figure(figsize=(dx/dpi, dy/dpi), dpi=dpi)
        mngr = plt.get_current_fig_manager()
        mngr.window.setGeometry(int((1920-dx)/2), int((1080-dy)/2), dx, dy)

        plt.subplot(2, 2, 1)
        plt.title('Errors Position')
        for up in self.updates_gnss:
            plt.axvline(x=up, linestyle='--', linewidth=0.3, color='k')
        for mo in self.motor_on:
            plt.axvline(x=mo, linestyle='--', linewidth=0.1, color='g')
        plt.plot(self.time, self.error_x, label='Error X')
        plt.plot(self.time, self.error_y, label='Error Y')
        plt.legend()

        plt.subplot(2, 2, 2)
        plt.title('Variance Position')
        plt.plot(self.time, self.cov_x, label='Variance X')
        plt.plot(self.time, self.cov_y, label='Variance Y')
        plt.legend()

        plt.subplot(2, 2, 3)
        plt.title('Error Direction')
        for ud in self.updates_compass:
            plt.axvline(x=ud, linestyle='--', linewidth=0.3, color='k')
        plt.plot(self.time, self.error_theta, label='Error Theta')
        plt.legend()

        plt.subplot(2, 2, 4)
        plt.title('Variance Direction')
        plt.plot(self.time, self.cov_theta, label='Variance Theta')
        plt.legend()

        plt.tight_layout()
        if save_name is not None:
            try:
                plt.savefig(save_name)
            except:
                print('Error saving metrics')
        if plot:
            plt.show()

class GlobalMetrics:
    def __init__(self, boats) -> None:
        self.metrics = {}
        for boat in boats:
            self.metrics[str(boat.uuid)] = Metrics()
    
    def get_metrics(self, uuid) -> Metrics:
        return self.metrics[str(uuid)]
    
    def plot_metrics(self, dx=1800, dy=900, dpi=100, save_path=None) -> None:
        for metric in self.metrics:
            self.metrics[metric].plot_metrics(dx, dy, dpi, save_path+"boat_"+metric+".png", plot=False)