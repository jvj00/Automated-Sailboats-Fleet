from errors.error import Error
import numpy as np

class MixedError(Error):
    def __init__(self, error, threshold):
        self.error = error
        self.threshold = threshold
    def get_sigma(self, value):
        return self.error * (self.threshold if np.abs(value) < self.threshold else np.abs(value)) /3.0
    def get_variance(self, value):
        return super().get_variance(value)
