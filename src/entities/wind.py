import numpy as np

class Wind:
    def __init__(self, density: float):
        self.density = density
        self.velocity = np.zeros(2)