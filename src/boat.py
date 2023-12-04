import numpy as np

class Boat:
    def __init__(self):
        self.mass = 100
        self.position = np.zeros(2)
        self.velocity = np.zeros(2)
        self.acceleration = np.zeros(2)
        self.heading = np.zeros(2)
        self.wing = Wing(self.heading)

class Wing:
    def __init__(self, heading):
        self.area = 15
        # heading is perpendicular to the surface of the wing, pointing forward
        self.heading = heading
