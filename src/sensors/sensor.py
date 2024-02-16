from numpy import random

class Sensor:
    def __init__(self, update_probability: float = 1):
        self.value = None
        self.update_probability = update_probability
    
    def can_measure(self) -> bool:
        return random.rand() <= self.update_probability
    
    def get_value(self):
        return self.value