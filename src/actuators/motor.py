class Motor:
    # max_power: max power of the motor (in Watt)
    # efficiency: average efficiency of the motor (between 0 and 1)
    # resolution: range of values that can represent the power. The motor is ideally controlled by a PWM, so the resolution
    # is given by 2^(n. of bits of the DAC)
    def __init__(self, max_power: float, efficiency: float, resolution: int):
        self.max_power = max_power
        self.efficiency = efficiency
        self.resolution = resolution
    
    def get_error(self):
        return self.max_power / self.resolution
    
    def get_sigma(self): # consider 3 sigma rule: encapsulate 99.7% of the values in 3 sigma
        return self.get_error() / 3.0
    
    def get_variance(self):
        return self.get_sigma()**2
