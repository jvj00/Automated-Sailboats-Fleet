class Error:
    def __init__(self):
        pass
    def get_sigma(self, value): # consider 3 sigma rule: encapsulate 99.7% of the values in 3 sigma
        pass
    def get_variance(self, value):
        return self.get_sigma(value) ** 2
