from errors.error import Error

class AbsoluteError(Error):
    def __init__(self, error):
        self.error = error
    def get_sigma(self, value):
        return self.error /3.0
    def get_variance(self, value):
        return super().get_variance(value)
