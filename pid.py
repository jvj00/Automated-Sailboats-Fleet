import numpy as np

class PID:
    def __init__(self, Kp, Ki, Kd, limits=(-np.inf, np.inf)):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.setpoint = 0.0
        self.prev_error = 0.0
        self.integral = 0.0
        self.limits = limits
    
    def set_target(self, target):
        self.setpoint = target

    def compute(self, current_value, dt):
        error = self.setpoint - current_value
        self.integral += error * dt
        derivative = (error - self.prev_error) / dt

        output = self.Kp * error + self.Ki * self.integral + self.Kd * derivative

        output = max(min(output, self.limits[1]), self.limits[0])

        self.prev_error = error
        
        return output
    
    def reset(self):
        self.prev_error = 0
        self.integral = 0