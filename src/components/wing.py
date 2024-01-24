from controllers.stepper_controller import StepperController


class Wing:
    def __init__(self, area: float, controller: StepperController):
        self.area = area
        self.controller = controller
