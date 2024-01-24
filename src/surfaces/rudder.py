from controllers.stepper_controller import StepperController


class Rudder:
    def __init__(self, controller: StepperController):
        self.controller = controller