# seadbed initialization
from actuators import Motor, MotorController, Rudder, Stepper, StepperController, Wing
from disegnino import Drawer
from entities import Boat, Wind, World
from environment import SeabedMap
import numpy as np
import matplotlib.pyplot as plt

from pid import PID
from utils import compute_angle_between, compute_distance, mod2pi, modpi, polar_to_cartesian

seabed = SeabedMap(0,0,0,0)
## wind initialization
wind = Wind(1.291)
world = World(9.81, wind, seabed)

# world initialization

win_width = 900
win_height = 500

world_width = 450
world_height = 250
drawer = Drawer(win_width, win_height, world_width, world_height)
drawer.debug = True

rudder_controller = StepperController(Stepper(100, 0.3), PID(0.5, 0, 0), np.pi * 0.25)
wing_controller = StepperController(Stepper(100, 0.3), PID(0.5, 0, 0))
motor_controller = MotorController(Motor(200, 0.85, 1024))

## boat initialization
boat = Boat(40, 10, Wing(15, wing_controller), Rudder(rudder_controller), motor_controller, seabed)

def simulation_motor_only(targets):

    current_target_idx = 0
    target_radius = 5
    route = []
    time = []
    rudder_angle = []
    rudder_speed = []

    boat.position = np.array([0.0, 0.0])
    boat.velocity = np.array([0.0, 0.0])
    boat.heading = polar_to_cartesian(1, 0)
    boat.wing = None

    dt = 0.1

    boat.set_target(targets[current_target_idx])

    for time_elapsed in np.arange(0, 100, dt):
        if compute_distance(targets[current_target_idx], boat.position) <= target_radius:
            current_target_idx += 1
            if current_target_idx >= len(targets):
                break
            boat.set_target(targets[current_target_idx])

        boat.follow_target(world.wind, dt, simulated_data=True, motor_only=True)

        world.update([boat], dt)

        route.append(np.copy(boat.position))
        time.append(time_elapsed)
        if boat.rudder is not None:
            rudder_angle.append(modpi(boat.rudder.controller.get_angle()))
            rudder_speed.append(boat.rudder.controller.speed * boat.rudder.controller.direction)

    drawer.draw_axis()
    drawer.draw_route(route, 'blue')
    for t in targets:
        drawer.draw_target(t)
    drawer.draw_wind(world.wind, np.array([world_width * 0.4, world_height * 0.4]))

    # plt.plot(time, list(map(lambda p: p[0], route)), label='Boat Velocity X')
    # plt.plot(time, list(map(lambda p: p[1], route)), label='Boat Velocity Y')
    if len(time) == len(rudder_angle):
        plt.plot(time, rudder_angle, label='Rudder angle')
        plt.plot(time, rudder_speed, label='Rudder speed')

    plt.legend()
    plt.show()

    while True:
        pass

def simulation_motor_wing(targets):

    current_target_idx = 0
    target_radius = 5
    route = []
    time = []
    rudder_angle = []
    rudder_speed = []
    wing_angle = []
    wing_speed = []

    boat.position = np.array([0.0, 0.0])
    boat.velocity = np.array([0.0, 0.0])
    boat.heading = polar_to_cartesian(1, 0)

    wind.velocity = np.array([12.0, -8.0])

    dt = 0.1

    boat.set_target(targets[current_target_idx])

    for time_elapsed in np.arange(0, 100, dt):
        if compute_distance(targets[current_target_idx], boat.position) <= target_radius:
            current_target_idx += 1
            if current_target_idx >= len(targets):
                break
            boat.set_target(targets[current_target_idx])

        boat.follow_target(world.wind, dt, simulated_data=True)

        world.update([boat], dt)

        route.append(np.copy(boat.position))
        time.append(time_elapsed)
        if boat.rudder is not None:
            rudder_angle.append(modpi(boat.rudder.controller.get_angle()))
            rudder_speed.append(boat.rudder.controller.speed * boat.rudder.controller.direction)
        if boat.wing is not None:
            wing_angle.append(modpi(boat.wing.controller.get_angle()))
            wing_speed.append(boat.wing.controller.speed * boat.wing.controller.direction)

    drawer.draw_axis()
    drawer.draw_route(route, 'blue')
    for t in targets:
        drawer.draw_target(t)
    drawer.draw_wind(world.wind, np.array([world_width * 0.4, world_height * 0.4]))

    # plt.plot(time, list(map(lambda p: p[0], route)), label='Boat Velocity X')
    # plt.plot(time, list(map(lambda p: p[1], route)), label='Boat Velocity Y')
    if len(time) == len(rudder_angle):
        plt.plot(time, rudder_angle, label='Rudder angle')
        plt.plot(time, rudder_speed, label='Rudder speed')
    
    if len(time) == len(wing_angle):
        plt.plot(time, wing_angle, label='Wing angle')
        plt.plot(time, wing_speed, label='Wing speed')
    plt.legend()
    plt.show()

    while True:
        pass

if __name__ == '__main__':
    # simulation_motor_only([np.array([50, 20]), np.array([100, -20]), np.array([150, 20]), np.array([200, -20])])
    # simulation_motor_only([np.array([50, 50]), np.array([100, -50]), np.array([150, 50]), np.array([200, -50])])
    # simulation_motor_only([np.array([50, 50]), np.array([-100, -50]), np.array([150, 50]), np.array([-200, -50])])
    
    simulation_motor_wing([np.array([50, 20]), np.array([100, -20]), np.array([150, 20]), np.array([200, -20])])
    # simulation_motor_wing([np.array([50, 50]), np.array([100, -50]), np.array([150, 50]), np.array([200, -50])])
    # simulation_motor_wing([np.array([50, 50]), np.array([-100, -50]), np.array([150, 50]), np.array([-200, -50])])
    