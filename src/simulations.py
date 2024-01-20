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

def simulation_1():

    targets = [np.array([50, 20]), np.array([100, -20]), np.array([150, 20]), np.array([200, -20])]
    current_target_idx = 0
    target_radius = 5
    route = []
    time = []
    rudder_angle = []
    rudder_computed_angle = []
    rudder_speed = []
    wing_angle = []

    wind.velocity = np.array([10.0, 0.0])

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
        target_direction = boat.target - boat.position
        angle_from_target = mod2pi(-compute_angle_between(boat.heading, target_direction))
        world.update([boat], dt)

        route.append(np.copy(boat.position))
        time.append(time_elapsed)
        if boat.rudder is not None:
            rudder_angle.append(modpi(boat.rudder.controller.get_angle()))
            rudder_computed_angle.append(modpi(angle_from_target))
            rudder_speed.append(boat.rudder.controller.speed)
        if boat.wing is not None:
            wing_angle.append(boat.wing.controller.get_angle())

    drawer.draw_axis()
    drawer.draw_route(route, 'blue')
    for t in targets:
        drawer.draw_target(t)
    drawer.draw_wind(world.wind, np.array([world_width * 0.4, world_height * 0.4]))

    # plt.plot(time, list(map(lambda p: p[0], route)), label='Boat Velocity X')
    # plt.plot(time, list(map(lambda p: p[1], route)), label='Boat Velocity Y')
    if len(time) == len(rudder_angle):
        plt.plot(time, rudder_angle, label='Rudder angle')
        plt.plot(time, rudder_computed_angle, label='Rudder computed angle')
        plt.plot(time, rudder_speed, label='Rudder speed')
    if len(time) == len(wing_angle):
        plt.plot(time, wing_angle, label='Wing angle')
    plt.legend()
    plt.show()

    while True:
        pass

def simulation_2():

    targets = [np.array([50, 50]), np.array([100, -50]), np.array([150, 50]), np.array([200, -50])]
    current_target_idx = 0
    target_radius = 5
    route = []
    time = []
    rudder_angle = []
    rudder_computed_angle = []
    wing_angle = []

    wind.velocity = np.array([10.0, 0.0])

    boat.position = np.array([0.0, 0.0])
    boat.velocity = np.array([0.0, 0.0])
    boat.heading = polar_to_cartesian(1, 0)
    boat.wing = None

    dt = 0.1

    boat.set_target(targets[current_target_idx])

    for time_elapsed in np.arange(0, 300, dt):
        if compute_distance(targets[current_target_idx], boat.position) <= target_radius:
            current_target_idx += 1
            if current_target_idx >= len(targets):
                break
            boat.set_target(targets[current_target_idx])
        boat.follow_target(world.wind, dt, simulated_data=True, motor_only=True)
        target_direction = boat.target - boat.position
        angle_from_target = mod2pi(-compute_angle_between(boat.heading, target_direction))
        world.update([boat], dt)

        route.append(np.copy(boat.position))
        time.append(time_elapsed)
        if boat.rudder is not None:
            rudder_angle.append(boat.rudder.controller.get_angle())
            rudder_computed_angle.append(angle_from_target)
        if boat.wing is not None:
            wing_angle.append(boat.wing.controller.get_angle())

    drawer.draw_axis()
    drawer.draw_route(route, 'blue')
    for t in targets:
        drawer.draw_target(t)
    drawer.draw_wind(world.wind, np.array([world_width * 0.4, world_height * 0.4]))

    # plt.plot(time, list(map(lambda p: p[0], route)), label='Boat Velocity X')
    # plt.plot(time, list(map(lambda p: p[1], route)), label='Boat Velocity Y')
    if len(time) == len(rudder_angle):
        plt.plot(time, rudder_angle, label='Rudder angle')
        plt.plot(time, rudder_computed_angle, label='Rudder computed angle')
    if len(time) == len(wing_angle):
        plt.plot(time, wing_angle, label='Wing angle')
    plt.legend()
    plt.show()

def simulation_3():

    targets = [np.array([50, 50]), np.array([-100, -50]), np.array([150, 50]), np.array([-200, -50])]
    current_target_idx = 0
    target_radius = 5
    route = []
    time = []
    rudder_angle = []
    rudder_computed_angle = []
    wing_angle = []

    wind.velocity = np.array([10.0, 0.0])

    boat.position = np.array([0.0, 0.0])
    boat.velocity = np.array([0.0, 0.0])
    boat.heading = polar_to_cartesian(1, 0)
    boat.wing = None

    dt = 0.1

    boat.set_target(targets[current_target_idx])

    for time_elapsed in np.arange(0, 500, dt):
        if compute_distance(targets[current_target_idx], boat.position) <= target_radius:
            current_target_idx += 1
            if current_target_idx >= len(targets):
                break
            boat.set_target(targets[current_target_idx])
        boat.follow_target(world.wind, dt, simulated_data=True, motor_only=True)
        target_direction = boat.target - boat.position
        angle_from_target = mod2pi(-compute_angle_between(boat.heading, target_direction))
        world.update([boat], dt)

        route.append(np.copy(boat.position))
        time.append(time_elapsed)
        if boat.rudder is not None:
            rudder_angle.append(modpi(boat.rudder.controller.get_angle()))
            rudder_computed_angle.append(modpi(angle_from_target))
        if boat.wing is not None:
            wing_angle.append(boat.wing.controller.get_angle())

    drawer.draw_axis()
    drawer.draw_route(route, 'blue')
    for t in targets:
        drawer.draw_target(t)
    drawer.draw_wind(world.wind, np.array([world_width * 0.4, world_height * 0.4]))

    # plt.plot(time, list(map(lambda p: p[0], route)), label='Boat Velocity X')
    # plt.plot(time, list(map(lambda p: p[1], route)), label='Boat Velocity Y')
    if len(time) == len(rudder_angle):
        plt.plot(time, rudder_angle, label='Rudder angle')
        plt.plot(time, rudder_computed_angle, label='Rudder computed angle')
    if len(time) == len(wing_angle):
        plt.plot(time, wing_angle, label='Wing angle')
    plt.legend()
    plt.show()

if __name__ == '__main__':
    simulation_1()
    # simulation_2()
    simulation_3()
    