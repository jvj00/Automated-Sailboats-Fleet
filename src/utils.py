from random import gauss
from datetime import datetime
import numpy as np

## VECTOR OPERATIONS
def normalize(vec):
    mag = compute_magnitude(vec)
    return np.array([vec[0] / mag, vec[1] / mag]) if mag != 0 else vec

def compute_angle(vec):
    return np.arctan2(vec[1], vec[0])

def compute_magnitude(vec):
    return np.sqrt(vec[0]*vec[0] + vec[1]*vec[1])

def compute_distance(v1, v2):
    return compute_magnitude(v2 - v1)

def cartesian_to_polar(vec):
    angle = compute_angle(vec)
    mag = compute_magnitude(vec)
    return mag, angle

def polar_to_cartesian(mag, angle):
    x = mag * np.cos(angle)
    y = mag * np.sin(angle)
    return np.array([x, y])

# converts the given local direction to an absolute direction (relative to the world system)
# s_direction: system direction (rotation vector)
# direction: direction to convert, relative to the system
def direction_local_to_world(s_direction, direction):
    angle_relative = compute_angle(direction)
    system_angle = compute_angle(s_direction)
    return polar_to_cartesian(1, system_angle + angle_relative)

def velocity_local_to_world(s_velocity, velocity):
    return velocity + s_velocity

def velocity_world_to_local(s_velocity, velocity):
    return velocity - s_velocity

# Angular Speed(ω)= Velocity / Turning Radius
# Turning radius = L / tan(th)
# where L is the lenght of the boat and th is the angle of the rudder
def compute_turning_radius(lenght, rudder_angle):
    d = np.tan(rudder_angle)
    if d == 0:
        return 0
    return lenght / np.tan(rudder_angle)

def compute_acceleration(force, mass):
    return force / mass

# source: ChatGPT
# see drag equation
# F drag​ = 0.5 × CD × ρ × A × (∣Vrelative∣**2)
# F wind = f_drag * (Vrelative / |Vrelative|)
# wind velocity is absolute (referenced to the ground)
# streamlined airfoils low: 0.02 - 0.05
# streamlined airfoils high: 0.2 - 0.5 - 1.0
def compute_wind_force(wind_velocity, wind_density, boat_velocity, boat_heading, wing_heading, wing_area, drag_damping: float):
    k = compute_drag_coeff(drag_damping, wind_density, wing_area)
    relative_wind_velocity = velocity_world_to_local(boat_velocity, wind_velocity)
    relative_wind_speed, relative_wind_angle = cartesian_to_polar(relative_wind_velocity)
    relative_wind_angle = compute_angle(boat_heading) - relative_wind_angle
    f_wind_local = k * (relative_wind_speed**2) * np.cos(compute_angle(wing_heading) - relative_wind_angle) * np.cos(relative_wind_angle)
    f_wind_local = polar_to_cartesian(f_wind_local, compute_angle(boat_heading))
    return f_wind_local

def compute_drag_coeff(drag_damping, wind_density, wing_area):
    return 0.5 * drag_damping * wind_density * wing_area

def compute_force(mass, acceleration):
    return mass * acceleration

def compute_friction_force(gravity, boat_mass, damping):
    return compute_force(boat_mass, gravity) * damping

def project_wind_to_boat(wind, wing_heading, boat_heading):
    wind = np.dot(wind, wing_heading) * wing_heading
    # project the projected force along the boat direction
    return np.dot(wind, boat_heading) * boat_heading

## RANDOM
def value_from_gaussian(mu, sigma):
    return gauss(mu, sigma)

## TIME
def now():
    return datetime.now().strftime("%H:%M:%S")

def mod2pi(angle):
    angle = np.fmod(angle, 2 * np.pi)

    if angle < 0:
        angle += 2 * np.pi

    return angle