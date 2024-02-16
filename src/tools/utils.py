from random import gauss
from datetime import datetime
import numpy as np

## VECTOR OPERATIONS
def normalize(vec):
    mag = compute_magnitude(vec)
    if mag == 0:
        return None
    return vec / mag

def compute_angle(vec):
    return mod2pi(np.arctan2(vec[1], vec[0]))

def compute_magnitude(vec):
    return np.sqrt(vec[0]**2 + vec[1]**2)

def compute_distance(v1, v2):
    return compute_magnitude(v2 - v1)

def compute_angle_between(vec1, vec2):
    th1 = compute_angle(vec1)
    th2 = compute_angle(vec2)
    return mod2pi(th1 - th2)

def is_angle_between(angle, min, max):
    if max > min:
        return angle >= min and angle <= max
    else:
        return min <= angle <= 2 * np.pi or 0 <= angle <=max

def polar_to_cartesian(mag, angle):
    x = mag * np.cos(angle)
    y = mag * np.sin(angle)
    return np.array([x, y])

def cartesian_to_polar(vec):
    angle = compute_angle(vec)
    mag = compute_magnitude(vec)
    return mag, angle

# Angular Speed(ω)= Velocity / Turning Radius
# Turning radius = L / tan(th)
# where L is the lenght of the boat and th is the angle of the rudder
def compute_turning_radius(lenght, rudder_angle):
    d = np.tan(rudder_angle)
    if d == 0:
        return 0
    return lenght / d

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
    wind_relative_to_vel = wind_velocity - boat_velocity
    local_wind_speed = compute_magnitude(wind_relative_to_vel)
    local_wind_dir = mod2pi(compute_angle(wind_relative_to_vel) - compute_angle(boat_heading))
    wind_force = k * local_wind_speed**2 * np.cos(compute_angle(wing_heading)-local_wind_dir) * np.cos(compute_angle(wing_heading))
    wind_force_vec = polar_to_cartesian(wind_force, compute_angle(boat_heading))
    return wind_force_vec

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
    # fix the float approx of np.fmod.
    # if the input angle is np.pi, the resulting angle will not be 0 by a very small error 
    if np.abs(angle) < 10**(-9):
        angle = 0

    if angle < 0:
        angle += 2 * np.pi

    return angle

def modpi(angle):
    while angle > np.pi:
        angle-=2*np.pi
    while angle < -np.pi:
        angle+=2*np.pi
    return angle

def compute_a(boat_drag_damping, wing_area, wind_density, motor_efficiency, dt):
    k_drag = compute_drag_coeff(boat_drag_damping, wind_density, wing_area)
    return np.array(
        [
            [dt, 0, k_drag * 0.5 * (dt**2), motor_efficiency * 0.5 * (dt**2), 0],
            [0, dt, 0, 0, 0],
            [0, 0, 0, 0, dt]
        ]
    )

def compute_motor_thrust(motor_power, efficiency, boat_velocity, boat_heading):
    thrust_mag = efficiency * motor_power / (np.abs(np.dot(boat_velocity, boat_heading))+1)
    return boat_heading * thrust_mag

def check_intersection_circle_line(center, radius, start, end):
    delta = end - start
    a = delta[0] ** 2 + delta[1] ** 2
    b = delta[0] * (start[0] - center[0]) + delta[1] * (start[1] - center[1])
    cc = (start[0] - center[0])**2 + (start[1] - center[1])**2 - radius ** 2
    delta = b * b - a * cc;
    if delta < 0 or a == 0:
        return False
    
    delta = np.sqrt(delta)
    t1 = (-b + delta) / a;
    t2 = (-b - delta) / a;
    
    return 0 <= t1 <= 1 or 0 <= t2 <= 1

def check_intersection_circle_circle(center_a, radius_a, center_b, radius_b) -> bool:
    return compute_magnitude(center_a - center_b) < radius_a + radius_b

def angle_from_steps(steps, resolution):
    return (steps / resolution) * (2 * np.pi)

def steps_from_angle(angle, resolution):
    return np.floor((angle / (2 * np.pi)) * resolution)

def is_multiple(a, b):
    a = int(a*1000)
    b = int(b*1000)
    return a % b == 0 or (a+1) % b == 0 or (a-1) % b == 0

def random_color():
    return "#%06x" % np.random.randint(0, 0xFFFFFF)