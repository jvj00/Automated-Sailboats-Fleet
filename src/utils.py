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