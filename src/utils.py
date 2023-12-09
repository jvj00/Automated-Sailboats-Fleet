from random import seed
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

## RANDOM
def value_from_gaussian(mu, sigma):
    return gauss(mu, sigma)

## TIME
def now():
    return datetime.now().strftime("%H:%M:%S")