# generate random Gaussian values
from random import seed
from random import gauss

def value_from_gaussian(mu, sigma):
    return gauss(mu, sigma)
