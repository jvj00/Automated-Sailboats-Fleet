# generate random Gaussian values
from random import seed
from random import gauss
from datetime import datetime

def value_from_gaussian(mu, sigma):
    return gauss(mu, sigma)

def now():
    return datetime.now().strftime("%H:%M:%S")