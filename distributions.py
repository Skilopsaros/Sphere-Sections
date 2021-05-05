import random as rng

def constant(x):
    return(x[0])

def from_range(low, high):
    return(rng.random()*(high-low)+low)
