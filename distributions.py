import random as rng
import numpy as np

def constant(x, return_ks= False, number_of_samples = 0):
    if return_ks:
        result = []
        for i in range(number_of_samples):
            result.append(x)
        return(result)
    return(x[0])

def from_range(limits, return_ks= False, number_of_samples = 0):
    if return_ks:
        result = list(np.linspace(limits[0], limits[1], number_of_samples))
        return(result)
    return(rng.random()*(limits[1]-limits[0])+limits[0])
