import random as rng
import numpy as np

def constant(x, return_ks= False, number_of_samples = 0):
    if return_ks:
        result = []
        for i in range(number_of_samples):
            result.append(x[0])
        return(result)
    return(x[0])

def from_range(limits, return_ks= False, number_of_samples = 0):
    if return_ks:
        result = []
        print(number_of_samples)
        for i in range(int(number_of_samples)):
            result.append(rng.random()*(limits[1]-limits[0])+limits[0])
        return(result)
    return(rng.random()*(limits[1]-limits[0])+limits[0])

def normal_distribution(specifications, return_ks= False, number_of_samples = 0):
    if return_ks:
        result = []
        for i in range(int(number_of_samples)):
            new_value = normal_distribution(specifications)
            result.append(new_value)
        return(result)
    answer = np.random.normal(specifications[0],specifications[1])
    if answer < 0:
        answer = normal_distribution(specifications)
    return(answer)
