import numpy as np
import random as rng

class sphere:
    def __init__(self,radius, coordinates):
        self.coordinates = coordinates
        self.radius = radius
        if pow(radius,2)>pow(coordinates[0],2):
            self.plane_cut_radius = pow(pow(radius,2)-pow(coordinates[0],2),0.5) # radius of section cut by y-z plane
        else:
            self.plane_cut_radius = False

    def get_distance(self,coordinates):
        dist_squared = 0
        for i in range(len(coordinates)):
            dist_squared += pow(coordinates-self.coordinates,2)
        return(pow(dist_squared,0.5))

def generate_sphere(spheres,radius,max_distance,dimensions=3,exception_limit=10000):
    def generate_coordinates():
        coordinates=[]
        for i in range(dimensions):
            coordinates[i] = max_distance*(random.random()*2-1)
        return(coordinates)
    success = 0
    n = 0
    while (0==success):
        n+=1
        if n > exception_limit:
            class LimitError:
                pass
            raise(LimitError)
        new_coordinates = generate_coordinates()
        success = 1
        for sphere in spheres:
            distance = sphere.get_distance(new_coordinates)
            if distance < (sphere.radius+radius):
                success=success*0
    return(sphere(radius, new_coordinates))

def generate_volume(number_of_spheres, distribution, max_distance):

    spheres = []
    sections = []
    while len(sections)<number_of_spheres:
        spheres.append(generate_sphere(spheres,distribution(),max_distance))
        if spheres[len(spheres)-1].plane_cut_radius != False:
            sections.append(spheres[len(spheres)-1].plane_cut_radius)
    return(sections)
