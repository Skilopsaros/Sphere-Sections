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
            dist_squared += pow(coordinates[i]-self.coordinates[i],2)
        return(pow(dist_squared,0.5))

def generate_sphere(spheres,radius,max_distance, max_x_distance,dimensions=3,exception_limit=10000):
    def generate_coordinates():
        coordinates=np.random.random(dimensions)
        coordinates = max_distance*(coordinates*2-1)
        coordinates[0] = max_x_distance*(rng.random()*2-1)
        return(coordinates)
    success = 0
    n = 0
    fails = 0
    while (0==success):
        n+=1
        if n > exception_limit:
            class LimitError:
                pass
            raise(LimitError)
        new_coordinates = generate_coordinates()
        success = 1
        for each in spheres:
            distance = each.get_distance(new_coordinates)
            if distance < (each.radius+radius):
                success=0
                fails +=1
    return(sphere(radius, new_coordinates), fails)

def generate_volume(number_of_spheres, distribution, max_x_distance, max_distance, *dist_parameters):
    spheres = []
    sections = []
    cut_spheres_radii = []
    total_fails = 0
    while len(sections)<number_of_spheres:
        new_sphere, fails = generate_sphere(spheres,distribution(dist_parameters),max_distance, max_x_distance)
        spheres.append(new_sphere)
        total_fails+=fails
        if spheres[len(spheres)-1].plane_cut_radius != False:
            sections.append(spheres[len(spheres)-1].plane_cut_radius)
            print(' '+str(len(sections))+ '/' + str(number_of_spheres), end="\r", flush=True)
            cut_spheres_radii.append(spheres[len(spheres)-1].radius)

    spheres_radii = []
    for i in spheres:
        spheres_radii.append(i.radius)
    print('total relocations:'+str(total_fails))
    return(sections,cut_spheres_radii, spheres_radii)


class simple_sphere:
    def __init__(self,radius, distance):
        self.distance = distance
        self.radius = radius
        if pow(radius,2)>pow(distance,2):
            self.plane_cut_radius = pow(pow(radius,2)-pow(distance,2),0.5) # radius of section cut by y-z plane
        else:
            self.plane_cut_radius = False

def generate_simple_sphere(radius,max_distance):
    def generate_distance():
        distance = max_distance*(rng.random()*2-1)
        return(distance)
    return(simple_sphere(radius, generate_distance()))


def generate_simple_volume(number_of_spheres, distribution, max_distance, *dist_parameters):
    spheres = []
    sections = []
    cut_spheres_radii = []
    while len(sections)<number_of_spheres:
        spheres.append(generate_simple_sphere(distribution(dist_parameters),max_distance))
        if spheres[len(spheres)-1].plane_cut_radius != False:
            sections.append(spheres[len(spheres)-1].plane_cut_radius)
            print(' '+str(len(sections))+ '/' + str(number_of_spheres), end="\r", flush=True)
            cut_spheres_radii.append(spheres[len(spheres)-1].radius)

    spheres_radii = []
    for i in spheres:
        spheres_radii.append(i.radius)
    return(sections,cut_spheres_radii, spheres_radii)
