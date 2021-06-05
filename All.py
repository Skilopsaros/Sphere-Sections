import numpy as np
import random as rng
from matplotlib import pyplot as plt
import matplotlib
from scipy import stats as st
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.sans-serif": ["Computer Modern Roman"]})

##################################################################################
########################### Defining the Distributions ###########################
##################################################################################
def constant(x, return_ks= False, number_of_samples = 0):
    if return_ks:
        result = []
        for i in range(int(number_of_samples)):
            result.append(x[0])
        return(result)
    return(x[0])

def from_range(limits, return_ks= False, number_of_samples = 0):
    if return_ks:
        result = []
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


##################################################################################
############## Defining the sphere objects and creating the volumes ##############
##################################################################################


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

##################################################################################
############ K-S tests and comparing overlap and non overlap voluems #############
##################################################################################
def calculate_ks(dis_1,dis_2):
    def calculate_comulative(dis):
        com = np.zeros(len(dis))
        sum = 0
        for i in range(len(dis)):
            sum += dis[i]
            for j in range(len(dis)):
                if j<=i:
                    com[i] += dis[j]
        return(com/sum)
    dis_1_c = calculate_comulative(dis_1)
    dis_2_c = calculate_comulative(dis_2)
    dif = abs(dis_1_c-dis_2_c)
    return(np.amax(dif))


def ks_full_test(set_1,set_2):
    max_values=np.amax(np.array([np.amax(np.array(set_1)),np.amax(np.array(set_2))]))
    delta = max_values/1000
    all_values = list(set_1)
    all_values.extend(list(set_2))
    def make_comulative(set, delta):
        com = np.zeros(1001)
        for i in set:
            com[int(np.floor(i/delta))]+=1
        return(com)
    ks_value = calculate_ks(make_comulative(set_1, delta),make_comulative(set_2, delta))

    ks_others = []
    for i in range(500):
        print(' '+str(i)+ '/' + str(500), end="\r", flush=True)
        rng.shuffle(all_values)
        first_set = all_values[:(int(len(all_values)/2))]
        second_set = all_values[(int(len(all_values)/2)):]
        ks_others.append(calculate_ks(make_comulative(first_set, delta),make_comulative(second_set, delta)))
    ks_others_array = np.array(ks_others)
    mean = np.mean(ks_others_array)
    sd = np.std(ks_others_array)
    return((ks_value-mean)/sd)

def compare_volums(number_of_spheres, distribution, max_x_distance, max_distance, *dist_parameters):
    sections  = [0,0]
    radii_cut = [0,0]
    radii_all = [0,0]
    sections[1], radii_cut[1], radii_all[1] = generate_simple_volume(number_of_spheres, distribution, max_x_distance, *dist_parameters)
    print()
    sections[0], radii_cut[0], radii_all[0] = generate_volume(number_of_spheres, distribution, max_x_distance, max_distance, *dist_parameters)
    print()
    plt.subplot(1,2,1)
    plt.hist(sections[0])
    plt.ylabel('$\phi_{(R)}$',fontsize='x-large')
    plt.xlabel('$R$',fontsize='x-large')
    plt.subplot(1,2,2)
    plt.xlabel('$R$',fontsize='x-large')
    plt.hist(sections[1])
    plt.show()


    return(ks_full_test(sections[0],sections[1]))

##################################################################################
######################## Calculating guess distributions #########################
##################################################################################

def calculate_fs(delta_R,n_max,phis):

    def A_coefficient(m,n,delta_R):
        return((delta_R*m)/(pow((pow(n,2)-pow(m,2)),0.5)))

    def calculate_f_n(phis,fs,delta_R,n_max,x):
        sum=0
        for i in range(x):
            sum += A_coefficient((n_max-x-1),(n_max-i),delta_R)*fs[n_max-i]
            #sum += A_coefficient((n_max-x-1),(n_max-i),delta_R)*fs[n_max-i] # maybe -1
        return((phis[n_max-x-1]-sum)/A_coefficient((n_max-x-1),(n_max-x),delta_R))

    fs=np.zeros(n_max+1)
    for i in range(n_max-2):
        fs[n_max-i-1]=calculate_f_n(phis,fs,delta_R,n_max,i)
        if fs[n_max-i-1]<0:
            fs[n_max-i-1]=0
    return(fs)

def alt_calculate_fs(delta_R,n_max,phis):
    def k_matrix_coeff(i,j):
        if i==j:
            return(delta_R*pow((i-0.75),0.5))
        elif j>i:
            return(delta_R*(pow(pow(j-1/2,2)-pow(i-1,2),0.5)-pow(pow(j-0.5,2)-pow(i,2),0.5)))
        else:
           return(0)
    k = np.zeros((n_max,n_max))
    for i in range(n_max):
        for j in range(n_max):
            k[i][j]=k_matrix_coeff(i+1,j+1)

    fs = np.zeros(n_max)
    for i in range(n_max):
        sum = 0
        for j in range(n_max):
            if j>i:
                sum+=k[i][j]*fs[j]
        fs[n_max-1-i]=(phis[n_max-1-i]-sum)/k[n_max-1-i][n_max-1-i]
        if fs[n_max-1-i]<0:
            fs[n_max-1-i]=0
    return(fs)


##################################################################################
#################################### Analysis ####################################
##################################################################################

def analyse_spheres(number_of_spheres, distribution, max_distance, number_of_bins, add_Error = False, delta_R=1, *dist_parameters):
    sections, radii_cut, radii_all = generate_simple_volume(number_of_spheres, distribution, max_distance, *dist_parameters)
    dis_radii_cut = np.zeros(number_of_bins)
    dis_radii_all = np.zeros(number_of_bins)
    phis = np.zeros(number_of_bins)
    print(sections[10])
    if add_Error:
        print('I AM HERE')
        for i in range(len(sections)):
            sections[i] += np.random.normal(0,sections[i]/10)
    print(sections[10])
    for i in sections:
        phis[int(np.floor(i/delta_R))]+=1
    for i in radii_cut:
        dis_radii_cut[int(np.floor(i/delta_R))]+=1
    for i in radii_all:
        dis_radii_all[int(np.floor(i/delta_R))]+=1
    fs = calculate_fs(delta_R,number_of_bins,phis)
    temp = np.array(list(fs)[:-1])
    fs = temp
    fs_alt = alt_calculate_fs(delta_R,number_of_bins,phis)

    for i in range(len(fs_alt)):
        if fs_alt[i]<0:
            fs_alt[i]=0
    for i in range(len(fs)):
        if fs[i]<0:
            fs[i]=0


    ks_alt = calculate_ks(fs_alt,dis_radii_all)
    ks = calculate_ks(fs,dis_radii_all)

    ks_expected = []
    print()
    for i in range(100):
        print(' '+str(i)+ '/' + str(100), end="\r", flush=True)
        data = [distribution(dist_parameters,return_ks=True,number_of_samples=np.floor(np.sum(fs))),distribution(dist_parameters,return_ks=True,number_of_samples=len(radii_all))]
        distributions = [np.zeros(number_of_bins),np.zeros(number_of_bins)]
        for j in range(len(data[0])):
            distributions[0][int(np.floor(data[0][j]))]+=1
        for j in range(len(data[1])):
            distributions[1][int(np.floor(data[1][j]))]+=1
        ks_expected.append(calculate_ks(distributions[0],distributions[1]))
    ks_expected_array = np.array(ks_expected)

    print('mean expected ks')
    print(np.mean(ks_expected_array))
    print('standard deviation expected ks')
    print(np.std(ks_expected_array))
    print('ks')
    print(ks)
    print('ks_alt')
    print(ks_alt)
    print('dis radii all')
    print(dis_radii_all)
    print('dis radii cut')
    print(dis_radii_cut)



    sum_fs = np.sum(fs)
    fs = fs/sum_fs
    print()
    print('sum fs')
    print(sum_fs)
    print('data points all')
    print(len(radii_all))
    sum_alt = np.sum(fs_alt)
    fs_alt = fs_alt/sum_alt
    print('fs')
    print(fs)
    print('alt fs')
    print(fs_alt)


    dis_radii_all=np.array(dis_radii_all)
    sum_all = np.sum(dis_radii_all)
    dis_radii_all=dis_radii_all/sum_all
    x = np.zeros(2*len(dis_radii_all)+2)
    y = np.zeros(2*len(dis_radii_all)+2)
    for i in range(len(dis_radii_all)+1):
        x[2*i]  =i*delta_R
        x[2*i+1]=i*delta_R
    for i in range(len(dis_radii_all)):
        y[2*i+1]=dis_radii_all[i]
        y[2*i+2]=dis_radii_all[i]


    positions = np.zeros(len(fs))
    for i in range(len(positions)):
        positions[i]=(i+0.5)*delta_R
    plt.bar(positions,fs,delta_R)
    plt.plot(x,y,color='red',linewidth = 3)
    plt.ylabel('$f_{(R)}$',fontsize='x-large')
    plt.xlabel('$R$',fontsize='x-large')
    plt.show()
    plt.clf()


    positions_alt = np.zeros(len(fs_alt))
    for i in range(len(positions_alt)):
        positions_alt[i]=(i+0.5)*delta_R
    plt.bar(positions_alt,fs_alt,delta_R)
    plt.plot(x,y,color='red',linewidth = 3)
    plt.ylabel('$f_{(R)}$',fontsize='x-large')
    plt.xlabel('$R$',fontsize='x-large')
    plt.show()

##################################################################################
#################################### Run Code ####################################
##################################################################################

analyse_spheres(30000, from_range, 21, 30, False, 1, 10,20)
#analyse_spheres(1000, constant, 21, 20, True, 1, 9.5)
#analyse_spheres(5000, normal_distribution, 40, 30, False, 1, 12,3)
#analyse_spheres(500, from_range, 8, 22, False, .1, 1,2)
#compare_volums(2000, from_range, 80, 2000, 10,20)
