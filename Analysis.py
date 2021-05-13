import numpy as np
import spheres as sph
import distributions as dis

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
    for i in range(n_max-1):
        fs[n_max-i]=calculate_f_n(phis,fs,delta_R,n_max,i)

    return(fs)

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


#1000, dis.from_range,80, 3000, 10,20
#print(calculate_fs(1,4,[0,10,8,2]))
def analyse_spheres(number_of_spheres, distribution, max_x_distance, max_distance, number_of_bins, delta_R=1, *dist_parameters):
    sections, radii_cut, radii_all = sph.generate_volume(number_of_spheres, distribution, max_x_distance, max_distance, *dist_parameters)
    dis_radii_cut = np.zeros(number_of_bins)
    dis_radii_all = np.zeros(number_of_bins)
    phis = np.zeros(number_of_bins)
    for i in sections:
        phis[int(np.floor(i))]+=1
    for i in radii_cut:
        dis_radii_cut[int(np.floor(i))]+=1
    for i in radii_all:
        dis_radii_all[int(np.floor(i))]+=1
    fs = calculate_fs(delta_R,number_of_bins,phis)
    print('dis radii all')
    print(dis_radii_all)
    print('dis radii cut')
    print(dis_radii_cut)
    print('phis')
    print(phis)
    print(calculate_fs(delta_R,number_of_bins,phis))
