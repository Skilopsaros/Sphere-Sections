import numpy as np
import spheres as sph
import distributions as dis
from matplotlib import pyplot as plt
import matplotlib
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.sans-serif": ["Computer Modern Roman"]})



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

##########################################

#def alt_calculate_fs(delta_R,n_max,phis):
#
#    def k_matrix_coeff(m,n):
#        if n==m:
#            return(pow((m-0.75),0.5))
#        elif n>m:
#            return((pow(pow(n-1/2,2)-pow(m-1,2),0.5)-pow(pow(n-0.5,2)-pow(m,2),0.5)))
#        else:
#           return(0)
#    k = np.zeros((n_max,n_max))
#    for n in range(n_max):
#        for m in range(n_max):
#            k[m][n]=k_matrix_coeff(m+1,n+1)
#    k_inv = np.linalg.inv(k)
#
#
#    fs = np.zeros(n_max)
#    for i in range(n_max):
#        for j in range(n_max):
#            fs[n_max-1-i]+=(1/delta_R)*k_inv[n_max-i-1][n_max-1-j]*phis[n_max-1-j]
#    return(fs)


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


########################################
#1000, dis.from_range,80, 3000, 10,20
#print(calculate_fs(1,4,[0,10,8,2]))
def analyse_spheres(number_of_spheres, distribution, max_x_distance, max_distance, number_of_bins, delta_R=1, *dist_parameters):
    sections, radii_cut, radii_all = sph.generate_volume(number_of_spheres, distribution, max_x_distance, max_distance, *dist_parameters)
    dis_radii_cut = np.zeros(number_of_bins)
    dis_radii_all = np.zeros(number_of_bins)
    phis = np.zeros(number_of_bins)
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
    for i in range(100):
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
analyse_spheres(3000, dis.from_range, 80, 2400000, 22, 1, 10,20)
#analyse_spheres(1000, dis.constant, 40, 40000, 450, 0.03, 9.5)
#analyse_spheres(3000, dis.normal_distribution, 50, 800000, 30, 1, 12,3)
#analyse_spheres(500, dis.from_range, 8, 12000, 22, .1, 1,2)
