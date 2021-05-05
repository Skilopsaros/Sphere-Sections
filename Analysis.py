import numpy as np
import spheres as sph
import distributions as dis


def calculate_fs(delta_R,n_max,phis):

    def A_coefficient(m,n,delta_R):
        return((delta_R*m)/(pow((pow(n,2)-pow(m,2)),0.5)))

    def calculate_f_n(phis,fs,delta_R,n_max,x):
        sum=0
        for i in range(x):
            sum += A_coefficient((n_max-x-1),(n_max-i),delta_R)*fs[n_max-i] # maybe -1
        return((phis[n_max-x-1]-sum)/A_coefficient((n_max-x-1),(n_max-x),delta_R))

    fs=np.zeros(n_max+1)
    for i in range(n_max-1):
        fs[n_max-i]=calculate_f_n(phis,fs,delta_R,n_max,i)

    return(fs)


sections = sph.generate_volume(500, dis.constant,50, 2000, 10)
phis = np.zeros(12)
for i in sections:
    phis[int(np.floor(i))]+=1
print(phis)
print(calculate_fs(1,11,phis))
