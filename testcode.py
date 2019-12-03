import matplotlib.pyplot as plt
from SFSD import SFSD
from optimal_params import optimal_params
import numpy as np

#%%
number_samps = 50
dimension = 2
iterations = 300

#%%
params = optimal_params(N=number_samps,d=dimension,p_0=1.2,verbose=0,summary=1) 

PSD,r0,r1,p0,r_min = params.compute_params()

#%%
len_r = 100
r = np.linspace(0.0001,2*r1,len_r)

pcf = SFSD(pcf=1,N=number_samps,d=dimension,r_0=r0,r_1=r1,p_0=p0,lr=0.001,MSE=0.0001,maxGD=iterations,len_r=len_r)

X,g,g0, error_func, losses = pcf.generate()

#%%
plt.figure()
plt.plot(r,g,label='PCF matching')
plt.plot(r,g0,label='Target PCF')
plt.xlabel('r')
plt.ylabel('PCF')
plt.legend()
plt.show()