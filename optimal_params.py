import matplotlib.pyplot as plt
import numpy as np
import scipy.special as  sc
import random


#%%
class optimal_params:

    def __init__(self, N,d,p_0,verbose,summary):
        self.N = N
        self.d = d
        self.p_0 = p_0
        self.bins = 100
        self.verbose = verbose
        self.summary = summary

        """ initial r0 by r_min of step design """
    def rmin(self):
        r_min = (sc.gamma(1+0.5*self.d)/(self.N*np.pi**(0.5*self.d)))**(1.0/self.d)
        return r_min

    """ initial r1 >> r0 """

    def r_1(self):
        if self.d > 4:
            scale = 1.5
        else:
            scale = 1.2

        return scale*self.rmin()

    """ Compute power spectral density """
    def PSD(self,r0,r1):
        V = 1
        k=np.linspace(1,1000,self.bins)
        nv=self.N/V
        P=1-nv*self.p_0*((6.28*r0/k)**(0.5*self.d))*sc.jve((0.5*self.d),k*r0)-nv*(1-self.p_0)*((6.28*r1/k)**(0.5*self.d))*sc.jve((0.5*self.d),k*r1)
        check=np.min(P)
        c_atzero=P[0]
        return check,P,c_atzero


    """ optimization procedure to find optimal r0 and r1 """

    def compute_params(self):
        """ Initialize  """
        check=1
        inc = 0.00000001
        c_atzero=check+1
        r0 = self.rmin()
        r_min = self.rmin()
        r1 = self.r_1()
        V = 1
        nv=self.N/V

        e = 0
        f = 0

        """ Iterate until the condition is satisfied """
        while( check > 0.0001):
            e = e+1
            f = f+1
            check,P,c_atzero = self.PSD(r0,r1)
            grad_1 = nv*self.p_0*((6.28*r0)**(0.5*self.d))*((self.d/r0)*sc.jve((0.5*self.d),r0)+sc.ive((0.5*self.d)-1,r0)-sc.jve((0.5*self.d)+1,r0))
            grad_2 = nv*(1-self.p_0)*((6.28*r1)**(0.5*self.d))*((self.d/r1)*sc.jve((0.5*self.d),r1)+sc.ive((0.5*self.d)-1,r1)-sc.jve((0.5*self.d)+1,r1))

            grad_norm = (check/r_min)+(r0*grad_1/r_min)

            r0=r0+inc*grad_norm
            r1=r1+inc*(grad_2*r0/r_min)

            if self.verbose == 1:
                if f == 1000:
                    print('iter =',e,'  check=',check)
                    f = 0


        r0=r0-inc*grad_1
        r1=r1-inc*grad_2

        check,P,c_atzero = self.PSD(r0,r1)

        """ Summary and figures """
        if self.summary == 1:

            print('---------------------------------------------------')
            print('N = ',self.N)
            print('d = ',self.d)
            print('r0 = ',r0)
            print('r1 = ',r1)
            print('p0 = ',self.p_0)
            print('delta = ',r1-r0)
            print('rmin_step =',r_min)
            print('gain = ',r0/r_min)
            print('---------------------------------------------------')

            plt.figure()
            plt.plot(P,label='PSD')
            ze=np.zeros((self.bins,1))
            plt.plot(ze,':r')
            plt.title('PSD : N='+str(self.N)+'  d='+str(self.d))
            plt.legend()
            #plt.show()

        return P,r0,r1,self.p_0,r_min

#%%
