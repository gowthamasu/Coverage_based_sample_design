import sys
from copy import deepcopy as dc
import numpy as np
import scipy.spatial.distance as  sp
import scipy.special as  sc
import pickle
import random
import progressbar
#%%

class SFSD:
    
    def __init__(self, pcf,N,d,r_0,r_1,p_0,lr,len_r,MSE,maxGD):
        """ pcf = 0 is step ---- pcf = 1 is stair """
        self.pcf = pcf      
        self.N = N
        self.d = d
        self.r_0 = r_0
        self.r_1 = r_1
        self.p_0 = p_0
        self.lr = lr
        self.len_r = len_r
        self.MSE = MSE
        self.maxGD = maxGD
        
    """ Selecting kernel width """    
    def choose_sigma(self):
        if self.d==2:
            sigma=0.0065            
        elif self.d==3:
            sigma = 0.007
        elif self.d==4:
            sigma =0.01
        elif self.d==5:
            sigma=0.01
        else:
            sigma=0.01            
        return sigma
    
    """ Computing the edge correction factor """     
    def edge_correction(self):
        
        max_r = 2*self.r_1
        r = np.linspace(0.0001,max_r,self.len_r)
    
        S_E = (2*(np.power(r,self.d-1))*np.pi**(.5*self.d))/sc.gamma(.5*self.d)
        
        
        if self.d == 2:
            Gamma_W = 1-r*(4/np.pi)+np.power(r,2)/np.pi
        elif self.d == 3:
            Gamma_W = 1.-r*(1.5)+np.power(r,2)*(2./np.pi)-np.power(r,3)*(1./(4*np.pi))
        elif self.d == 4:
            Gamma_W = 1.00042915-r*(1.6942318)+np.power(r,2)*0.93467419-np.power(r,3)*0.18312574       
        elif self.d == 5:
            Gamma_W = 1.00025108-r*(1.86570145)+np.power(r,2)*1.21545701-np.power(r,3)*0.29559135     
        else:
            Gamma_W = 0.99934601-r*(2.00754381)+np.power(r,2)*1.45216743-np.power(r,3)*0.39422813    
        
        const =  np.power(np.multiply(Gamma_W,S_E),-1)/(self.N*(self.N-1)) 
        
        return const
    
    """ Faster Gaussian kernel computation """     
    def G_kern(self,Y,sigma):
        Lo=np.zeros(Y.shape)
        Y=np.abs(Y)
        cond_A=Y<0.5
        set_A=np.extract(cond_A,Y)
        
        a=1/(sigma**2)
        b=np.square(set_A)
        c=0.5642/sigma
        L=c*np.exp(-b*a)
        
        np.place(Lo,cond_A,L)
        return Lo

    """ r_min for step design """    
    def rmin(self):
        r_min = (sc.gamma(1+0.5*self.d)/(self.N*np.pi**(0.5*self.d)))**(1.0/self.d) 
        return r_min
    
    
    """ defining PCF and initial calculation """  
    def initial_calculation(self):
        max_r = 2*self.r_1
        r = np.linspace(0.0001,max_r,self.len_r)
        r_min = self.rmin()
        """ Poisson Disk Sampling PCF approximation """
        if self.pcf ==1:
            g0 = np.piecewise(r, [r < r_min, r >= r_min], [0, 1])
            g0[np.max(np.where(g0==0))] = 0.5
    
        if self.pcf ==0:
            
            g01 = np.piecewise(r, [r < self.r_1, r > self.r_1], [0, 1])
            g02 = np.piecewise(r, [r > self.r_1, r <= self.r_1], [0, 1]) 
            g03 = np.piecewise(r, [r > self.r_0, r <= self.r_0], [0, 1])
            g0 = self.p_0*(g02-g03)+g01
            g0[np.max(np.where(g0==0))] = 0.5
                
            
        w_r = 0.1
        weight = np.piecewise(r, [r <= r_min, r > r_min], [w_r, 1])
        """ Generate Initial Pointset """
       
        X_init = np.random.rand(self.N,self.d)
        D = sp.squareform(sp.pdist(X_init))      
        np.fill_diagonal(D,float('inf'))  
        D_old = dc(D)
        G_init_local = np.random.rand(self.N,self.len_r)
        sigma = self.choose_sigma()
        const = self.edge_correction()
        
        """ Calculate initial local PCFs """
        for i in range(1,self.N+1):           
            Y = (np.mat(D[i-1]))-(np.transpose(np.matrix(r)))
            Lo=self.G_kern(Y,sigma)
            Ks = Lo.sum(axis = 1)
            G_init_local[i-1] = np.multiply(np.array(const).flatten(),np.array(Ks).flatten())
            
            
        """ Find if PCF is realizable by N points """
        if min(g0) < 0:        
            print("PCF can not be realized with given number of point samples")        
        else:        
           """ Calculate initial global PCF """
           g = G_init_local.sum(axis=0)
           g_init = dc(g)
           
           return g0, g_init, D, D_old, weight, G_init_local, X_init
       
        
    """ Sample generation """  
    def generate(self):        
        """ Gradient descent based PCF matching for point synthesis """
        const = self.edge_correction()
        sigma = self.choose_sigma()
        g0, g_init, D, D_old, weight, G_init_local, X_init = self.initial_calculation()
        max_r = 2*self.r_1
        r = np.linspace(0.0001,max_r,self.len_r)
        g=g_init
        
        
        """ Initialize parameters """
        j = 1
        X = dc(X_init)
        G_local_old = (G_init_local)
        error_func = []
        error=np.square(np.subtract(g, g0)).mean()
        error_func.append(error)
        
        losses=[]
        
        alpha = self.lr
        beta1=0.9
        beta2=0.999
        eps=1/np.power(10,8)
        
        m_t=np.zeros((self.N,self.d))
        v_t=np.zeros((self.N,self.d))
        
        m_hat=np.zeros((self.N,self.d))
        v_hat=np.zeros((self.N,self.d))
        
        """ Gradient Descent Loop """
              
        with progressbar.ProgressBar(max_value=self.maxGD) as bar:

            while(error>self.MSE and j < self.maxGD):
    
                j = j+1
    
    
                for i in range(1,self.N+1):       
    
                    
                    G = np.divide(np.multiply(g-g0,const), weight) 
                    U = X[i-1,:] - X
                    N1 = np.linalg.norm(U,axis = 1)+0.0001
                    D_old[i-1] = dc(N1)
                    D_old[i-1][D_old[i-1]==0] = float('inf')
                    Y_old = (np.mat(D_old[i-1]))-(np.transpose(np.matrix(r)))
                    Lo_old = self.G_kern(Y_old,sigma)
                    N1[i-1] = 1
                    N2 = np.tile(np.transpose(np.matrix(N1)),(1,self.d))
                    U = np.divide(U,N2)
                    Y_old[np.isinf(Y_old)] = 0
                    Z = np.multiply(np.multiply(np.transpose(np.tile(G,(self.N,1))),Y_old ),Lo_old)
                    W = Z.sum(axis=0)            
                    M = np.multiply(U,np.transpose(np.tile(W,(self.d,1))))
                    
                    grad = - (M.sum(axis=0)/np.linalg.norm(M.sum(axis=0)+0.0001))
    
                    """ ADAM gradient update """
                    m_t[i-1,:]= beta1 * m_t[i-1,:] + (1-beta1) * grad
                    v_t[i-1,:]= beta2 * v_t[i-1,:] + (1-beta2) * np.square(grad)
                    
                    m_hat[i-1,:]=m_t[i-1,:]/(1-beta1)
                    v_hat[i-1,:]=v_t[i-1,:]/(1-beta2)
                    
                    adam_grad = alpha * m_hat[i-1,:]/(np.sqrt(v_hat[i-1,:])+eps)
                
                    X[i-1,:] = np.subtract(X[i-1,:], adam_grad)   
      
                        
                    X[np.isnan(X)]  = 0 
                    """ Points should be in the bounded region """
                    for jj in range(self.d):
                        if np.mod(i,3)== 0:
                            
                            if X[i-1,jj] > 1:
                               X[i-1,jj] = np.random.rand(1)
                            if X[i-1,jj] < 0:
                               X[i-1,jj] = np.random.rand(1)
                        else:
                            if X[i-1,jj] > 1:
                               X[i-1,jj] = .999
                            if X[i-1,jj] < 0:
                               X[i-1,jj] = 0.001
                
                    U = X[i-1,:] - X
                    D[i-1] = np.linalg.norm(U,axis = 1)
                    D[i-1][D[i-1]==0] = float('inf')
                    Y = (np.mat(D[i-1]))-(np.transpose(np.matrix(r)))
                    Lo = self.G_kern(Y,sigma)
                    Ks = Lo.sum(axis = 1)
                    G_local_new = np.multiply(np.array(const).flatten(),np.array(Ks).flatten())                
                    G_local_old[i-1,:]=G_local_new
        
                    """ Update local PCF of all the point """ 
                g = G_local_old.sum(axis=0)
                g[g<0]=0
                       
                Loss = np.mean(np.divide(np.multiply(g-g0,const), weight)) 
                losses.append(Loss)
                error=np.square(np.subtract(g, g0)).mean()
                error_func.append(error)               
                                
#                print ( 'epoch :'+str(j)+'  MSE error:'+str(error) +'  Loss:'+str(Loss)) 
                bar.update(j)
                
        return X,g,g0, error_func, losses        
    

#%%



