# %%
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
from scipy.stats import norm,multivariate_normal
from scipy import integrate

sigma=np.arange(0,1,0.1)
lam=1
gam=1
r=0.1
t=0.25
K=100
S0=100

cov_stock=-lam/np.sqrt(1+np.power(lam,2))
z=multivariate_normal([0,0],[[1,cov_stock],[cov_stock,1]])

def integrandGSN(y):
    return norm.pdf(y)*norm.cdf(lam*y+gam)/norm.cdf(gam/np.sqrt(1+np.power(lam,2)))
def SFGSN(x):
    return integrate.quad(integrandGSN,x,+np.inf)[0]   



def ECO(s):
    h1=(lam*s*np.sqrt(t)+gam)/np.sqrt(1+np.power(lam,2))
    h2=gam/np.sqrt(1+np.power(lam,2))
    w=(np.log(S0/K)+((r+(np.power(s,2)/2))*t)-np.log(norm.cdf(h1)/norm.cdf(h2)))/(s*np.sqrt(t))
        
    return S0*(1-z.cdf([h1,-w]))-np.exp(-r*t)*K*SFGSN(-w+(s*np.sqrt(t)))   

y=[ECO(s) for s in sigma]
plt.plot(sigma,y)
plt.show()
plt.savefig('ECO.png')
#%%
x=[0,0]
cov_stock=-lam/np.sqrt(1+np.power(lam,2))
z=multivariate_normal([0,0],[[1,cov_stock],[cov_stock,1]])
print(z.cdf(x))

# %%
from scipy import integrate
def integrandGSN(y):
    return norm.pdf(y)*norm.cdf(lam*y+gam)/norm.cdf(gam/np.sqrt(1+np.power(lam,2)))
def SFGSN(x):
    return integrate.quad(integrandGSN,x,+np.inf)[0]   


# %%
