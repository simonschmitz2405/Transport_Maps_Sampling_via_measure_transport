import numpy as np
import scipy.stats as sts  
import matplotlib.pyplot as plt 
from scipy import integrate


def likelihood(x):
    return sts.norm.pdf(x,scale=0.8)

def prior(x):
    return sts.norm.pdf(x,loc=-1,scale=0.5)

def integrand(x):
    return likelihood(x) * prior(x)

def post(x):
    result = integrate.quad(integrand, -np.inf, np.inf)[0]
    return integrand(x)/result

x = np.linspace(-3,3,1000)

plt.plot(x,likelihood(x),label='Likelihood(x)')
plt.plot(x,prior(x),label='Prior(x)')
plt.plot(x,post(x),label='Posterior(x)')
plt.legend()
plt.show()
