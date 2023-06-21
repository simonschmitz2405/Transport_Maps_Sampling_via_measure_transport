import numpy as np
import matplotlib.pyplot as plt  
from scipy.misc import derivative
from scipy import integrate




def g(x):
    return 1+1.4*x**2+1.2*x**3-4.5*x**4+2*x**5

print(integrate.quad(lambda t: np.exp(derivative(g,t,dx=1e-3)),0,-1)[0])
def S(x):
    list = []
    for i in range(len(x)):
        list.append(integrate.quad(lambda t: np.exp(derivative(g,t,dx=1e-3)),0,x[i])[0])
    return list
    
 
x = np.linspace(-0.4,1.4,100)
plt.plot(x,g(x),label='g(x)')
plt.plot(x,derivative(g,x,dx=1e-3),label=r'$\partial_{x}g(x)$')
plt.plot(x,np.exp(derivative(g,x,dx=1e-3)),label=r'$\exp(\partial_{x}g(x))$')
plt.legend()
plt.show()

plt.plot(x,g(x),label='g(x)')
plt.plot(x,S(x),label='T(x)')
plt.legend()
plt.show()

