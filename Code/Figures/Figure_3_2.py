import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal

# Parameter for Distribution
mu_x = 1
sigma_x = 0.1

x = np.linspace(-1, 2, 1000)
y = np.linspace(-0.5, 3, 1000)

# Plot des Koordinatensystems
fig, ax = plt.subplots()
ax.plot(x, np.zeros_like(x), color='black', linewidth=1)  # x-Achse
ax.plot(np.zeros_like(y)-0.5, y, color='black', linewidth=1)  # y-Achse

# Define the transport map
def T(x):
    return 2**x -0.5

# Plot the Reference distribution
pdf_x = multivariate_normal.pdf(x, mean=mu_x, cov=sigma_x)
ax.plot(x, pdf_x, color='blue', label='Reference')

# Plot the exponential function
ax.plot(x,T(x),color='green', label='T(X)')


# Plot the Target distribution
ax.plot(-T(pdf_x),x,color='red',label='Target')

ax.text(-1.6,3.0,r'target random' '\n' 'variable $Y$',size=15)
ax.text(1.0,-0.8, r"reference random" "\n" r"variable $X$",size=15)
ax.text(1.4,0.8,r'$\nu_{\eta}$',size=20)
ax.text(-1.5,1.5,r'$\nu_{\pi}$',size=20)
ax.text(0.8,3,r'$Y = T(X)$',size=20)
plt.axis('off')
plt.show()
