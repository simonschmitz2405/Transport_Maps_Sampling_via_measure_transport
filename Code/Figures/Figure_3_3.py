import matplotlib.pyplot as plt
import numpy as np
import numpy.linalg as npla
import scipy.stats as stats
import scipy.linalg as scila
import SpectralToolbox.Spectral1D as S1D
import TransportMaps as TM
import TransportMaps.Maps as MAPS
import TransportMaps.Distributions as DIST
import TransportMaps.Diagnostics as DIAG
import TransportMaps.KL as KL
import TransportMaps.Maps.Functionals as FUNC


a = 0.9
b = 1
mu = np.ones(2)
cov = np.array([[1.,0.2],[0.8,1.]])

# Construct the reference measure eta and target measure pi
eta = DIST.StandardNormalDistribution(2)
pi = DIST.BananaDistribution(a=a,b=b,mu=mu,sigma2=cov)

# Visualize the contour of reference and target measure
ndiscr = 100
x = np.linspace(-3,3,ndiscr)
y = np.linspace(-9,3,ndiscr)
xx,yy = np.meshgrid(x,y)
x2d = np.vstack((xx.flatten(),yy.flatten())).T 
pdf2d = pi.pdf(x2d).reshape(xx.shape)
pdf2d_eta = eta.pdf(x2d).reshape(xx.shape)
fig, (ax1,ax2) = plt.subplots(1,2)
ax1.contour(xx,yy,pdf2d_eta)
ax2.contour(xx,yy,pdf2d)
plt.show()


# Define the IntegratedExponentialApproximation of Transport Map
order = 3
Tk_list = []
active_vars = []
c_basis_list = [S1D.HermiteProbabilistsPolynomial()]
c_orders_list = [0]
c_approx = FUNC.MonotonicLinearSpanApproximation(
    c_basis_list, spantype='full', order_list=c_orders_list)
e_basis_list = [S1D.ConstantExtendedHermiteProbabilistsFunction()]
e_orders_list = [order]
e_approx = FUNC.MonotonicLinearSpanApproximation(
    e_basis_list, spantype='full', order_list=e_orders_list)
Tk = FUNC.MonotonicIntegratedExponentialApproximation(c_approx, e_approx)
Tk_list.append( Tk )
active_vars.append( [0] )
#T = MAPS.IntegratedExponentialTriangularTransportMap(active_vars, Tk_list)
T = MAPS.assemble_IsotropicIntegratedExponentialTriangularTransportMap(2,order,'full')

# Define pushforward and pullback of transport map
push_eta = DIST.PushForwardParametricTransportMapDistribution(T, eta)
pull_pi = DIST.PullBackParametricTransportMapDistribution(T, pi)


# Define the Kullback-Leibler Divergence
qtype = 3           # Gauss quadrature
qparams = [10] * 2  # Quadrature order
reg = None          # No regularization
tol = 1e-5         # Optimization tolerance
ders = 2            # Use gradient and Hessian
log = KL.minimize_kl_divergence(
    eta, pull_pi, qtype=qtype, qparams=qparams, regularization=reg,
    tol=tol, ders=ders)

# Sample from reference measure and pushforward
M = 1000
samples = push_eta.rvs(M)

# Visualize the samples
fig, (ax1,ax2) = plt.subplots(1,2)
#ax1.contour(xx,yy,pdf2d_eta)
#ax2.contour(xx,yy,pdf2d)
ax1.scatter(eta.rvs(M)[:,0],eta.rvs(M)[:,1], c='#8cb63c', s=1.)
ax2.scatter(samples[:,0], samples[:,1], c='#8cb63c', s=1.)
plt.show()
