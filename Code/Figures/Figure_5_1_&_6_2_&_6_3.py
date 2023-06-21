import logging
import warnings
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
import SpectralToolbox.Spectral1D as S1D
import TransportMaps as TM
import TransportMaps.Maps.Functionals as FUNC
import TransportMaps.Maps as MAPS
from TransportMaps import KL
warnings.simplefilter("ignore")
TM.setLogLevel(logging.INFO)

import TransportMaps.Distributions as DIST

class FiskDistribution(DIST.Distribution):
    def __init__(self, alpha, beta, gamma):
        super(FiskDistribution,self).__init__(1)
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.dist = stats.fisk(c=alpha, scale=beta, loc=gamma)
    def pdf(self, x, params=None, *args, **kwargs):
        return self.dist.pdf(x).flatten()
    def log_pdf(self, x, params=None, *args, **kwargs):
        return self.dist.logpdf(x).flatten()
    def grad_x_log_pdf(self, x, params=None, *args, **kwargs):
        a = self.alpha
        b = self.beta
        g = self.gamma
        z = (x-g)/b
        return (a/b)*((z/(1+z))**(a+1))
    def hess_x_log_pdf(self, x, params=None, *args, **kwargs):
        a = self.alpha
        b = self.beta
        g = self.gamma
        z = (x-g)/b
        return -((a/b)*((z/(1+z))**(a+1))*(z/(1+z)))/b**2.0

alpha = 4.0
beta = 3.0
gamma = 0.0
pi = FiskDistribution(alpha,beta,gamma)

x = np.linspace(-10.0, 40.0, 100).reshape((100,1))
plt.figure()
plt.plot(x, pi.pdf(x))

class FiskTransportMap(object):
    def __init__(self, alpha, beta, gamma):
        self.tar = stats.fisk(c=alpha, scale=beta, loc=gamma)
        self.ref = stats.norm(0.,1.)
    def evaluate(self, x, params=None):
        if isinstance(x,float):
            x = np.array([[x]])
        if x.ndim == 1:
            x = x[:,None]
        out = self.tar.ppf( self.ref.cdf(x) )
        return out
    def __call__(self, x):
        return self.evaluate(x)

Tstar = FiskTransportMap(alpha,beta,gamma)

x_tm = np.linspace(-4,4,100).reshape((100,1))
def plot_mapping(tar_star, Tstar, tar=None, T=None):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax_twx = ax.twinx()
    ax_twy = ax.twiny()
    ax.plot(x_tm, Tstar(x_tm), 'k-', label=r"$T^\star$") # Map
    n01, = ax_twx.plot(x_tm, stats.norm(0.,1.).pdf(x_tm), '-b') # N(0,1)
    g, = ax_twy.plot(tar_star.pdf(Tstar(x_tm)), Tstar(x_tm), '-r') # Gumbel
    if T is not None:
        ax.plot(x_tm, T(x_tm), 'k--', label=r"$\hat{T}$") # Map
    if tar is not None:
        ax_twy.plot(tar.pdf(Tstar(x_tm)), Tstar(x_tm), '--r') # Gumbel
    ax.set_ylabel(r"Map")
    ax_twx.set_ylabel('N(0,1)')
    ax_twx.yaxis.label.set_color(n01.get_color())
    ax_twy.set_xlabel('Fisk')
    ax_twy.xaxis.label.set_color(g.get_color())
    ax.legend(loc = (0.1, 0.8))
    
plot_mapping(pi, Tstar)
plt.show()


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
T = MAPS.IntegratedExponentialTriangularTransportMap(active_vars, Tk_list)

rho = DIST.StandardNormalDistribution(1)
push_rho = DIST.PushForwardParametricTransportMapDistribution(T, rho)
pull_pi = DIST.PullBackParametricTransportMapDistribution(T, pi)

qtype = 3      # Gauss quadrature
qparams = [20] # Quadrature order
reg = None     # No regularization
tol = 1e-10    # Optimization tolerance
ders = 0       # Use gradient and Hessian
log = KL.minimize_kl_divergence(
    rho, pull_pi, qtype=qtype, qparams=qparams, regularization=reg,
    tol=tol, ders=ders)

plot_mapping(pi, Tstar, push_rho, T)
plt.show()

