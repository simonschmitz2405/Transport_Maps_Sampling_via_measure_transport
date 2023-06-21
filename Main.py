import numpy as np
import TransportMaps.Distributions as DIST
import TransportMaps.Distributions.Inference as DISTINF
import TransportMaps.Likelihoods as LKL
import TransportMaps.Maps as MAPS
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import TransportMaps.Diagnostics as DIAG
from TransportMaps import KL
import time


'''
Define the Posterior distribution by generating synthetic data
and then using these data in the likelihood. 
'''

# Define the Lorenz system/Lorenz equations
def lorenz_system(t, sys, sigma,rho,beta):
    A = sys[0]
    B = sys[1]
    C = sys[2]
    dA_dt = sigma * (B - A)
    dB_dt = A * (rho - C) - B
    dC_dt = A * B - beta * C
    return np.array([dA_dt, dB_dt, dC_dt])

# Set times on which ODE is solved
num_steps = 10000
t_span = np.array([0,2])
times = np.linspace(t_span[0],t_span[1],num_steps)

# Set initial condition
sys0 = np.array([0,1,20])

# Define the uniform prior distribution
class Uniform3D(DIST.Distribution):
    r"""
    Args: low1,high2,low2,high2,low3,high3
    """
    def __init__(self, low1, high1, low2, high2, low3, high3):
        super(Uniform3D, self).__init__(3)  #self.dim = 3
        self.low1 = low1
        self.high1 = high1
        self.low2 = low2
        self.high2 = high2
        self.low3 = low3
        self.high3 = high3
        self.volume = (high1 - low1) * (high2 - low2) * (high3 - low3)

    def log_pdf(self, x, *args, **kwargs):
        # x.shape is (m,3)
        # Returns array of (m,) elements
        cond1 = np.logical_and(x[:, 0] >= self.low1, x[:, 0] <= self.high1)
        cond2 = np.logical_and(x[:, 1] >= self.low2, x[:, 1] <= self.high2)
        cond3 = np.logical_and(x[:, 2] >= self.low3, x[:, 2] <= self.high3)
        valid_points = np.logical_and(np.logical_and(cond1, cond2), cond3)
        out = np.log(self.volume) * valid_points.astype(float)
        return out.flatten()

    def grad_x_log_pdf(self, x, *args, **kwargs):
        # x.shape is (m,3)
        # Returns array of (m,3) elements
        cond1 = np.logical_and(x[:, 0] >= self.low1, x[:, 0] <= self.high1)
        cond2 = np.logical_and(x[:, 1] >= self.low2, x[:, 1] <= self.high2)
        cond3 = np.logical_and(x[:, 2] >= self.low3, x[:, 2] <= self.high3)
        valid_points = np.logical_and(np.logical_and(cond1, cond2), cond3)
        grad = np.zeros_like(x)
        grad[valid_points] = 1.0 / self.volume
        return grad

    def hess_x_log_pdf(self, x, *args, **kwargs):
        # x.shape is (m,3)
        # Returns array of (m,3,3) elements
        cond1 = np.logical_and(x[:, 0] >= self.low1, x[:, 0] <= self.high1)
        cond2 = np.logical_and(x[:, 1] >= self.low2, x[:, 1] <= self.high2)
        cond3 = np.logical_and(x[:, 2] >= self.low3, x[:, 2] <= self.high3)
        valid_points = np.logical_and(np.logical_and(cond1, cond2), cond3)
        hess = np.zeros((x.shape[0], 3, 3))
        hess[valid_points] = 0.0
        return hess

# Construct the uniform prior distribution
prior = Uniform3D(3.5,4.5,5.5,6.5,6.5,7.5)

# Define the ForwardOperator
class ForwardOperator(MAPS.Map):
    r"""
    Args: dim_in, dim_out
    """
    def __init__(self):
        super(ForwardOperator, self).__init__(dim_in=3, dim_out=15)
    def evaluate(self, x, *args, **kwargs):
        out = np.ones([x.shape[0],15])
        for i in range(0,x.shape[0]):
            soln = solve_ivp(lorenz_system, t_span, sys0, t_eval=times, args=tuple(x[i,:]))
            A = soln.y[0][10:][::2000]
            B = soln.y[1][10:][::2000]
            C = soln.y[2][10:][::2000]
            new_line = np.vstack((A,B,C)).flatten()
            out[i,:] = new_line
        return out
    
# Initialize the map
Gmap = ForwardOperator()

# Initialize the noise model
noise = DIST.NormalDistribution(np.zeros(15), np.eye(15)*0.5)

# Generate some sythetic data
d = Gmap.evaluate(np.array([[4,6,7]]))[0,:]

# Initialize the log-likelihood
logL = LKL.AdditiveLogLikelihood(d, noise, Gmap)

# Create the posterior distribution by asemblying with log-likelihood and prior
post = DISTINF.BayesPosteriorDistribution(logL, prior)

# Visualize the prior distribution
fig = plt.figure(figsize=(6,6))
varstr = [r"$\rho$", r"$\sigma$", r"$\beta$"]
fig = DIAG.plotAlignedConditionals(prior, range_vec=[1,11],numPointsXax=50, fig=fig, show_flag=False, vartitles=varstr, title='Prior',show_title=True)
plt.show()

# Visualize the posterior distribution
fig = plt.figure(figsize=(6,6))
varstr = [r"$\rho$", r"$\sigma$", r"$\beta$"]
fig = DIAG.plotAlignedConditionals(post, range_vec=[1,11],numPointsXax=50, fig=fig, show_flag=False, vartitles=varstr,title='Posterior',show_title=True)
plt.show()


'''
Construct the Transport Map with order 1
'''
dim = post.dim
order = 1

dim = post.dim
qtype = 3          # Gauss quadrature
qparams = [6]*dim  # Quadrature order
reg = None         # No regularization
tol = 100          # Optimization tolerance
ders = 0           # Do not use gradient and Hessian

start_time_Construct_TM1 = time.time() # Start the timer
T = MAPS.assemble_IsotropicIntegratedExponentialTriangularTransportMap(dim, order, 'total') # Initialize the transport map

# Generate rho, the pull back and the push forward and start timer
rho = DIST.NormalDistribution(np.array([4,6,7]),covariance=np.eye(3))
pull_pi = DIST.PullBackParametricTransportMapDistribution(T, post)
push_rho = DIST.PushForwardParametricTransportMapDistribution(T, rho)

log = KL.minimize_kl_divergence(rho, pull_pi, qtype=qtype, qparams=qparams, regularization=reg, tol=tol, ders=ders) # Minimize the KLD
end_time_Construct_TM1 = time.time() # End the timer


# Calculate the elapsed time
elapsed_time_Construct_TM1 = end_time_Construct_TM1 - start_time_Construct_TM1

# Print the elapsed time
print("Elapsed time for Construction of TM with order 1:", elapsed_time_Construct_TM1, "seconds")


# Define the number of samples 
Nsample_TM = 10000

# Generate transport maps samples and stop the time
start_time_TM2 = time.time() # Start the timer
samples = push_rho.rvs(Nsample_TM) # Generate samples
end_time_TM2 = time.time() # End the timer

# Visualize the aligned conditonals of push forward
fig = plt.figure(figsize=(6,6))
varstr = [r"$\rho$", r"$\sigma$", r"$\beta$"]
fig = DIAG.plotAlignedConditionals(push_rho, range_vec=[0,10],numPointsXax=50, fig=fig, show_flag=False, vartitles=varstr,title='Push forward of TM with order 1',show_title=True)
plt.show()

# Visualize the aligned marginals for samples generated of push forward
fig = plt.figure(figsize=(6,6))
varstr = [r"$\rho$", r"$\sigma$", r"$\beta$"]
fig = DIAG.plotAlignedMarginals(samples,fig=fig,show_flag=False, vartitles=varstr, figname='Marginals',title='Samples_TM with order 1')
plt.show()


# Calculate the elapsed time
elapsed_time_TM2 = end_time_TM2 - start_time_TM2

# Print the elapsed time
print("Elapsed time for TM with order 1:", elapsed_time_TM2, "seconds")


'''
Construct the Transport Map with order 2
'''
dim = post.dim
order = 2

dim = post.dim
qtype = 3          # Gauss quadrature
qparams = [6]*dim  # Quadrature order
reg = None         # No regularization
tol = 100        # Optimization tolerance
ders = 0           # Do not use gradient and Hessian

start_time_Construct_TM2 = time.time() # Start the timerstart_time_MCMC = time.time() # Start the timer
T = MAPS.assemble_IsotropicIntegratedExponentialTriangularTransportMap(dim, order, 'total') # Initialize transport map

# Generate rho, the pull back and the push forward and start timer
rho = DIST.NormalDistribution(np.array([4,6,7]),covariance=np.eye(3))
pull_pi = DIST.PullBackParametricTransportMapDistribution(T, post)
push_rho = DIST.PushForwardParametricTransportMapDistribution(T, rho)

log = KL.minimize_kl_divergence(rho, pull_pi, qtype=qtype, qparams=qparams, regularization=reg, tol=tol, ders=ders) # minimize KLD
end_time_Construct_TM2 = time.time() # End the timer


# Calculate the elapsed time
elapsed_time_Construct_TM2 = end_time_Construct_TM2 - start_time_Construct_TM2

# Print the elapsed time
print("Elapsed time for Construction of TM with order 2:", elapsed_time_Construct_TM2, "seconds")


# Generate transport maps samples and stop the time
start_time_TM2 = time.time() # Start the timer
samples = push_rho.rvs(Nsample_TM) # generate samples
end_time_TM2 = time.time() # End the timer

# Visualize aligned conditionals of push forward
fig = plt.figure(figsize=(6,6))
varstr = [r"$\rho$", r"$\sigma$", r"$\beta$"]
fig = DIAG.plotAlignedConditionals(push_rho, range_vec=[0,10],numPointsXax=50, fig=fig, show_flag=False, vartitles=varstr,title='Push forward of TM with order 2',show_title=True)
plt.show()

# Visualize aligned marginals of samples of push forward
fig = plt.figure(figsize=(6,6))
varstr = [r"$\rho$", r"$\sigma$", r"$\beta$"]
fig = DIAG.plotAlignedMarginals(samples,fig=fig,show_flag=False, vartitles=varstr, figname='Marginals',title='Samples_TM with order 2')
plt.show()




# Calculate the elapsed time
elapsed_time_TM2 = end_time_TM2 - start_time_TM2

# Print the elapsed time
print("Elapsed time for TM with order 2:", elapsed_time_TM2, "seconds")


'''
Sample distribution via MCMC
'''
Nsample_MCMC = 10000 # Number of samples of Markov Chain
sig = 1 # Sigma for the new proposal steps
x0 = np.array([[1,1,1]]) # Initial value 

# Define the Metropolis Hasting step
def MHstep(x0,sig):
    # generate candidate
    xp0 = np.random.normal(loc=0,scale=sig)
    xp1 = np.random.normal(loc=0,scale=sig)
    xp2 = np.random.normal(loc=0,scale=sig)
    xp0 = x0[:,0] + xp0
    xp1 = x0[:,1] + xp1
    xp2 = x0[:,2] + xp2
    xp = np.array([[xp0[0],xp1[0],xp2[0]]])
    
    accprob = post.pdf(xp)/post.pdf(x0) # acceptance prob
    u = np.random.uniform(size=1) 
    if u <= accprob:
        x1 = xp # new point is candidate
        a = 1 # note acceptance
    else:
        x1 = x0 # new point is the same as old one
        a = 0 # note rejections
    return x1, a

# Define the Metropolis Hasting Algorithm
def MHSample(Nsample,x0):
    X = []
    x = x0
    for i in range(Nsample):
        x,a = MHstep(x,sig)
        X.append(x)
    return X

# Generate Metropolis Hasting Samples and stop the time
start_time_MCMC = time.time() # Start the timer
Samples = MHSample(Nsample_MCMC,x0)
end_time_MCMC = time.time() # End the timer


# Calculate the elapsed time
elapsed_time_MCMC = end_time_MCMC - start_time_MCMC

# Print the elapsed time
print("Elapsed time for MCMC:", elapsed_time_MCMC, "seconds")


# Formate Samples into Matrix
samples = np.array([sublist[0] for sublist in Samples])

# Visualize the samples of MCMC
fig = plt.figure(figsize=(6,6))
varstr = [r"$rho$",r"$sigma$",r"$beta$"]
fig = DIAG.plotAlignedMarginals(samples,fig=fig,show_flag=False, vartitles=varstr, figname='Marginals',title='Samples_MCMC')
plt.show()

# Visualize the scatter samples of MCMC
fig = plt.figure(figsize=(6,6))
varstr = [r"$rho$",r"$sigma$",r"$beta$"]
fig = DIAG.plotAlignedScatters(samples, s=1, bins=50, vartitles=varstr,fig=fig, show_flag=False)
plt.show()