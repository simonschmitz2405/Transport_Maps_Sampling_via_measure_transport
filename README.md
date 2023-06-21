# Thesis - Transport Maps: Sampling via measure transport - Simon Schmitz

This is the implementation for the Bachelor's thesis: Transport Maps: Sampling via measure transport.

In statistics and data science, we often have to deal with complicated probability measures.
A common goal is to explore this complex probability measure by generating samples
or by estimating statistics of this given target measure. Avoiding classical stochastic
exploration tools such as Markov Chain Monte Carlo (MCMC) the measure transport
approach, described by Marzouk in the paper [13], provides an efficient and fast way of
sampling from this desired complex target probability measure. Specifically, suppose we
have either access to samples X1, · · · , XM or access to the evaluation of the target measure
νπ up to the normalization constant. Given this information, we can construct either the
indirect or the direct transport map T. Hereby, the transport map is a deterministic
coupling between a complicated target probability measure νπ and a simple reference
measure νη. If the densities exist the transport map is a deterministic coupling between
the target and reference density π and η, respectively. In fact, to sample easily from νπ
one chooses a common simple measure νη, e.g., a standard Gaussian or uniform measure.
The idea of the measure transport is, once the transport map is computed by solving
an optimization problem, to generate easily independent and unweighted samples from
the complicated target measure by sampling from the easy reference measure and then
pushing the samples forward through the map T. Transport maps are a powerful method
with a lot of applications. Being able to sample conditionals of an underlying distribution,
transport maps are an incredibly useful tool in Bayesian inference whereby the target
measure is the posterior measure of a quantity of interest given data. By computing an
indirect transport map we push forward the target to the reference measure via convex
optimization. One could then recover the direct transport map by inverting the indirect
transport map pointwise. In this work, we focus on constructing lower triangular transport
maps, i.e., an approximation of the Knothe-Rosenblatt-Rearrangement. This makes the
optimal transport formulation convex and separable, thus feasible to compute. Moreover,
the transport map becomes unique in construction. This bachelor thesis will describe the
theoretical foundation of the optimal transport map approach. By constructing a suitable
approximation of the transport map we show this approach by demonstrating a parameter
estimation example of a Bayesian inverse problem.
