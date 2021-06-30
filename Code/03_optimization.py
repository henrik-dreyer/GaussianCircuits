from differentiable_covariance import CovarianceMatrix, ising_energy_density
import numpy as np
from matplotlib import pyplot as plt
import os
from time import time
from jax import jit, grad
from scipy.optimize import minimize



def cost_function(ts,L,p, Jx, Jz):
    '''
    Convention for ts: (L-1) t_j (XX), p times, then L t_j (Z), p times
    total number of parameters: p*(L-1) + p*L = 2*p*L - p
    '''
    psi = CovarianceMatrix(L)

    indXX = 0
    indZ = (L-1)*p

    for n in range(p):
        psi.apply_XX_layer(ts[indXX:indXX+L-1])
        psi.apply_Z_layer(ts[indZ:indZ + L])
        indXX = indXX + L-1
        indZ = indZ + L

    energy = psi.energy(Jx, Jz)
    return energy


p = 5
L = 10
g = 0.5
Jz = [g] * L
Jx = [1] * (L - 1)
exact_energy = ising_energy_density(Jz, Jx)

t_initial = np.random.rand(p*(L-1) + p*L)*np.pi/2
cost_density = lambda ts: cost_function(ts, L, p, Jx, Jz).real/L
cost_grad = grad(cost_density)

print(cost_density(t_initial).real)

topt = time()
time2solution = time()
res = minimize(cost_density, t_initial, method='BFGS', jac=cost_grad,
               options={'maxiter': 1e6, 'gtol': 1e-7, 'disp': True, 'eps': 6e-14})

time2solution = time() - time2solution

print("Optimization finished after {:.3f} seconds".format(time2solution))
print("Energy density at optimum = {}".format(res.fun))
print("Exact energy = {}".format(exact_energy))