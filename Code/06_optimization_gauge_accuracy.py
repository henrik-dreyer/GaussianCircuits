from covariance_matrix import CovarianceMatrix, ising_energy_density
import numpy as np
from matplotlib import pyplot as plt
import os
from jax import jit
from time import time
from scipy.optimize import minimize

def cost_function(ts,L,p, Jx, Jz):
    '''
    Convention for ts: (L-1) t_j (XX), p times, then L t_j (Z), p times
    total number of parameters: p*(L-1) + p*L = 2*p*L - p
    '''
    psi = CovarianceMatrix(L)

    indXX = 0
    indZ = L*p

    for n in range(p):
        psi.apply_XX_layer(ts[indXX:indXX + L])
        psi.apply_Z_layer(ts[indZ:indZ + L])
        indXX = indXX + L
        indZ = indZ + L

    energy = psi.energy(Jx, Jz)
    return energy


L = 4
p = L//2
g = 0.5
Jz = [g] * L
Jx = [1] * L
exact_energy = ising_energy_density(Jz, Jx)

np.random.seed(100)
cost_density = lambda ts: cost_function(ts, L, p, Jx, Jz).real/L


print("Starting optimization L={}, p={}".format(L,p))

for n in range(10):
    t_initial = np.random.rand(2 * p * L) * np.pi / 2
    time2solution = time()
    res = minimize(cost_density, t_initial, method='Nelder-Mead',
                   options={'maxiter': 1e6, 'disp': True})

    time2solution = time() - time2solution

    print("Optimization finished after {:.3f} seconds".format(time2solution))
    print("Energy density at optimum = {}".format(res.fun))
    print("Exact energy = {}".format(exact_energy))