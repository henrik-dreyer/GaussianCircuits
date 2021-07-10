from differentiable_covariance import CovarianceMatrix, ising_energy_density
import jax.numpy as jnp
import numpy as np
from jax import grad, jit
from matplotlib import pyplot as plt
import os
from time import time
from scipy.optimize import basinhopping, minimize
import pandas as pd

def cost_function(ts, L, p, Jx, Jz):
    '''
    Convention for ts: (L-1) t_j (XX), p times, then L t_j (Z), p times
    total number of parameters: p*(L-1) + p*L = 2*p*L - p
    '''
    psi = CovarianceMatrix(L)

    indXX = 0
    indZ = (L-1)*p

    for n in range(p):
        psi.apply_XX_layer(jnp.append(ts[indXX:indXX + L-1],0))
        psi.apply_Z_layer(ts[indZ:indZ + L])
        indXX = indXX + L-1
        indZ = indZ + L

    energy = psi.energy(Jx, Jz)
    return energy

# Script parameters
L = 8
np.random.seed(11)

# Generate parameters for random Ising Hamiltonian
Jz = np.random.rand(L)
Jx = jnp.append(np.random.rand(L-1), 0)

# Get analytical energy density for Ising Hamiltonian
exact_energy = ising_energy_density(Jz, Jx)

# Track values of p already computed
ps = []
# Optimal energies for each value of p
energies = []
# Number of iterations needed to converge for each value of p
nits = []

for p in range(1, L+1):
    t_initial = np.random.rand(p * (L-1) + p * L) * jnp.pi/2

    cost_density = jit(lambda ts: cost_function(ts, L, p, Jx, Jz).real/L)

    cost_grad = jit(grad(cost_density))

    print("Starting optimization L={}, p={}".format(L,p))
    topt = time()
    time2solution = time()

    # Comment/uncomment for global/local optimisation

    #res = basinhopping(cost_density, t_initial, minimizer_kwargs = {"method": "BFGS", "options": {"maxiter": 1e9}, "jac": cost_grad})
    res = minimize(cost_density, t_initial, method = "BFGS", options = {"maxiter": 1e9}, jac = cost_grad)


    time2solution = time() - time2solution

    print("Optimization finished after {:.3f} seconds".format(time2solution))
    print("Energy density at optimum = {}".format(res.fun))
    print("Exact energy = {}".format(exact_energy))

    ps.append(p)
    energies.append(res.fun)
    nits.append(res.nit)

    df = pd.DataFrame([[p, res.fun, res.nit, time2solution, res.nfev, res.njev]], columns = ["p", "minimum", "iterations", "time", "nfev", "njev"])
    df.to_csv('../Data/ProducingScript=' + os.path.basename(__file__) + '.csv', mode = "a")

#    Uncomment to produce plots as results are produced

#    # Produce plot if we have more than one value of p computed
#    if len(ps)>1:
#        plt.figure(1)
#        plt.title('Targeting $H=\sum_j (randX_j) X_j X_{j+1} + (randZ_j) Z_j$ with non-TI free QAOA. L=6')
#        plt.loglog(ps, [x-exact_energy for x in energies], 'o')
#        plt.xlabel('$p$')
#        plt.ylabel('$E- E_{exact}$')
#        plt.savefig('../Plots/ProducingScript=' + os.path.basename(__file__) + '_energy.png')
#        plt.show()
#
#        plt.figure(2)
#        fit = jnp.polyfit(jnp.log(ps), jnp.log(nits), 1)
#        x = jnp.linspace(min(ps), max(ps), 100)
#        y = jnp.exp(jnp.polyval(fit, jnp.log(x)))
#        
#        plt.title('Targeting $H=\sum_j (randX_j) X_j X_{j+1} + (randZ_j) Z_j$ with non-TI free QAOA. ' + f'L={L}')
#        plt.loglog(x,y,'--',label='Fit: #iter ~ $p^{{{:.2f}}}$'.format(fit[0]))
#        plt.loglog(ps,nits,'o',label='Data')
#        plt.xlabel('$p$')
#        plt.ylabel('$n_{iterations}$ of BFGS')
#        plt.legend(loc='best')
#        plt.savefig('../Plots/ProducingScript=' + os.path.basename(__file__) + '_iterations.png')
#        plt.show()
