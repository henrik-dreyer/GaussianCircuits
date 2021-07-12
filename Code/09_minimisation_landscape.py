from collections import defaultdict
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
energies = defaultdict(list)
# Time taken for each sample
times = defaultdict(list)

# Track if header has alrady been written when storing results
write_header = True

sample_nr = 1000

for p in range(1, L+1):
    t_initial = np.random.rand(p * (L-1) + p * L) * jnp.pi/2

    cost_density = jit(lambda ts: cost_function(ts, L, p, Jx, Jz).real/L)

    cost_grad = jit(grad(cost_density))

    print("Starting optimization L={}, p={}".format(L,p))

    for i in range(sample_nr):

        t = np.random.rand(p * (L - 1) + p * L) * np.pi / 2

        # Comment/uncomment for global/local optimisation

        res_start = time()
        res = cost_density(t)
        tres = time() - res_start

        grad_start = time()
        g = cost_grad(t)
        tgrad = time() - grad_start

        print("Energy density at sample = {}".format(res))

        ps.append(p)
        energies[p].append(res)
        times[p].append(tres)

        df = pd.DataFrame([[Jz, Jx, t, exact_energy, p, res, g, tres, tgrad]], columns = ["jz", "jx", "angles", "exact_energy", "p", "val", "grad", "time_res", "time_grad"])
        df.to_csv('../Data/ProducingScript=' + os.path.basename(__file__) + '.csv', mode = "a", index_label = False, header = write_header)

        write_header = False

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
