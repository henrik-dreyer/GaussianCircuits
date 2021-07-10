from covariance_matrix import CovarianceMatrix, ising_energy_density
import numpy as np
from matplotlib import pyplot as plt
import os
from time import time
from scipy.optimize import minimize
import pandas as pd

def cost_function(ts,L,p, Jx, Jz):
    '''
    Convention for ts: (L-1) t_j (XX), p times, then L t_j (Z), p times
    total number of parameters: p*(L-1) + p*L = 2*p*L - p
    '''
    psi = CovarianceMatrix(L)

    indXX = 0
    indZ = (L-1)*p

    for n in range(p):
        psi.apply_XX_layer(np.append(ts[indXX:indXX + L-1],0))
        psi.apply_Z_layer(ts[indZ:indZ + L])
        indXX = indXX + L-1
        indZ = indZ + L

    energy = psi.energy(Jx, Jz)
    return energy


L = 8

np.random.seed(11)
Jz = np.random.rand(L)
Jx = np.append(np.random.rand(L-1),0)
exact_energy = ising_energy_density(Jz, Jx)

ps=[]
energies=[]
nits=[]
for p in range(1,L+1):
    t_initial = np.random.rand(p*(L-1) + p*L)*np.pi/2
    cost_density = lambda ts: cost_function(ts, L, p, Jx, Jz).real/L

    print("Starting optimization L={}, p={}".format(L,p))
    topt = time()
    time2solution = time()
    res = minimize(cost_density, t_initial, method='Nelder-Mead',
                   options={'maxiter': 1e9, 'disp': True})

    time2solution = time() - time2solution

    print("Optimization finished after {:.3f} seconds".format(time2solution))
    print("Energy density at optimum = {}".format(res.fun))
    print("Exact energy = {}".format(exact_energy))

    ps.append(p)
    energies.append(res.fun)
    nits.append(res.nit)

    df = pd.DataFrame([[p, res.fun, res.nit, time2solution, res.nfev]], columns = ["p", "minimum", "iterations", "time", "nfev"])
    df.to_csv('../Data/ProducingScript=' + os.path.basename(__file__) + '.csv', mode = "a")


# Uncomment to plot results as they are computed

#    if len(ps)>1:
#        plt.figure(1)
#        plt.title('Targeting $H=\sum_j (randX_j) X_j X_{j+1} + (randZ_j) Z_j$ with non-TI free QAOA. L=6')
#        plt.loglog(ps,[x-exact_energy for x in energies], 'o')
#        plt.xlabel('$p$')
#        plt.ylabel('$E- E_{exact}$')
#        plt.savefig('../Plots/ProducingScript=' + os.path.basename(__file__) + '_energy.png')
#        plt.show()
#
#        plt.figure(2)
#        fit = np.polyfit(np.log(ps), np.log(nits), 1)
#        x = np.linspace(min(ps), max(ps), 100)
#        y = np.exp(np.polyval(fit, np.log(x)))
#
#        plt.title('Targeting $H=\sum_j (randX_j) X_j X_{j+1} + (randZ_j) Z_j$ with non-TI free QAOA. ' + f'L={L}')
#        plt.loglog(x,y,'--',label='Fit: #iter ~ $p^{{{:.2f}}}$'.format(fit[0]))
#        plt.loglog(ps,nits,'o',label='Data')
#        plt.xlabel('$p$')
#        plt.ylabel('$n_{iterations}$ of Nelder-Mead')
#        plt.legend(loc='best')
#        plt.savefig('../Plots/ProducingScript=' + os.path.basename(__file__) + '_iterations.png')
#        plt.show()
