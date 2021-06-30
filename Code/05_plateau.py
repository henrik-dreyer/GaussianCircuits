from differentiable_covariance import CovarianceMatrix, ising_energy_density
import numpy as np
from matplotlib import pyplot as plt
import os
from time import time
from jax import jit, grad
from scipy.optimize import minimize
import pandas as pd

#n=3
#execution_number = pd.DataFrame(np.array(allgrads).T, columns=['Execution Number'])

def cost_function(ts, L, p, Jx, Jz):
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

np.random.seed(1)
n_samples = 100
g=0.5

labels=[]
means=[]
vars=[]
Ls =[]
allgrads=[]
for LL in np.logspace(0.3,3,17):
    L=round(LL/2.)*2
    p=L//2
    Jz = [g]*L#np.random.rand(L)*2-1
    Jx = [1]*(L-1)#np.random.rand(L-1)*2-1
    cost_density = lambda ts: cost_function(ts, L, p, Jx, Jz).real/L
    cost_grad = grad(cost_density)

    grads=[]

    for j in range(n_samples):
        t0=time()
        print("Sample #{}".format(j))
        t = np.random.rand(p * (L - 1) + p * L) * np.pi / 2
        single_grad = cost_grad(t)
        ind_derivatice = len(single_grad)//2
        grads.append(single_grad[ind_derivatice].item())
        t1=time()
        print("Time per Sample = {}".format(t1-t0))

    allgrads.append(grads)
    labels.append('L={}'.format(L))
    means.append(np.mean(grads))
    vars.append(np.var(grads))
    Ls.append(L)

    #plt.figure(1)
    #plt.loglog(Ls, means,'o',label='Means')

    plt.figure(2)

    if len(Ls)>1:
        fit = np.polyfit(np.log(Ls), np.log(vars), 1)
        x = np.linspace(min(Ls), max(Ls), 100)
        y = np.exp(np.polyval(fit, np.log(x)))
        plt.loglog(x,y,'--',label='Fit: $var ~ L^{{{:.2f}}}$'.format(fit[0]))

    plt.title('Targeting $H=\sum_j X_j X_{j+1} + 0.5 Z_j$ with non-TI free square QAOA.')
    plt.loglog(Ls,vars, 'o', label='Variances')
    plt.xlabel('$L$')
    plt.legend(loc='best')
    plt.ylabel('var($ dE / dt)$ ({} samples and derivate w.r.t. central parameter)'.format(n_samples))
    plt.savefig('../Plots/ProducingScript=' + os.path.basename(__file__) + '_05.png')
    plt.show()

    df = pd.DataFrame(np.array(allgrads).T, columns=labels)
    df.to_csv('../Data/ProducingScript=' + os.path.basename(__file__) + '05_gradients.csv')
