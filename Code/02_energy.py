from covariance_matrix import ising_energy_density
import numpy as np
from matplotlib import pyplot as plt
import os
from scipy.integrate import quad

def e(h, p):
  return -np.sqrt(1 + h ** 2 - 2 * h * np.cos(p)) / 2 / np.pi

def exact_energy(h):
  hval = h
  func = lambda p: e(hval, p)
  result = quad(func, -np.pi, +np.pi)#, epsabs=1e-20, limit=200)
  return result[0]



for L in [10,20,40,80]:
    energies = []
    exact_pbc_energies = []
    gs = []
    for g in np.linspace(0,1.6):
        Jz = [g]*L
        Jx = [1]*(L-1)

        energies.append(ising_energy_density(Jz,Jx))
        exact_pbc_energies.append(exact_energy(g))
        gs.append(g)
    plt.plot(gs, energies, 'o', markersize=4, label='Via Covariance on {} sites'.format(L))

plt.title('Energy density of $H=\sum_j X_j X_{j+1} + g Z_j$')
plt.plot(gs,exact_pbc_energies,'-',linewidth=3, label='Exact value in Thermodynamic Limit')
plt.legend(loc='best')
plt.xlabel('$g$')
plt.ylabel('$E/L$')
plt.savefig('../Plots/ProducingScript=' + os.path.basename(__file__) + '.png')
plt.show()
