from covariance_matrix import CovarianceMatrix
import numpy as np
from matplotlib import pyplot as plt
import os

# This script produces a plot comparing the entropy
# obtained in two different ways from the bipartition 
# of a state which is evolved using a \sum XX Hamiltonian, 
# One using only covariance matrices, and the other using 
# analytical results.

# This is a sanity check to see if the covariance matrix machinery
# is working

# TODO: Transform this script into a unit test


# Stores different times state gets evolved for
ts=[]
# Stores entropy obtained from covariance matrices
Ss_ferm=[]
# Stores entropy obtained from analytical formula
Ss_exact=[]

for t in np.linspace(0, np.pi/2):
    # From covariance matrices
    psi = CovarianceMatrix(L=2)
    print("Entropy at start = {}".format(psi.entropy()))

    psi.apply_XX_layer(ts=[t])
    ts.append(t)
    Ss_ferm.append(psi.entropy())

    # Analytical result for probability spectrum of evolved state
    li = np.array([1 / 2 * (1 - np.cos(2 * t)), 1 / 2 * (1 + np.cos(2 * t))])
    print("Eigenvalues of reduced density matrix from ED={}".format(li))
    S_exact = -np.sum(li * np.log(li))
    Ss_exact.append(S_exact)


#Plot entropies

plt.title('Single-Qubit Entropy of $exp(-it Z_1 Z_2) |++>$')
plt.plot(ts,Ss_ferm,'o',label='Via Correlations')
plt.plot(ts,Ss_exact,'x',label='Via ED')
plt.legend(loc='best')
plt.xlabel('$t$')
plt.ylabel('$S$')
plt.savefig('../Plots/ProducingScript=' + os.path.basename(__file__) + '.png')
plt.show()
