from covariance_matrix import CovarianceMatrix
import numpy as np
from matplotlib import pyplot as plt
import os

ts=[]
Ss_ferm=[]
Ss_exact=[]
for t in np.linspace(0,np.pi/2):
    psi = CovarianceMatrix(L=2)
    print("Entropy at start = {}".format(psi.entropy()))

    psi.apply_XX_layer(ts=[t])
    ts.append(t)
    Ss_ferm.append(psi.entropy())

    #From ED
    li = np.array([1 / 2 * (1 - np.cos(2 * t)), 1 / 2 * (1 + np.cos(2 * t))])
    print("Eigenvalues of reduced density matrix from ED={}".format(li))
    S_exact = -np.sum(li * np.log(li))
    Ss_exact.append(S_exact)


plt.title('Single-Qubit Entropy of $exp(-it Z_1 Z_2) |++>$')
plt.plot(ts,Ss_ferm,'o',label='Via Correlations')
plt.plot(ts,Ss_exact,'x',label='Via ED')
plt.legend(loc='best')
plt.xlabel('$t$')
plt.ylabel('$S$')
plt.savefig('../Plots/ProducingScript=' + os.path.basename(__file__) + '.png')
plt.show()
