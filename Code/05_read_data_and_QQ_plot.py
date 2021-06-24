import numpy as np
import pandas as pd
import os
from matplotlib import pyplot as plt
import pylab
import scipy.stats as stats

from scipy.stats import chi2
L=100
p=L
file = '../Data/ProducingScript=03_random_entropy.pyL={},p={}.csv'.format(L,p)
data = pd.read_csv(file)
Ss_ferm = data['Half_Chain_Entropy'].values.tolist()
avg = np.mean(Ss_ferm)
var = np.var(Ss_ferm)
n = len(Ss_ferm)
var = var * n / (n - 1)

def gaussian(x, mu, sig):
    return (np.sqrt(2*np.pi)*sig)**(-1) * np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))


stats.probplot( np.array(Ss_ferm), dist="norm", plot=plt)
plt.title('QQ-Plot. L={}, Depth=$2*{}$, #samples={}'.format(p, L, n))
#plt.xlabel('$S$')
plt.legend(loc='best')
#plt.ylabel('Relative Probability')
plt.savefig('../Plots/ProducingScript=' + os.path.basename(__file__) + 'non_root_L={},p={}_QQ-plot.png'.format(L, p))
plt.show()