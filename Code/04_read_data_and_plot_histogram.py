import numpy as np
import pandas as pd
import os
from matplotlib import pyplot as plt
import pylab
import scipy.stats as stats

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



# Using Scott's Rule
n_bins = int(( np.max(Ss_ferm) - np.min(Ss_ferm) ) * n ** (1 / 3) / (3.49 * np.sqrt(var)))
plt.figure(1)

plt.title('L={}, Depth=$2*{}$, #samples={}'.format(p, L, n))
n_out, bins, patches = plt.hist(np.sqrt(Ss_ferm), n_bins, density=True, alpha=0.75, label='Data')
x_values = np.linspace(avg - 4 * var, avg + 4 * var, 120)
y_values = gaussian(x_values, avg, np.sqrt(var))
#plt.plot(x_values, y_values, '-', label='Maximum Likelihood Gaussian')
plt.xlabel('$S$')
#plt.legend(loc='best')
plt.ylabel('Relative Probability')
plt.savefig('../Plots/ProducingScript=' + os.path.basename(__file__) + 'L={},p={}_Histogram.png'.format(L, p))
plt.show()