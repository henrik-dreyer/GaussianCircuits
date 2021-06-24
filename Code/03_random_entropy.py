from coherent_state import SparseState
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import pathlib
import os
import math

np.set_printoptions(precision=3)
n_precision=3
print("\nDigits of display precision = {}".format(n_precision))
n_samples = 100000
L=50
p=L
Ss_ferm=[]
ns=[]
loglikelihoods=[]

def gaussian(x, mu, sig):
    return  (np.sqrt(2*np.pi)*sig)**(-1) * np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))

for n in range(1,n_samples):

    ts = np.random.rand(2*p)*np.pi/2
    f_params = dict(L=L)
    f=SparseState(f_params)
    #print("Setting up Coherent State with h={} on L={}".format(f.h,L))

    for j in range(0,2*p,2):
        #print("Evolution")
        #print("Changing basis to h=0")
        f.change_basis(0)
        #print("f(k) = {}".format(f.fk.round(n_precision)))
        #print("Evolving for t = {:.2f}".format(ts[j]))
        f.eigenstate_evolve(ts[j])
        #print("Changing basis to h=inf")
        f.change_basis('inf')
        #print("Evolving for t = {:.2f}".format(ts[j+1]))
        f.eigenstate_evolve(ts[j+1])
        #print("="*40)

    #print("Computing Entropy")
    f.set_Ck()
    f.set_Fk()
    f.set_Cxy_Fxy()
    f.trace_out(np.arange(L//2))

    Ss_ferm.append(f.entropy)

    if n%500==0:
        print('Sample #{}'.format(n))

        Ss_ferm_np = np.array(Ss_ferm)[~np.isnan(Ss_ferm)]

        nnans = sum(math.isnan(x) for x in Ss_ferm)
        print('Number of nan={}'.format(nnans))
        n_clean = n-nnans

        avg = np.nanmean(Ss_ferm_np)
        var = np.nanvar(Ss_ferm_np)
        #loglikelihood = -n/2*np.log(2*np.pi) - n/2*np.log(var) - 1/2/var * np.sum( (Ss_ferm-avg)**2)
        #ns.append(n)
        #loglikelihoods.append(loglikelihood)

        df = pd.DataFrame(Ss_ferm_np, columns=['Half_Chain_Entropy'])
        df.to_csv('../Data/ProducingScript='+os.path.basename(__file__)+'L={},p={}.csv'.format(L,p))



        plt.figure(2)
        plt.title('L={}, Depth=$2*{}$, #samples={}'.format(p,L,n_clean))
        plt.plot(ns, loglikelihoods,'o')
        plt.xlabel('$n_{{Samples}}$')
        plt.ylabel('Log-Likelihood to Gaussian')
        plt.savefig('../Plots/ProducingScript='+os.path.basename(__file__)+'L={},p={}_Likelihood.png'.format(L,p))
        plt.show()


        #Using Scott's Rule
        n_bins = int( (np.nanmax(Ss_ferm_np) - np.nanmin(Ss_ferm_np)) * n_clean**(1/3) / (3.49 * np.sqrt(var)) )
        plt.figure(1)
        plt.title('L={}, Depth=$2*{}$, #samples={}'.format(p,L,n_clean))

        var = var * n/(n-1)


        n_out, bins, patches = plt.hist(Ss_ferm_np, n_bins, density=True, alpha=0.75,label='Data')

        x_values = np.linspace(avg-4*var, avg+4*var, 120)
        y_values = gaussian(x_values, avg, np.sqrt(var))
        plt.plot(x_values, y_values,'-',label='Maximum Likelihood Gaussian')
        plt.xlabel('$S$')
        plt.legend(loc='best')
        plt.ylabel('Relative Probability')
        plt.savefig('../Plots/ProducingScript='+os.path.basename(__file__)+'L={},p={}_Histogram.png'.format(L,p))
        plt.show()





print('done')