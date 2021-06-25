import numpy as np
from scipy.linalg import expm

class CovarianceMatrix():

    """
    This class provides the interface for propagating a state
    through a 1D Gaussian circuit and compute entanglement properties


    Parameters
    ----------
    L: (int) System size


    Attributes
    ----------
    Q : Fredholm discretization. Must be a power of 2, Q = 2^q

    R : Effective system size. Must be an half-odd integer multiple of Q,
                                R = (2r+1)/2 * Q

    f_initial: A function f(k) to populate initial values.

    lambda_grid: Array on which f(lambda) is evaluated.
    mu1_grid: Array on which f(mu1) is evaluated.
    mu2_grid: Array on which f(mu2) is evaluated.
    k_grid: Array on which f(k) is evaluated. Grids are spaced
            in such a way as to avoid numerical singularities
            when computing Fredholm determinants.
    """


    def __init__(self, L, Gamma_initial=None):
        self.L=L
        if Gamma_initial==None:
            self.Gamma = np.diag([-1, 0]* (L-1) + [-1], 1)+np.diag([+1, 0]* (L-1)+[+1], -1)
        else:
            self.Gamma = Gamma_initial

    def apply_random_matchgate(self):
        h = np.random.rand(2*self.L,2*self.L)
        h = h - h.T
        self.Gamma = expm(4*h).dot(self.Gamma).dot( expm(-4*h) )

        #Not normalised



    def entropy(self, sites=None):
        """
        Entanglement entropy after tracing out 'sites'

        Parameters
        ----------
        sites: (List of Integers) The sites to be traced out (complement remains)
        """

        if sites==None:
            sites = np.arange(self.L//2)
        modes_to_delete = [x*2 for x in sites] + [x*2+1 for x in sites]

        Gamma = np.delete(self.Gamma, modes_to_delete, axis=0)
        Gamma = np.delete(Gamma, modes_to_delete, axis=1)

        eigvals = np.sort(np.linalg.eigh(-1j*Gamma)[0])[::-1]
        eigvals = eigvals[0:eigvals.shape[0] // 2]

        return ff_entropy(eigvals)


def ff_entropy(eigvals):
    eigvals = np.array([x for x in eigvals if abs(x - 1) > 1e-10])
    epsl = np.arctanh(eigvals) * 2
    entropy = np.sum(np.log(1 + np.exp(-epsl))) + np.sum(epsl / (np.exp(epsl) + 1))
    return entropy