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
    L: (int) System size
    """


    def __init__(self, L, Gamma_initial=None):
        self.L=L
        if Gamma_initial==None:
            self.Gamma = np.diag([-1, 0]* (L-1) + [-1], 1)+np.diag([+1, 0]* (L-1)+[+1], -1)
        else:
            self.Gamma = Gamma_initial
        self.M = np.eye(2*L) + 1j*self.Gamma

    def apply_XX_layer(self, ts):
        """
        Propagates the state through a layer of \sum_j t_j X_j X_{j+1} gates (open boundaries)

        Parameters
        ----------
        ts: (List of Real numbers of size L-1) The times/angles. Set all equal for pseudo-translational-invariance
        """
        #If you want to enforce no terms across the boundary:
        #h = np.diag([item for items in zip([0] * len(ts),ts) for item in items] + [0],1)
        #h = h - h.T
        h = make_gXX(ts)
        self.M = expm(2*h).dot(self.M).dot(expm(-2*h))

    def apply_Z_layer(self, ts):
        """
        Propagates the state through a layer of \sum_j t_j Z_j gates (open boundaries)

        Parameters
        ----------
        ts: (List of Real numbers of size L) The times/angles. Set all equal for pseudo-translational-invariance
        """
        h = make_hZ(ts)
        self.M = expm(2*h).dot(self.M).dot(expm(-2*h))


    def energy(self, Jx, Jz):
        """
        Energy with respect to an OBC Ising Hamiltonian of the form
            H = \sum_{j} Jx_j X_j X_{j+1} + J_z_j Z_j

        Parameters
        ----------
        Jx: (List of L-1 Reals) Coupling strengths of X_j X_{j+1} terms
        Jz: (List of L Reals) Transverse field strengths Z_j
        """
        h = make_gXX(Jx)
        h = h + make_hZ(Jz)
        return np.trace(h.dot(-1j * self.M - np.eye(self.L * 2))) / 2


    def entropy(self, sites=None):
        """
        Entanglement entropy after tracing out 'sites'

        Parameters
        ----------
        sites: (List of Integers) The sites to be traced out (complement remains). Default = left half
        """

        if sites==None:
            sites = np.arange(self.L//2)
        modes_to_delete = [x*2 for x in sites] + [x*2+1 for x in sites]

        Gamma = -1j*(self.M - np.eye(2*self.L))
        Gamma = np.delete(Gamma, modes_to_delete, axis=0)
        Gamma = np.delete(Gamma, modes_to_delete, axis=1)

        eigvals = np.sort(np.linalg.eigh(-1j*Gamma)[0])[::-1]
        eigvals = eigvals[0:eigvals.shape[0] // 2]

        return ff_entropy(eigvals)

def ff_entropy(eigvals):
    eigvals = np.array([x for x in eigvals if abs(x - 1) > 1e-10])
    epsl = np.arctanh(eigvals) * 2
    entropy = np.sum(np.log(1 + np.exp(-epsl))) + np.sum(epsl / (np.exp(epsl) + 1))
    return entropy

def ising_energy_density(Jz, Jx):
    L = len(Jz)
    h = make_gXX(Jx)
    h = h + make_hZ(Jz)
    eigvals = np.linalg.eigh(1j*h)[0]
    filled = [num for num in eigvals if num < 0]
    energy_density = np.sum(filled)/L
    return energy_density

def make_gXX(ts):
    h = np.diag([item for items in zip([0] * (len(ts)-1), ts[:-1]) for item in items] + [0], 1)
    h[0,-1]  = ts[-1]
    h = h - h.T
    return h

def make_hZ(ts):
    h = np.diag([item for items in zip(ts, [0] * len(ts)) for item in items][:-1], 1)
    h = h - h.T
    return h