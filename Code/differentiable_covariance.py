import jax.numpy as jnp
from jax import grad, jit, vmap
from jax.scipy.linalg import expm

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
            self.Gamma = jnp.diag( jnp.array([-1, 0]* (L-1) + [-1]), 1)+jnp.diag( jnp.array([+1, 0]* (L-1)+[+1]), -1)
        else:
            self.Gamma = Gamma_initial
        self.M = jnp.eye(2*L) + 1j*self.Gamma

    def apply_XX_layer(self, ts):
        """
        Propagates the state through a layer of \sum_j t_j X_j X_{j+1} gates (open boundaries)

        Parameters
        ----------
        ts: (List of Real numbers of size L-1) The times/angles. Set all equal for pseudo-translational-invariance
        """
        h = jnp.diag( jnp.array([item for items in zip([0] * len(ts),ts) for item in items] + [0]),1)
        h = h - h.T
        self.M = expm(2*h).dot(self.M).dot(expm(-2*h))

    def apply_Z_layer(self, ts):
        """
        Propagates the state through a layer of \sum_j t_j Z_j gates (open boundaries)

        Parameters
        ----------
        ts: (List of Real numbers of size L) The times/angles. Set all equal for pseudo-translational-invariance
        """
        h = jnp.diag( jnp.array([item for items in zip(ts,[0] * len(ts)) for item in items][:-1]) ,1)
        h = h - h.T
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
        h = jnp.diag( jnp.array([item for items in zip([0] * len(Jx),Jx) for item in items] + [0]) ,1)
        h = h + jnp.diag( jnp.array([item for items in zip(Jz,[0] * len(Jz)) for item in items][:-1]) ,1)
        h = h - h.T

        return jnp.trace(h.dot( -1j* self.M - jnp.eye(self.L*2)))/2


    def entropy(self, sites=None):
        """
        Entanglement entropy after tracing out 'sites'

        Parameters
        ----------
        sites: (List of Integers) The sites to be traced out (complement remains). Default = left half
        """

        if sites==None:
            sites = jnp.arange(self.L//2)
        modes_to_delete = [x*2 for x in sites] + [x*2+1 for x in sites]

        Gamma = -1j*(self.M - jnp.eye(2*self.L))
        Gamma = jnp.delete(Gamma, modes_to_delete, axis=0)
        Gamma = jnp.delete(Gamma, modes_to_delete, axis=1)

        eigvals = jnp.sort(jnp.linalg.eigh(-1j*Gamma)[0])[::-1]
        eigvals = eigvals[0:eigvals.shape[0] // 2]

        return ff_entropy(eigvals)

def ff_entropy(eigvals):
    eigvals = jnp.array([x for x in eigvals if abs(x - 1) > 1e-10])
    epsl = jnp.arctanh(eigvals) * 2
    entropy = jnp.sum(jnp.log(1 + jnp.exp(-epsl))) + jnp.sum(epsl / (jnp.exp(epsl) + 1))
    return entropy

def ising_energy_density(Jz, Jx):
    L = len(Jz)
    h = jnp.diag( jnp.array([item for items in zip([0] * len(Jx), Jx) for item in items] + [0]), 1)
    h = h + jnp.diag( jnp.array([item for items in zip(Jz, [0] * len(Jz)) for item in items][:-1]), 1)
    h = h - h.T
    eigvals = jnp.linalg.eigh(1j*h)[0]
    filled = jnp.array([num for num in eigvals if num < 0])
    energy_density = jnp.sum(filled)/L
    return energy_density