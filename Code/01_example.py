import numpy as np
from covariance_matrix import CovarianceMatrix


psi = CovarianceMatrix(L=2)
print(psi.Gamma)
psi.apply_random_matchgate_layer()
print(psi.Gamma.round(3))
