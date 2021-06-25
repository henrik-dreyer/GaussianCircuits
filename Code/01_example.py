from covariance_matrix import CovarianceMatrix

psi = CovarianceMatrix(L=2)
print("Entropy before = {}".format(psi.entropy()))
psi.apply_random_matchgate()
print("Entropy after = {:.2f}".format(psi.entropy()))