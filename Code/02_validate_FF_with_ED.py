from coherent_state import SparseState
from matplotlib import pyplot as plt
import numpy as np
np.set_printoptions(precision=3)
n_precision=3
from scipy.linalg import expm

def ED(t):
    v=1/2*np.array([1,1,1,1])
    v=np.dot( expm(-1j*t*np.diag([1,-1,-1,1])),v)
    rho = np.outer(v,np.conj(v))
    return rho

print("\nDigits of precision = {}".format(n_precision))
L=2

ts=[]
Ss_exact=[]
Ss_ferm=[]
for t in np.linspace(0,np.pi/2,30):
    #t=np.pi/4
    ED(t)
    print("=" * 80)
    print("t={}".format(t))
    f_params = dict(L=L)
    f=SparseState(f_params)
    print("Setting up Coherent State with h={} on L={}".format(f.h,L))
    print("NS+ = {}".format(f.NS_plus))
    print("f(k) = {}".format(f.fk.round(n_precision)))
    print("="*40)

    print("Evolution")
    print("Changing basis to h=0")
    f.change_basis(0)
    print("f(k) = {}".format(f.fk.round(n_precision)))
    print("Evolving for t = {:.2f}, i.e. pi/{}".format(t, np.pi/t))
    f.eigenstate_evolve(t)
    print("f(k) = {}".format(f.fk.round(n_precision)))
    print("Changing basis to h=inf")
    f.change_basis('inf')
    print("f(k) = {}".format(f.fk.round(n_precision)))
    print("="*40)


    print("Setting Correlation Matrices")
    f.set_Ck()
    f.set_Fk()
    print("C_k = <c_k^\dagger c_k> = {}".format(f.Ck.round(n_precision)))
    print("F_k = <c_k c_-k> = {}".format(f.Ck.round(n_precision)))
    print("Fourier Transforming to Real Space using f(k) = -f(-k)")
    f.set_Cxy_Fxy()
    print("C_xy = <c_x^\dagger c_y> = \n {} \n".format(f.Cxy.round(n_precision)))
    print("F_xy = <c_x c_y> = \n {}".format(f.Fxy.round(n_precision)))
    print("="*40)
    print("Tracing out qubit 0")


    f.trace_out([0])
    print("Reduced C_xy = \n {} \n".format(f.Cxy.round(n_precision)))
    print("Reduced F_xy = \n {} \n".format(f.Fxy.round(n_precision)))
    print("In Majorana basis:")
    print("iGamma_mn = <a_m a_n> - Id = \n {} \n".format(f.iGamma.round(n_precision)))
    print("Eigenvalues =  {}".format(f.eigvals.round(n_precision)))
    print("Epsilon_l =  {}".format(f.epsl.round(n_precision)))

    print("Entropy =  {}".format(f.entropy))
    ts.append(t)
    Ss_ferm.append(f.entropy)

    li = np.array([1/2*(1-np.cos(2*t)),1/2*(1+np.cos(2*t))])
    print("Eigenvalues of reduced density matrix from ED={}".format(li))
    S_exact = -np.sum( li*np.log(li) )
    Ss_exact.append(S_exact)
    print("Entropy from ED=  {}".format(S_exact))

plt.title('Single-Qubit Entropy of $exp(-it Z_1 Z_2) |++>$')
plt.plot(ts,Ss_ferm,'o',label='Free Fermions')
plt.plot(ts,Ss_exact,'x',label='ED')
plt.legend(loc='best')
plt.xlabel('$t$')
plt.ylabel('$S$')
plt.savefig('01_Same_speed.png')
plt.show()
