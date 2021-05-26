import numpy as np
from scipy.special import erf
from scipy.linalg import eigh_tridiagonal
def SM_potential(r,R,L=19.0,R_l=4.,R_r=3.1,R_f=5.):
    f1 = 1./np.abs(L/2.-R)
    f2 = 1./np.abs(L/2.+R)
    f3 = -erf(np.abs(R-r)/R_f)/np.abs(R-r)
    f4 = -erf(np.abs(r-L/2.)/R_r)/np.abs(r-L/2.)
    f5 = -erf(np.abs(r+L/2.)/R_l)/np.abs(r+L/2.)
    return f1+f2+f3+f4+f5


L = 19.0

R_f = 5.
R_l = 4.
R_r = 3.1

dr = 0.1
dR = 0.1

r_array = np.arange(-L/2.+2*dr,L/2.-2*dr,dr)
R_array = np.arange(-L/2.+2*dR,L/2.-2*dR,dR)

Nr = np.size(r_array)


def hamiltonian(dr,Nr,r_array,R):
    H = np.zeros([Nr,Nr])

    H += SM_potential(r_array,R)*np.eye(Nr)
    
    H += np.eye(Nr)/dr**2
    H += -np.eye(Nr,k=1)/(2.*dr**2)
    H += -np.eye(Nr,k=-1)/(2.*dr**2)
    return H

# W,V = eigh_tridiagonal(SM_potential(r_array,R_array[int(Nr/2)]) + 1./dr**2, (-1./(2.*dr**2))*np.ones(Nr-1))
print(SM_potential(r_array,R_array[int(Nr/2)]))

