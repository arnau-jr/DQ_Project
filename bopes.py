import numpy as np
from scipy.special import erf

def SM_potential(r,R,L=1.,R_l=1.,R_r=1.,R_f=1.):
    f1 = 1./np.abs(L/2.-R)
    f2 = 1./np.abs(L/2.+R)
    f3 = -erf(np.abs(R-r)/R_f)/np.abs(R-r)
    f4 = -erf(np.abs(r-L/2.)/R_r)/np.abs(r-L/2.)
    f5 = -erf(np.abs(r+L/2.)/R_l)/np.abs(r+L/2.)
    return f1+f2+f3+f4+f5