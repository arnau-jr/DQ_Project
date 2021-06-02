import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation

def wave_packet(x,x0=-4.,sigma=1/np.sqrt(2.85)):
    """
    Initializing the Gaussian wavepacket around x0 = -4 and with sigma = 1/sqrt(2.85) at the ground state
    of the system. Instead of splitting the function in real and imaginary part, the wavepacket 
    is initialized with both parts by multiplying with np.exp(1j+1).
    """
    return np.cdouble(np.exp(-(x-x0)**2/(2*sigma**2))/np.sqrt(np.sqrt(np.pi)*sigma))

## parameters and preliminaries
dR = 0.1
R_array = np.arange(-9.,9.,dR)
dt = 1e-4

## loading the saved arrays from the bopes step
non_adiabatic = np.load("non_adiabatic_coupling.npy")
eigenstates = np.load("eigenvstates.npy")
eigenvalues = np.load("eigenvalues.npy")

NR,Nr,N_states = eigenstates.shape

psi0 = np.zeros([Nr*NR])
for i in range(0,Nr):
    for j in range(0,NR):
        psi0[NR*i+j] = eigenstates[np.where(R_array==-4.)[0],i,1]*wave_packet(R_array[j])

print(eigenstates)


def compute_f(dx,m,x_array,psi,V,**kwargs):
    #Trick to do the psi(x+dx) and psi(x-dx)
    psi_plus = np.roll(psi,1)
    psi_plus[-1] = 0.
    psi_minus = np.roll(psi,-1)
    psi_minus[0] = 0.

    #Computation of the right hand side of Schr. eq.
    f = 1j*(psi_plus - 2*psi + psi_minus)/(2.*m*dx**2) - 1j*V(x_array,**kwargs)*psi
    return f

def evolve_psi_RK4(dt,dx,m,x_array,psi,V,**kwargs):

    k1 = compute_f(dx,m,x_array,psi             ,V,**kwargs)
    k2 = compute_f(dx,m,x_array,psi + (dt/2.)*k1,V,**kwargs)
    k3 = compute_f(dx,m,x_array,psi + (dt/2.)*k2,V,**kwargs)
    k4 = compute_f(dx,m,x_array,psi +  dt*k3    ,V,**kwargs)
    return psi + (dt/6.)*(k1 + 2.*k2 + 2.*k3 + k4)
