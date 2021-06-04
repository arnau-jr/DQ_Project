import numpy as np
import matplotlib.pyplot as plt
from scipy import linalg as la
from scipy.special import erf

def SM_potential(r,R,L=19.0,R_l=4.,R_r=3.1,R_f=5.):

    """
    Computation of the Shin-Metiu (SM) potential. The electron
    is embedded in r-space and the moving ion is embedded in
    R-space. Note the difference of lower and upper case 
    notation of each. Each part of the function (f1,f2,f3,f4,f5) 
    is representing one term of the potential. And for the cases
    that the system would blow up, the limits of the functions
    are used (f3,f4,f5).

        Args:
            r: 1D np.array, r-space vector
            R: float number, specific point in R-space
            L: float number, size of the system 
            R_l: float number, parameter given for this regime
            R_r: float number, parameter given for this regime
            R_f: float number, parameter given for this regime

        Returns:
            SM: 1D np.array, SM-potential vector over r-space
    """

    f1 = 1./np.abs(L/2.-R)
    f2 = 1./np.abs(L/2.+R)

    f3 = np.where(np.abs(R-r)<1e-8,\
          -2./(R_f*np.sqrt(np.pi)),\
           -erf(np.abs(R-r)/R_f)/np.abs(R-r))

    f4 = np.where(np.abs(r-L/2.)<1e-8,\
        -2./(R_r*np.sqrt(np.pi)),\
        -erf(np.abs(r-L/2.)/R_r)/np.abs(r-L/2.))

    f5 = np.where(np.abs(r+L/2.)<1e-8,\
        -2./(R_l*np.sqrt(np.pi)),\
        -erf(np.abs(r+L/2.)/R_l)/np.abs(r+L/2.))

    SM = f1+f2+f3+f4+f5

    return SM

## changed discretization step, otherwise too big of a matrix 
dr = 0.25
dR = 0.25
r_array = np.arange(-19,19,dr)
R_array = np.arange(-9,9,dR)

## with mesgrid two arrays can be parsed through the potential function
## without the same size
r_arr, R_arr = np.meshgrid(r_array, R_array)
Nr = np.size(r_array)
NR = np.size(R_array)
pot = SM_potential(r=r_arr, R=R_arr)
#print(pot.shape)

## reshape the potential vector and transfrom into N*N matrix
N = Nr*NR
pot_new = pot.reshape(N)*np.eye((N))

def laplacian(dr,N):

    """
        The Laplacian is represented as a vector of shape N*N.

        Args: 
            dr: float number, discretization step in space
            N: int number, size of the laplacian matrix

        Returns: 
            L: 2D np.array, matrix with upper-, lower- and diagonal
               values up to third order. 
    """
    
    L = np.zeros((N,N))
    
    #Diagonal
    L += (-49./18.)*-np.eye(N)/(2.*dr**2)

    #First off diagonal
    L += (3./2.)*-np.eye(N,k=1)/(2.*dr**2)
    L += (3./2.)*-np.eye(N,k=-1)/(2.*dr**2)

    #Second off diagonal
    L += (-3./20.)*-np.eye(N,k=2)/(2.*dr**2)
    L += (-3./20.)*-np.eye(N,k=-2)/(2.*dr**2)

    #Third off diagonal
    L += (1./90.)*-np.eye(N,k=3)/(2.*dr**2)
    L += (1./90.)*-np.eye(N,k=-3)/(2.*dr**2)
        
    return L


## compute laplacian for the kinetic parts and put the into the same space
## with the kronecker product
m = 1
M = 1836.152673

Te = laplacian(dr,Nr)/m
Tp = laplacian(dR,NR)/M

Te_full = np.kron(Te,np.eye(NR))
Tp_full = np.kron(np.eye(Nr),Tp)

## put the terms together and you get the full hamiltonian for the system
full_hamiltonian = pot_new + Te_full + Tp_full
#print(full_hamiltonian.shape)

def wave_packet(x,x0=-4.,sigma=1/np.sqrt(2.85)):
    """
    Initializing the Gaussian wavepacket around x0 = -4 and with sigma = 1/sqrt(2.85) at the ground state
    of the system. Instead of splitting the function in real and imaginary part, the wavepacket 
    is initialized with both parts by multiplying with np.exp(1j+1).
    """
    return np.exp(-(x-x0)**2/(2*sigma**2))/np.sqrt(np.sqrt(np.pi)*sigma)


eigenstates = np.load("eigenvstates.npy")
ex1 = eigenstates[:,:,1]
psi0 = np.zeros([Nr*NR])

for i in range(0,Nr):
    for j in range(0,NR):
        psi0[NR*i+j] = ex1[np.where(R_array==-4.)[0],i]*wave_packet(R_array[j])

#print(psi0.shape)

def compute_f(hamiltonian, wave):
    return hamiltonian@wave

def evolve_psi_RK4(dt,hamiltonian,wave):

    k1 = compute_f(hamiltonian,wave)
    k2 = compute_f(hamiltonian,wave + (dt/2.)*k1)
    k3 = compute_f(hamiltonian,wave + (dt/2.)*k2)
    k4 = compute_f(hamiltonian,wave +  dt*k3    )
    return wave + (dt/6.)*(k1 + 2.*k2 + 2.*k3 + k4)

dt = 1e-2
#endtime = 30/(2.418884e-2)
endtime = 10
time = np.arange(0,endtime,dt)
time_len = np.size(time)
psi_len = np.size(psi0)
psi_evolved = np.zeros((int(time_len/1000), psi_len))

for i in range(time_len):
    print(i," of ",time_len,end='\r')
    if i%1000 == 0:
        psi_evolved[int(i/1000),:] = evolve_psi_RK4(dt,full_hamiltonian,psi0)

with open("psi_evolved.npy","wb") as f:
    np.save(f,psi_evolved)