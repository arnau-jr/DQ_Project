import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erf
from scipy.linalg import eigh_tridiagonal,eigh


#Discretization and array definitions
L = 19.0

R_f = 5.
R_l = 4.
R_r = 3.1

dr = 0.1
dR = 0.1

r_array = np.arange(-L,L,dr)
R_array = np.arange(-6.,6.,dR)

Nr = np.size(r_array)
NR = np.size(R_array)

#Hamiltonian and potential definitions

def SM_potential(r,R,L=19.0,R_l=4.,R_r=3.1,R_f=5.):
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
    return f1+f2+f3+f4+f5

# def hamiltonian(dr,Nr,r_array,R):
#     H = np.zeros([Nr,Nr])

#     H += SM_potential(r_array,R)*np.eye(Nr)
    
#     H += np.eye(Nr)/dr**2
#     H += -np.eye(Nr,k=1)/(2.*dr**2)
#     H += -np.eye(Nr,k=-1)/(2.*dr**2)
#     return H

def hamiltonian(dr,Nr,r_array,R):
    H = np.zeros([Nr,Nr])

    H += SM_potential(r_array,R)*np.eye(Nr)
    
    #Diagonal
    H += (-49./18.)*-np.eye(Nr)/(2.*dr**2)

    #First off diagonal
    H += (3./2.)*-np.eye(Nr,k=1)/(2.*dr**2)
    H += (3./2.)*-np.eye(Nr,k=-1)/(2.*dr**2)

    #Second off diagonal
    H += (-3./20.)*-np.eye(Nr,k=2)/(2.*dr**2)
    H += (-3./20.)*-np.eye(Nr,k=-2)/(2.*dr**2)

    #Third off diagonal
    H += (1./90.)*-np.eye(Nr,k=3)/(2.*dr**2)
    H += (1./90.)*-np.eye(Nr,k=-3)/(2.*dr**2)
    return H

#Generation of the BOPEs

N_states = 3 #How many BOPE states we want to save

eigenvalues = np.zeros([NR,N_states])
eigenstates = np.zeros([NR,Nr,N_states])

print("Diagonalizing ",Nr," by ",Nr," matrices")
for i in range(0,NR):
    print(i," of ",NR,end='\r')
    W,V = eigh(hamiltonian(dr,Nr,r_array,R_array[i]))

    for j in range(0,N_states):
        eigenvalues[i,j] = W[j]
        eigenstates[i,:,j] = V[:,j]


with open("eigenvalues.npy","wb") as f:
    np.save(f,eigenvalues)

with open("eigenvstates.npy","wb") as f:
    np.save(f,eigenstates)

plt.xlabel(r"$R(a_0)$")
plt.ylabel(r"$E(E_H)$")
plt.plot(R_array,eigenvalues[:,0],label="Ground state")
plt.plot(R_array,eigenvalues[:,1],label="First excited state")
plt.plot(R_array,eigenvalues[:,2],label="Second excited state")
plt.legend()
plt.savefig("bopes.png")
plt.close()


#Computing Non-Adiabatic couplings

def get_nonadiabatic_couplings(NR,dr,N_states,eigenstates,M=1836.152673):
    S = np.zeros([NR,N_states,N_states])

    for iR in range(0,NR):
        for i in range(0,N_states):
            for j in range(i,N_states):
                phi_plus = np.roll(eigenstates[iR,:,i],1)
                phi_plus[-1] = 0.
                phi_minus = np.roll(eigenstates[iR,:,i],-1)
                phi_minus[0] = 0.

                #Apply laplacian and integrate (assuming phi=0. at the edges)
                S[iR,i,j] = dr*np.sum(eigenstates[iR,:,j]*(1./M)\
                    *((eigenstates[iR,:,i] - 0.5*phi_plus - 0.5*phi_minus)/dr**2))

    return S

S = get_nonadiabatic_couplings(NR,dr,N_states,eigenstates)

with open("non_adiabatic_coupling.npy","wb") as f:
    np.save(f,S)