import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erf
from scipy.linalg import eigh_tridiagonal,eigh
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


def hamiltonian(dr,Nr,r_array,R):
    H = np.zeros([Nr,Nr])

    H += SM_potential(r_array,R)*np.eye(Nr)
    
    H += np.eye(Nr)/dr**2
    H += -np.eye(Nr,k=1)/(2.*dr**2)
    H += -np.eye(Nr,k=-1)/(2.*dr**2)
    return H

# def hamiltonian(dr,Nr,r_array,R):
#     H = np.zeros([Nr,Nr])

#     H += SM_potential(r_array,R)*np.eye(Nr)
    
#     #Diagonal
#     H += (-49./18.)*-np.eye(Nr)/(2.*dr**2)

#     #First off diagonal
#     H += (3./2.)*-np.eye(Nr,k=1)/(2.*dr**2)
#     H += (3./2.)*-np.eye(Nr,k=-1)/(2.*dr**2)

#     #Second off diagonal
#     H += (-3./20.)*-np.eye(Nr,k=2)/(2.*dr**2)
#     H += (-3./20.)*-np.eye(Nr,k=-2)/(2.*dr**2)

#     #Third off diagonal
#     H += (1./90.)*-np.eye(Nr,k=3)/(2.*dr**2)
#     H += (1./90.)*-np.eye(Nr,k=-3)/(2.*dr**2)
#     return H

# W,V = eigh_tridiagonal(SM_potential(r_array,R_array[int(Nr/2)]) + 1./dr**2, (-1./(2.*dr**2))*np.ones(Nr-1))
# print(SM_potential(r_array,R_array[0]))
print(Nr,NR)

GS_e = np.zeros([NR])
FE_e = np.zeros([NR])
SE_e = np.zeros([NR]) 

for i in range(0,NR):
    print(i)
    # diagonal = SM_potential(r_array,R_array[i]) + 1./dr**2
    # offdiagonal =(-1./(2.*dr**2))*np.ones(Nr-1)
    # W,V = eigh_tridiagonal(diagonal, offdiagonal)
    W,V = eigh(hamiltonian(dr,Nr,r_array,R_array[i]))


    GS_e[i] = W[0]
    FE_e[i] = W[1]
    SE_e[i] = W[2]

plt.plot(R_array,GS_e)
plt.plot(R_array,FE_e)
plt.plot(R_array,SE_e)
plt.savefig("bopes.png")
plt.show() 