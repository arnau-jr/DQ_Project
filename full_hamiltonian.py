import numpy as np
import matplotlib.pyplot as plt
from parameters import *
from scipy import sparse as sp
from scipy.special import erf
from bopes import SM_potential

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
    L += (-205./72.)*np.eye(N)/(dr**2)

    #First off diagonal
    L += (8./5.)*np.eye(N,k=1)/(dr**2)
    L += (8./5.)*np.eye(N,k=-1)/(dr**2)

    #Second off diagonal
    L += (-1./5.)*np.eye(N,k=2)/(dr**2)
    L += (-1./5.)*np.eye(N,k=-2)/(dr**2)

    #Third off diagonal
    L += (8./315.)*np.eye(N,k=3)/(dr**2)
    L += (8./315.)*np.eye(N,k=-3)/(dr**2)

    #Fourth off diagonal
    L += (-1./560.)*np.eye(N,k=4)/(dr**2)
    L += (-1./560.)*np.eye(N,k=-4)/(dr**2)
        
    return L


def build_hamiltonian():

    ## with mesgrid two arrays can be parsed through the potential function
    ## without the same size
    r_arr, R_arr = np.meshgrid(r_array, R_array)
    pot = SM_potential(r=r_arr, R=R_arr)
    #print(pot.shape)

    ## reshape the potential vector and transfrom into N*N matrix
    N = Nr*NR
    pot_new = pot.reshape(N)

    ## compute laplacian for the kinetic parts and put the into the same space
    ## with the kronecker product
    m = 1
    M = 1836.152673

    Te = -laplacian(dr,Nr)/(2*m)
    Tp = -laplacian(dR,NR)/(2*M)

    Te_full = sp.kron(Te,np.eye(NR))
    Tp_full = sp.kron(np.eye(Nr),Tp)

    ## put the terms together and you get the full hamiltonian for the system
    full_hamiltonian = sp.diags(pot_new) + Te_full + Tp_full

    return full_hamiltonian

