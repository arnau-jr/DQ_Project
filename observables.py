import numpy as np
from numpy import matlib


def get_adiabatic_pops(NR,Nr,dR,dr,N_states,eigenstates,psi):
    pops = np.zeros(N_states)


    for m in range(0,N_states):
        flat_eigenstate = eigenstates[:,:,m].flatten("F")
        flat_chi2_m = flat_eigenstate*psi
        chi2_m = np.sum(flat_chi2_m.reshape(Nr,NR),0)*dr
        pops[m] = np.sum(np.abs(chi2_m)**2)*dR

    return pops

def get_decoherence_dynamics(NR,Nr,dR,dr,N_states,eigenstates,psi):
    Dnm = np.zeros((N_states,N_states))


    for m in range(0,N_states):
        for n in range(m,N_states):
            flat_eigenstate = eigenstates[:,:,m].flatten("F")
            flat_chi2_m = flat_eigenstate*psi

            flat_eigenstate = eigenstates[:,:,n].flatten("F")
            flat_chi2_n = flat_eigenstate*psi

            chi2_m = np.sum(flat_chi2_m.reshape(Nr,NR),0)*dr
            chi2_n = np.sum(flat_chi2_n.reshape(Nr,NR),0)*dr

            Dnm[n,m] = np.sum(np.abs(chi2_m)**2*np.abs(chi2_n)**2)*dR
            Dnm[m,n] = Dnm[n,m]


    return Dnm


def get_reduced_nuclear_density(NR,Nr,dr,psi):
    rho = np.abs(psi)**2
    den = np.zeros([NR])
    rhobis = np.reshape(rho,(Nr,NR))

    den = np.sum(rhobis,0)*dr
    return den

def get_reduced_electron_density(NR,Nr,dR,psi):
    rho = np.abs(psi)**2
    den = np.zeros([Nr])
    rhobis = np.reshape(rho,(Nr,NR))
    
    den = np.sum(rhobis,1)*dR
    return den