import numpy as np

def get_adiabatic_pops(NR,Nr,dR,N_states,eigenstates,psi):
    pops = np.zeros(N_states)
    chi2_m = np.zeros(NR)

    for m in range(0,N_states):
        for i in range(0,NR):
            chi2_m[i] = np.abs(np.conj(eigenstates[i::Nr,m])*psi[i::Nr])**2 #We have to rebuild eigenstates
        pops[m] = np.sum(chi2_m)*dR

    return pops


def get_reduced_nuclear_density(NR,Nr,dr,psi):
    rho = np.abs(psi)**2
    den = np.zeros([NR])
    #Should work, but loop is inefficient
    for i in range(NR):
        den[i] = np.sum(rho[i::Nr])*dr

    return den

def get_reduced_electron_density(NR,Nr,dR,psi):
    rho = np.abs(psi)**2
    den = np.zeros([Nr])
    #Should work, but loop is inefficient
    for i in range(Nr):
        den[i] = np.sum(rho[i*NR:(i+1)*NR])*dR

    return den