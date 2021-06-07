import numpy as np
import matplotlib.pyplot as plt
import full_hamiltonian as fh
import observables as obv
from matplotlib import animation
from numpy import matlib

def wave_packet(x,x0=-4.,sigma=1/np.sqrt(2.85)):
    """
    Initializing the Gaussian wavepacket around x0 = -4 and with sigma = 1/sqrt(2.85) at the ground state
    of the system. Instead of splitting the function in real and imaginary part, the wavepacket 
    is initialized with both parts by multiplying with np.exp(1j+1).
    """
    return np.cdouble(np.exp(-(x-x0)**2/(2*sigma**2))/np.sqrt(np.sqrt(np.pi)*sigma))

## parameters and preliminaries
dr = 0.25
dR = 0.25
r_array = np.arange(-19,19,dr)
R_array = np.arange(-9,9,dR)
dt = 1e-4

## loading the saved arrays from the bopes step
non_adiabatic = np.load("non_adiabatic_coupling.npy")
eigenstates = np.load("eigenvstates.npy")
eigenvalues = np.load("eigenvalues.npy")

NR,Nr,N_states = eigenstates.shape
ex1 = eigenstates[:,:,1]
psi0 = np.zeros([Nr*NR],dtype=np.complex64)


psi0 = np.outer(ex1[np.where(R_array==-4.)[0][0],:],wave_packet(R_array)).flatten()

#### THIS IS THE PART OF THE NEWLY CONSTRUCTED WAVE
wave_p = wave_packet(x=R_array)
wave_p_vec = matlib.repmat(wave_p,1,Nr).T
phi0 = ex1.flatten()*wave_p_vec
wave = phi0.sum(axis=0)
#### IT IS NORMED FOR NOW SINCE THE EIGENSTATES ARE NOT
wave = wave/(np.sqrt(np.sum(np.abs(wave)**2)*dr*dR))


dt = 1e-4
#endtime = 30/(2.418884e-2)
endtime = 0.01

def compute_f(hamiltonian, wave):
    return hamiltonian@wave

def evolve_psi_RK4(dt,hamiltonian,wave):

    k1 = compute_f(hamiltonian,wave)
    k2 = compute_f(hamiltonian,wave + (dt/2.)*k1)
    k3 = compute_f(hamiltonian,wave + (dt/2.)*k2)
    k4 = compute_f(hamiltonian,wave +  dt*k3    )
    return wave + (dt/6.)*(k1 + 2.*k2 + 2.*k3 + k4)

def simulate(psi,hamiltonian,dt,endtime,snaps):
    time = np.arange(0,endtime,dt)
    time_len = np.size(time)
    print(time_len)
    psi_len = np.size(psi)
    psi_evolved = np.zeros((int(time_len/snaps), psi_len),dtype=np.complex64)
    nucleus_evolved = np.zeros((int(time_len/snaps),NR))
    print(psi_evolved.shape)
    temp_psi = psi
    for i in range(time_len-1):
        print(i," of ",time_len,end='\r')
        temp_psi = evolve_psi_RK4(dt,hamiltonian,temp_psi)
        if i%snaps == 0:
            psi_evolved[int(i/snaps)] = temp_psi
            nucleus_evolved[int(i/snaps),:] = obv.get_reduced_nuclear_density(NR,Nr,dr,temp_psi)
    
    return psi_evolved,nucleus_evolved

full_hamiltonian_mat = fh.build_hamiltonian()
psi_evolved,nucleus_evolved = simulate(psi=psi0,hamiltonian=full_hamiltonian_mat,dt=1e-7,endtime=1e-5,snaps=10)

with open("psi_evolved.npy","wb") as f:
    np.save(f,psi_evolved)


## A: Some plots for testing different stuff
# plt.figure()
# plt.plot(np.abs(psi0)**2)
# plt.plot(ex1[np.where(R_array==-4.)[0][0],:])
# plt.plot(R_array,obv.get_reduced_nuclear_density(NR,Nr,dr,psi0))
# plt.plot(R_array,np.abs(wave_packet(R_array))**2)
# plt.plot(r_array,obv.get_reduced_electron_density(NR,Nr,dR,psi0))
# plt.plot(r_array,np.abs(ex1[np.where(R_array==-4.)[0][0],:])**2)
# plt.plot(R_array,nucleus_evolved[0,:])
# plt.plot(R_array,nucleus_evolved[1,:])
# plt.show()
# plt.close()

