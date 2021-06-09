import numpy as np
import matplotlib.pyplot as plt
import full_hamiltonian as fh
import observables as obv
from parameters import *
from matplotlib import animation
from numpy import matlib

def wave_packet(x,x0=-4.,sigma=1/np.sqrt(2.85)):
    """
    Initializing the Gaussian wavepacket around x0 = -4 and with sigma = 1/sqrt(2.85) at the ground state
    of the system. Instead of splitting the function in real and imaginary part, the wavepacket 
    is initialized with both parts by multiplying with np.exp(1j+1).
    """
    return np.cdouble(np.exp(-(x-x0)**2/(2*sigma**2))/np.sqrt(np.sqrt(np.pi)*sigma))

def compute_f(hamiltonian, wave):
    return -1j*hamiltonian.dot(wave)

def evolve_psi_RK4(dt,hamiltonian,wave):

    k1 = compute_f(hamiltonian,wave)
    k2 = compute_f(hamiltonian,wave + (dt/2.)*k1)
    k3 = compute_f(hamiltonian,wave + (dt/2.)*k2)
    k4 = compute_f(hamiltonian,wave +  dt*k3    )
    return wave + (dt/6.)*(k1 + 2.*k2 + 2.*k3 + k4)

def simulate(psi,hamiltonian,dt,endtime,snaps):
    time = np.arange(0,endtime,dt)
    time_len = np.size(time)
    print("Simulation time is: ",time_len)
    psi_len = np.size(psi)
    psi_evolved = np.zeros((int(time_len/snaps), psi_len),dtype=np.complex64)
    nucleus_evolved = np.zeros((int(time_len/snaps),NR))
    elec_evolved = np.zeros((int(time_len/snaps),Nr))

    temp_psi = psi
    for i in range(time_len-1):
        print(i," of ",time_len,end='\r')
        temp_psi = evolve_psi_RK4(dt,hamiltonian,temp_psi)
        if i%snaps == 0:
            psi_evolved[int(i/snaps)] = temp_psi
            nucleus_evolved[int(i/snaps),:] = obv.get_reduced_nuclear_density(NR,Nr,dr,temp_psi)
            elec_evolved[int(i/snaps),:] = obv.get_reduced_electron_density(NR,Nr,dR,temp_psi)
            norm_nuc = np.sum(np.abs(nucleus_evolved[int(i/snaps)])**2)*dR
            norm_elec = np.sum(np.abs(elec_evolved[int(i/snaps)])**2)*dr
            print("The norm of the nuc_wave is: ", norm_nuc)
            print("The norm of the eelc_wave is: ", norm_elec)

    return elec_evolved,nucleus_evolved


# with open("psi_evolved.npy","wb") as f:
#     np.save(f,psi_evolved)


## A: Some plots for testing different stuff
# plt.figure()
# plt.plot(np.abs(psi0)**2)
# plt.plot(ex1[np.where(R_array==-4.)[0][0],:])
# plt.plot(R_array,obv.get_reduced_nuclear_density(NR,Nr,dr,wave),label="Reduced")
# plt.plot(R_array,np.abs(wave_packet(R_array))**2,label="Wave packet")
# plt.plot(r_array,obv.get_reduced_electron_density(NR,Nr,dR,psi0))
# plt.plot(r_array,np.abs(ex1[np.where(R_array==-4.)[0][0],:])**2)
# plt.plot(R_array,nucleus_evolved[0,:],label="tinitial")
# plt.plot(R_array,nucleus_evolved[50,:],label="tfinal")
# plt.legend()
# plt.show()
# plt.close()
