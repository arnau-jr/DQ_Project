import numpy as np
import matplotlib.pyplot as plt
import full_hamiltonian as fh
import observables as obv
from parameters import *
from matplotlib import animation
from numpy import matlib

def wave_packet(x,x0=-4.,sigma=1/np.sqrt(2.85)):
    """
    Initializing the Gaussian wavepacket around x0 = -4 and with sigma = 1/sqrt(2.85).
    The wave is computed into imaginary space by np.cdouble, in order to avoid splitting
    the wave packet later into real and imaginary part.
        Args: 
            x: 1D np.array, discretized space vector

        Returns: 
            double complex 1D np.array, wave over 
    """
    return np.cdouble(np.exp(-(x-x0)**2/(2*sigma**2))/np.sqrt(np.sqrt(np.pi)*sigma))

def compute_f(hamiltonian, wave):
    """
        Matrix multiplication part of the Fourth order Runge Kutta Method.
        Args: 
            hamiltonian: 2D np.array, full hamiltonian computed for the system (electron-nuclei)
                         can be a normal matrix or a sparse one with the dimensions NR*Nr
            wave: 1D np.array, computed wavepacket over the space NR*Nr

        Returns: 
            complex matrix product of the hamiltonian with the wave
    """

    return -1j*hamiltonian.dot(wave)

def evolve_psi_RK4(dt,hamiltonian,wave):
    """
        Fourth order Runge-Kutta Method to propagate a wave in space and time.
        Args: 
            dt: int number, discretization step in time
            hamiltonian: 2D np.array, full hamiltonian computed for the system (electron-nuclei)
                         can be a normal matrix or a sparse one with the dimensions NR*Nr
            wave: 1D np.array, computed wavepacket over the space NR*Nr

        Returns:
            wave: 1D np.array, with the dimension NR*Nr that is propagated through time and space
    """
    k1 = compute_f(hamiltonian,wave)
    k2 = compute_f(hamiltonian,wave + (dt/2.)*k1)
    k3 = compute_f(hamiltonian,wave + (dt/2.)*k2)
    k4 = compute_f(hamiltonian,wave +  dt*k3    )
    return wave + (dt/6.)*(k1 + 2.*k2 + 2.*k3 + k4)

def simulate(psi,hamiltonian,eigenstates,dt,endtime,snaps):

    """
    Simulation of the wave packet is happening here in the code. The time array, the wave packet,
    and the np.arrays for the propagated nucleus and electron are intialised and propagated.
    Args:
        psi: 1D np.array, wave packet with shape NR*Nr
        hamiltonian: 2D np.array, full hamiltonian for the system (electron-nuclei-hamiltonian)
        eigenstates: (NR,Nr,N_states) np.array, used for computing adiabiatic populations 
        dt: int number, discretization time step
        endtime: int number, length of the simulation 
        snaps: int number, number of snapshots taken of the simulation

    Returns: 
        elec_evolved: 2D np.array, evolved electronic wave through space and time with dimensions [endtime, Nr]
        nucleus_evolved: 2D np.array, evolved nucleus wave through space and time with dimensions [endtime, NR]

    """

    time = np.linspace(0,endtime,endtime)
    time_len = time.shape[0]
    print("Simulation time is: ",endtime)
    psi_len = np.size(psi)
    psi_evolved = np.zeros((int(time_len/snaps), psi_len),dtype=np.complex64)
    nucleus_evolved = np.zeros((int(time_len/snaps),NR))
    elec_evolved = np.zeros((int(time_len/snaps),Nr))
    pops_evolved = np.zeros((int(time_len/snaps),N_states))
    deco_evolved = np.zeros((int(time_len/snaps),N_states,N_states))

    temp_psi = psi
    psi_evolved[0] = temp_psi
    nucleus_evolved[0,:] = obv.get_reduced_nuclear_density(NR,Nr,dr,temp_psi)
    elec_evolved[0,:] = obv.get_reduced_electron_density(NR,Nr,dR,temp_psi)
    pops_evolved[0,:] = obv.get_adiabatic_pops(NR,Nr,dR,dr,N_states,eigenstates,temp_psi)
    deco_evolved[0,:] = obv.get_decoherence_dynamics(NR,Nr,dR,dr,N_states,eigenstates,temp_psi)
    norm_nuc = np.sum(nucleus_evolved[0])*dR
    norm_elec = np.sum(elec_evolved[0])*dr
    norm_psi = np.sum(np.abs(temp_psi)**2)*dr*dR
    

    f_norm = open("norms.dat","w") #File for norms
    f_norm.write("t "+" Nuclear norm "+" Electron norm "+" Total norm \n")
    f_norm.write(str(0.0) + " " + str(norm_nuc) + " " + str(norm_elec) + " " + str(norm_psi) +"\n")

    for i in range(time_len):
        print(i," of ",time_len,end=' \r')
        temp_psi = evolve_psi_RK4(dt,hamiltonian,temp_psi)
        if i%snaps == 0:
            psi_evolved[int(i/snaps)] = temp_psi
            nucleus_evolved[int(i/snaps),:] = obv.get_reduced_nuclear_density(NR,Nr,dr,temp_psi)
            elec_evolved[int(i/snaps),:] = obv.get_reduced_electron_density(NR,Nr,dR,temp_psi)
            pops_evolved[int(i/snaps),:] = obv.get_adiabatic_pops(NR,Nr,dR,dr,N_states,eigenstates,temp_psi)
            deco_evolved[int(i/snaps),:] = obv.get_decoherence_dynamics(NR,Nr,dR,dr,N_states,eigenstates,temp_psi)
            
            norm_nuc = np.sum(nucleus_evolved[int(i/snaps)])*dR
            norm_elec = np.sum(elec_evolved[int(i/snaps)])*dr
            norm_psi = np.sum(np.abs(temp_psi)**2)*dr*dR

            f_norm.write(str(i*dt) + " " + str(norm_nuc) + " " + str(norm_elec) + " " + str(norm_psi) +"\n")
    f_norm.close()
    return elec_evolved,nucleus_evolved,pops_evolved,deco_evolved


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
