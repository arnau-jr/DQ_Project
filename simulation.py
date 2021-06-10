import numpy as np
import matplotlib.pyplot as plt
import full_hamiltonian as fh
import observables as obv
from matplotlib import animation
from numpy import matlib
from wave_prop import simulate, wave_packet
from bopes import compute_eigen
from parameters import *

if __name__== '__main__':

    print("Should the eigenstates and eigenvalues be recalculated? (y/n) \n")
    answer = input()
    if answer.lower()=="y":
        eigenvalues, eigenstates = compute_eigen(Nr,NR,r_array,R_array,N_states)

        with open("eigenvalues.npy","wb") as f:
            np.save(f,eigenvalues)

        with open("eigenstates.npy","wb") as g:
            np.save(g,eigenstates)

    
    else:
        ## loading the saved arrays from the bopes step
        eigenstates = np.load("eigenstates.npy")

    #plt.figure()
    #plt.imshow(eigenstates[:,:,1])
    #plt.show()

    ex1 = eigenstates[:,:,1]

    #### THIS IS THE PART OF THE NEWLY CONSTRUCTED WAVE
    wave_p = wave_packet(x=R_array)
    wave = ex1.flatten(order="F")*matlib.repmat(wave_p,1,Nr).flatten()

    # wave = wave/(np.sqrt(np.sum(np.abs(wave)**2)*dr*dR))
    print("Norm of initial wave:",np.sum(np.abs(wave)**2)*dr*dR)
    print("Norm of initial nuclear density:",np.sum(obv.get_reduced_nuclear_density(NR,Nr,dr,wave))*dR)
    print("Norm of initial electronic density:",np.sum(obv.get_reduced_electron_density(NR,Nr,dR,wave))*dr)
    print("Adiabatic pops of initial wave:",obv.get_adiabatic_pops(NR,Nr,dR,dr,N_states,eigenstates,wave))
    print("Deco. dynamics pops of initial wave:\n",obv.get_decoherence_dynamics(NR,Nr,dR,dr,N_states,eigenstates,wave))

    plt.figure()
    plt.xlabel(r"$R(a_0)$")
    plt.ylabel(r"$\rho_N (R)$")
    plt.xlim([-9,9])
    plt.plot(R_array,obv.get_reduced_nuclear_density(NR,Nr,dr,wave),label="Reduced")
    plt.plot(R_array,np.abs(wave_packet(R_array))**2,label="Wave packet")
    plt.legend()
    plt.savefig("pics/initial_nuclear_density.png")
    plt.close()

    plt.figure()
    plt.xlabel(r"$r(a_0)$")
    plt.ylabel(r"$\rho_e (r)$")
    plt.xlim([-9,9])
    plt.plot(r_array,obv.get_reduced_electron_density(NR,Nr,dR,wave),label="Reduced")
    # plt.plot(r_array,np.abs(ex1)**2,label="First excited eigenstate")
    plt.legend()
    plt.savefig("pics/initial_electron_density.png")
    plt.close()

    full_hamiltonian_mat = fh.build_hamiltonian()
    elec_evolved, nucleus_evolved, pops_evolved = simulate(psi=wave,hamiltonian=full_hamiltonian_mat,eigenstates=eigenstates,dt=dt,endtime=end,snaps=10)
    
    #Animation nucleus
    fig,ax = plt.subplots(1,1)

    ax.set_xlabel("R")
    ax.set_ylabel(r"$\rho_N(R)$")
    mod_line = ax.plot(R_array,nucleus_evolved[0,:])
    ax.legend([r"$\rho_N(R)$"])

    def animat(i):
        mod_line[0].set_ydata(nucleus_evolved[i-1,:])
        return mod_line

    ani = animation.FuncAnimation(fig,animat,frames=nucleus_evolved.shape[0],interval=10.)
    writervideo = animation.FFMpegWriter(fps=60) 
    ani.save("pics/ani_nucleus.mp4", writer=writervideo,progress_callback =lambda i, n: print(f"Saving frame {i} of {n}",end="\r"))

    #plt.show()
    plt.close()

    #Animation electron
    fig,ax = plt.subplots(1,1)

    ax.set_xlabel("r")
    ax.set_ylabel(r"$\rho_N(r)$")
    mod_line = ax.plot(r_array,elec_evolved[0,:])
    ax.legend([r"$\rho_N(r)$"])

    def animat(i):
        mod_line[0].set_ydata(elec_evolved[i-1,:])
        return mod_line

    ani = animation.FuncAnimation(fig,animat,frames=elec_evolved.shape[0],interval=10.)
    writervideo = animation.FFMpegWriter(fps=60) 
    ani.save("pics/ani_elec.mp4", writer=writervideo,progress_callback =lambda i, n: print(f"Saving frame {i} of {n}",end="\r"))

    #plt.show()
    plt.close()



    plt.figure()
    plt.xlabel(r"$t(fs)$")
    plt.ylabel(r"$P_m$")
    plt.xlim([0,30])
    plt.plot(np.arange(0,tmax,dt*auTofs*10),pops_evolved[:,0],label="Ground state")
    plt.plot(np.arange(0,tmax,dt*auTofs*10),pops_evolved[:,1],label="First excited state")
    plt.plot(np.arange(0,tmax,dt*auTofs*10),pops_evolved[:,2],label="Second excited state")
    plt.plot(np.arange(0,tmax,dt*auTofs*10),np.sum(pops_evolved,axis=-1),label="Total population")
    plt.legend()
    plt.savefig("pics/adiabatic_populations.png")
    plt.close()
