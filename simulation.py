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
        eigenvalues, eigenstates = compute_eigen(Nr,NR,r_array,R_array)

        with open("eigenvalues.npy","wb") as f:
            np.save(f,eigenvalues)

        with open("eigenstates.npy","wb") as g:
            np.save(g,eigenstates)

    
    else:
        ## loading the saved arrays from the bopes step
        eigenstates = np.load("eigenstates.npy")


    ex1 = eigenstates[:,:,1]

    #### THIS IS THE PART OF THE NEWLY CONSTRUCTED WAVE
    wave_p = wave_packet(x=R_array)
    wave = ex1.flatten(order="F")*matlib.repmat(wave_p,1,Nr).flatten()

    # wave = wave/(np.sqrt(np.sum(np.abs(wave)**2)*dr*dR))
    print("Norm of initial wave:",np.sum(np.abs(wave)**2)*dr*dR)
    print("Adiabatic pops of initial wave:",obv.get_adiabatic_pops(NR,Nr,dR,dr,N_states,eigenstates,wave))
    print("Deco. dynamics pops of initial wave:\n",obv.get_decoherence_dynamics(NR,Nr,dR,dr,N_states,eigenstates,wave))

    full_hamiltonian_mat = fh.build_hamiltonian()
    psi_evolved, nucleus_evolved = simulate(psi=wave,hamiltonian=full_hamiltonian_mat,dt=dt,endtime=0.1,snaps=1)

    print(psi_evolved.shape)
    
    #Animation
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
    ani.save("ani_nucleus.mp4", writer=writervideo,progress_callback =lambda i, n: print(f"Saving frame {i} of {n}",end="\r"))

    plt.show()
    plt.close()
    
