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
    plt.xlabel(r"$R (\rm a_0)$")
    plt.ylabel(r"$\rho_N (R)$")
    plt.xlim([-10,10])
    plt.plot(R_array,obv.get_reduced_nuclear_density(NR,Nr,dr,wave),label="Reduced nuclear density")
    # plt.plot(R_array,np.abs(wave_packet(R_array))**2,label="Wave packet")
    plt.axvline(-9.5,linestyle="--",color="r",linewidth=0.5)
    plt.axvline( 9.5,linestyle="--",color="r",linewidth=0.5)
    plt.legend()
    plt.savefig("pics/initial_nuclear_density.png")
    plt.close()

    plt.figure()
    plt.xlabel(r"$r (\rm a_0)$")
    plt.ylabel(r"$\rho_e (r)$")
    plt.xlim([-25,25])
    plt.plot(r_array,obv.get_reduced_electron_density(NR,Nr,dR,wave),label="Reduced electronic density")
    plt.axvline(-9.5,linestyle="--",color="r",linewidth=0.5)
    plt.axvline( 9.5,linestyle="--",color="r",linewidth=0.5)
    plt.legend()
    plt.savefig("pics/initial_electron_density.png")
    plt.close()

    full_hamiltonian_mat = fh.build_hamiltonian()
    elec_evolved, nucleus_evolved, pops_evolved,deco_evolved \
        = simulate(psi=wave,hamiltonian=full_hamiltonian_mat,eigenstates=eigenstates,dt=dt,endtime=end,snaps=snaps)
    
    #Animation nucleus
    fig,ax = plt.subplots(1,1)

    ax.set_xlabel(r"$R (\rm a_0)$")
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

    ax.set_xlabel(r"$r (\rm a_0)$")
    ax.set_ylabel(r"$\rho_e(r)$")
    mod_line = ax.plot(r_array,elec_evolved[0,:])
    ax.legend([r"$\rho_e(r)$"])

    def animat(i):
        mod_line[0].set_ydata(elec_evolved[i-1,:])
        return mod_line

    ani = animation.FuncAnimation(fig,animat,frames=elec_evolved.shape[0],interval=10.)
    writervideo = animation.FFMpegWriter(fps=60) 
    ani.save("pics/ani_elec.mp4", writer=writervideo,progress_callback =lambda i, n: print(f"Saving frame {i} of {n}",end="\r"))

    #plt.show()
    plt.close()

   #Animation nuclear + pops
    fig,ax = plt.subplots(2,1)

    ax[0].set_xlabel(r"$R (\rm a_0)$")
    ax[0].set_ylabel(r"$\rho_N(R)$")
    mod_line = ax[0].plot(R_array,nucleus_evolved[0,:])
    ax[0].legend([r"$\rho_N(R)$"])


    ax[1].set_xlabel(r"$t (\rm fs)$")
    ax[1].set_ylabel(r"$P_m (t)$")
    pop1_line = ax[1].plot(t_snaps_array[0],pops_evolved[0,0])
    pop2_line = ax[1].plot(t_snaps_array[0],pops_evolved[0,1])
    ax[1].set_xlim([0,30])
    ax[1].legend([r"$P_1 (t)$",r"$P_2 (t)$"])
    plt.tight_layout()

    def animat(i):
        mod_line[0].set_ydata(nucleus_evolved[i-1,:])

        pop1_line[0].set_data(t_snaps_array[:i],pops_evolved[:i,0])
        pop2_line[0].set_data(t_snaps_array[:i],pops_evolved[:i,1])
        return mod_line,pop1_line,pop2_line

    ani = animation.FuncAnimation(fig,animat,frames=nucleus_evolved.shape[0],interval=10.)
    writervideo = animation.FFMpegWriter(fps=60) 
    ani.save("pics/ani_pops_nuc.mp4", writer=writervideo,progress_callback =lambda i, n: print(f"Saving frame {i} of {n}",end="\r"))

    #plt.show()
    plt.close()


    plt.figure()
    plt.xlabel(r"$t (\rm fs)$")
    plt.ylabel(r"$P_m (t)$")
    plt.xlim([0,30])
    plt.plot(t_snaps_array,pops_evolved[:,0],label="Ground state")
    plt.plot(t_snaps_array,pops_evolved[:,1],label="First excited state")
    plt.plot(t_snaps_array,pops_evolved[:,2],label="Second excited state")
    plt.plot(t_snaps_array,np.sum(pops_evolved,axis=-1),label="Total population")
    plt.legend()
    plt.savefig("pics/adiabatic_populations.png")
    plt.close()


    plt.figure()
    plt.xlabel(r"$t (\rm fs)$")
    plt.ylabel(r"$D_{12} (t)$")
    plt.xlim([0,30])
    plt.plot(t_snaps_array,deco_evolved[:,0,1])
    plt.savefig("pics/D12.png")
    plt.close()
