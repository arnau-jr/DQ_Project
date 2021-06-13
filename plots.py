import numpy as np
import matplotlib.pyplot as plt
from parameters import *
import matplotlib as mpl


eigenstates = np.load("eigenstates.npy")

# Meshgrid conviently produces 2D-Grids
XX, YY = np.meshgrid(r_array, R_array)

eig0 = eigenstates[:,:,2]

# Create a new figure
figsize_x, figsize_y = mpl.rcParams["figure.figsize"]
fig = plt.figure(figsize=(figsize_x * 2, figsize_y))

# Add a subplot capable of plotting in 3 dimensions
ax_3d = fig.add_subplot(121, projection='3d')

cont_3d = ax_3d.plot_surface(
    XX, YY, eig0,                # x, y, and z values as 2D-arrays
   cstride=1,                    # Downsampling in one direction
   rstride=1,                    # Downsampling in other direction
   linewidth=10,                 
   facecolors=mpl.cm.jet(eig0),   # Sets the facecolors to RGB values according to functional values
                                # Calling cm.jet(VV) transforms the numbers into RGB-Values
   vmin=0,                      # Bottom of colorbar
   vmax=1                       # Top of colorbar
)                      

# Add a second, regular subplot for the 2D contour plot
ax_2d = fig.add_subplot(122)
cont_2d = ax_2d.contourf(
    eig0, 
    50,              # Sampling rate in each direction
    vmin=0,          # Bottom of colorbar, same as above
    vmax=1,          # Top of colorbar, same as above
    cmap=mpl.cm.jet  # Colormap
    )

plt.colorbar(cont_2d)  # Plots the colorbar for 2d contour diagram

# Loop over both axes to label them and adjust the tick sizes
for a in fig.axes[:-1]:
    a.set_ylabel(r'R($a_0$)', size=10)
    a.set_xlabel(r'r($a_0$)', size=10)
    a.tick_params(labelsize=5)

ax_3d.set_zlabel('energy [a.u.]', size = 10)

#fig.suptitle("Ground state")
#fig.suptitle("First excited state")
fig.suptitle("Second excited state")



#plt.savefig("pics/groundstate.png")
#plt.savefig("pics/first_excited.png")
plt.savefig("pics/second_excited.png")
