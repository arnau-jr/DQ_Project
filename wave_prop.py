import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation

## parameters and preliminaries
dR = 0.1
R_array = np.arange(-6.,6.,dR)
dt = 1e-4

def wave_packet(x,x0=-4.,sigma=1/np.sqrt(2.85)):
    """
    Initializing the Gaussian wavepacket around x0 = -4 and with sigma = 1/sqrt(2.85) at the ground state
    of the system. Instead of splitting the function in real and imaginary part, the wavepacket 
    is initialized with both parts by multiplying with np.exp(1j+1).
    """
    return np.exp(1j+1)*np.exp(-(x-x0)**2/(2*sigma**2))/np.sqrt(np.sqrt(np.pi)*sigma)