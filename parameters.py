import numpy as np

L = 19.0

R_f = 5.
R_l = 4.
R_r = 3.1

dr = 1
dR = 0.01

r_array = np.arange(-60.,60.,dr)
R_array = np.arange(-9.,9.,dR)

Nr = np.size(r_array)
NR = np.size(R_array)

N_states = 3

dt = 1e-4