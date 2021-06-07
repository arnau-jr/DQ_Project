import numpy as np

L = 19.0

R_f = 5.
R_l = 4.
R_r = 3.1

dr = 0.25
dR = 0.25

r_array = np.arange(-120.,120.,dr)
R_array = np.arange(-9.,9.,dR)

Nr = np.size(r_array)
NR = np.size(R_array)

dt = 1e-1