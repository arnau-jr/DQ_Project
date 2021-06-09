import numpy as np

L = 19.0

R_f = 5.
R_l = 4.
R_r = 3.1

dr = 0.6
dR = 0.06

r_array = np.arange(-75.,75.,dr)
R_array = np.arange(-9.,9.,dR)

Nr = np.size(r_array)
NR = np.size(R_array)

N_states = 3

dt = 1e-1
tmax = 30
auTofs = 2.4e-2
#end = int(tmax/(dt*auTofs))
end = 12500 ## so its a round number because it would be around 12499
