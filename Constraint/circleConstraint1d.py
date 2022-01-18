import numpy as np
# circle constraint 1d
# C = |x| - r

r = 5
x_new = - 6
v = 0
C = abs(x_new)
J_new = x_new / abs(x_new)
x_old = - 5
J_old = x_new / abs(x_new)
time = 0
timeFinal = 100
mass = 1.0
W = 1.0 / mass
f = 0
x_record = np.zeros((timeFinal))
while time < timeFinal:
    
    J_new = x_new / abs(x_new)
    dJdt = J_new - J_old
    J_old = J_new
    dxdt = x_new - x_old
    
    k = J_new * W * J_new
    b = - J_new * W * f - dJdt * dxdt
    lam = b / k
    f_c = J_new * lam
    v = v + (f + f_c) / mass
    x_new = x_new + v 
    x_record[time] = x_new
    time += 1