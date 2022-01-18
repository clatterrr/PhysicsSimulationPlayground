import numpy as np
# 完成
x0 = np.array([0,4],dtype = float)
xcm0 = (x0[0] + x0[1]) * 0.5
q = x0 - xcm0
f = np.array([0,0],dtype = float)
x = x0.copy()
x[0] = 3
x[1] = 5
mass = 1
dt = 0.5
time = 0
timeFinal = 100
x_record  = np.zeros((2,timeFinal))
while time < timeFinal:
    x_record[:,time] = x
    v = f / mass * dt
    x = x + dt * v
    
    xcm = (x[0] + x[1]) * 0.5
    p = x - xcm
    Apq = 0
    Aqq = 0
    for i in range(2):
        Apq += mass * p[i] * q[i]
        Aqq += mass * q[i] * q[i]
    S = np.sqrt(Apq * Apq)
    R = 1
    if abs(S) > 1e-10:
        R = Apq / S
    g = xcm + R * q
    for i in range(2):
        if g[i] > 4:
            f[i] = -2
            g[i] = 4
        else:
            f[i] = 0
    v = (g - x) / dt
    x = g.copy()
    time += 1