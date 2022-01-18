import numpy as np

x = np.array([4,6],dtype = float)
m0 = 0.1
m1 = 0.1

W = np.array([[1.0/m0,0],[0,1.0/m1]])

time = 0
timeFinal = 100
x_record = np.zeros((timeFinal+1,2))
J = np.zeros((2))
v = np.zeros((2))
dt = 0.01
beta = 20
x_record[0] = x
while time < timeFinal:
    x_record[time+1] = x
    J = np.array([1,-1],dtype = float)
    # 1 维情况，K 是一维矩阵，也就是只有一个数字
    K = J @ W @ J.T
    
    C = abs(x[1] - x[0])
    dCdt = J @ v
    b = beta / dt * C
    lam = (dCdt + b) / K
    F_c = lam * J
    v = dt * F_c
    x = x + dt * v 
    time += 1
    
