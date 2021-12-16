import numpy as np

# 初始位置
X0 = 1
X1 = 3
Dm = X1 - X0
W = X1 - X0
Dminv = 1 / Dm

# 顶点速度
v0 = 0
v1 = 0
# 顶点现在位置
x0 = 1
x1 = 5
# 拉梅常数
mu = 2
la = 2
# 时间步长和质量
dt = 1
mass = 1

time = 0
timeFinal = 100
# 记录位置变化
xt = np.zeros((2,timeFinal))
while(time < timeFinal):
    
    Kmat = np.zeros((2,2))
    
    Ds = x1 - x0
    F = Ds * Dminv
    E = (F * F - 1)/2
    trE = E
    P = F*(2*mu*E + la*trE)
    H = - W * P * Dminv
    f1 = H
    f0 = - f1
    
    dD0 = -1
    dD1 = 1
    
    dF0 = dD0 * Dminv
    dF1 = dD1 * Dminv
    
    dE0 = (dF0 * F + F * dF0) / 2
    dE1 = (dF1 * F + F * dF1) / 2
    
    dP0 = dF0 * (2*mu*E + la*trE) + F*(2*mu*dE0 + la*dE0)
    dP1 = dF1 * (2*mu*E + la*trE) + F*(2*mu*dE1 + la*dE1)
    
    dH0 = - W * dP0 * Dminv
    Kmat[1,0] = dH0
    Kmat[0,0] =  - dH0
    dH1 = - W * dP1 * Dminv
    Kmat[0,1] = - dH1
    Kmat[1,1] = dH1
    
    A = mass * np.identity(2) - Kmat * dt * dt
    b = np.array([v0,v1]) * mass + dt * np.array([f0,f1])
    x = np.dot(np.linalg.inv(A),b)
    
    v0 = x[0]
    v1 = x[1]
    x0 = x0 + v0 * dt
    x1 = x1 + v1 * dt
    
    xt[0,time] = x0
    xt[1,time] = x1
    time += 1
        
    
