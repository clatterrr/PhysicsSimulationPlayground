import numpy as np
A = np.array([[3,0,1],[0,4,2],[1,2,3]],dtype = float)
b = np.array([1,0,0],dtype = float)
nmax = 3

x = np.zeros((nmax)) # 待求解的值
p = np.zeros((nmax)) # 方向
res = b - np.dot(A,x) # 残差
bnrm2 = np.linalg.norm(b)
rho1 = 1
tmax = 1000
xt = np.zeros((tmax,nmax))
for t in range(0,tmax):
    rho = np.dot(np.transpose(res),res)
    beta = 0
    if t > 0:
        beta = rho / rho1
    p = res + beta*p
    Ap = np.dot(A,p)
    alpha = rho / np.dot(np.transpose(p),Ap)
    xt[t,:] = x
    x = x + alpha * p
    res = res - alpha * Ap
    error = np.linalg.norm(res) / bnrm2
    if error < 1e-8:
        break
    rho1 = rho