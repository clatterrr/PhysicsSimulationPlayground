import numpy as np
# 4.3.12 General Conjuagate Residual 广义共轭残差法
# 参考 《迭代方法和预处理技术》谷同祥 等 编著 科技出版社
# 个人水平有限，程序可能有错，仅供参考
A = np.array([[3,0,1],[0,4,2],[1,2,3]],dtype = float)
b = np.array([1,0,0],dtype = float)
nmax = 3

x = np.zeros((nmax)) # 待求解的值
r = b - np.dot(A,x) # 残差
d = np.dot(np.transpose(A),r)
tmax = 100
Adtime = np.zeros((tmax,nmax))
dtime = np.zeros((tmax,nmax))
for t in range(0,tmax):
    rho = np.dot(np.transpose(r),r)
    if abs(rho) < 1e-10:
        break
    Ad = np.dot(A,d)
    Adtime[t,:] = Ad
    dtime[t,:] = d
    alpha = np.dot(np.transpose(r),Ad) / np.dot(np.transpose(Ad), Ad)
    x = x + alpha * d
    r = r - alpha * Ad
    Ar = np.dot(A,r)
    beta = np.zeros((tmax))
    term = 0
    for i in range(t):
        factor = np.dot(np.transpose(Ar), Adtime[i,:])
        factor1 = np.dot(np.transpose(Adtime[i,:]),Adtime[i,:])
        beta[i] = - factor / factor1
        term = beta[i] * dtime[i,:]
    d = r + term
        