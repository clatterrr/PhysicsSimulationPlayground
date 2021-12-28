import numpy as np
# 4.2.5 Preconditioned Conjuagate Gradient
# 参考 《迭代方法和预处理技术》谷同祥 等 编著 科技出版社
# 个人水平有限，程序可能有错，仅供参考
A = np.array([[3,0,1],[0,4,2],[1,2,3]],dtype = float)
b = np.array([1,0,0],dtype = float)
n = 3
L = A.copy()
for k in range(n):
    L[k,k] = np.sqrt(L[k,k])
    for i in range(k+1,n):
        if L[i,k] != 0:
            L[i,k] = L[i,k] / L[k,k]
    for j in range(k+1,n):
        for i in range(j,n):
            if L[i,j] != 0:
                L[i,j] = L[i,j] - L[i,k] * L[j,k]
for i in range(n):
    for j in range(i+1,n):
        L[i,j] = 0

def preconditioner():
    global L
    global r
    # 求解 L * result = rhs，即 result = L^-1 rhs
    resultTemp = np.zeros((n))
    for i in range(n):
        resultTemp[i] = r[i] / L[i,i]
        for j in range(i):
            resultTemp[i] -= L[i,j] / L[i,i] * resultTemp[j]
    # 求解 L^T * result = rhs 即 result = L^T^-1 rhs
    result = np.zeros((n))
    for i in range(n-1,-1,-1):
        result[i] = resultTemp[i] / L[i,i]
        for j in range(i+1,n):
            result[i] -= L[j,i] / L[i,i] * resultTemp[j]
            
    return result

x = np.zeros((n)) # 待求解的值
p = np.zeros((n)) # 方向
r = b - np.dot(A,x) # 残差
z = preconditioner()
d = z.copy() # 方向
tmax = 1000
for t in range(0,tmax):
    Ad = np.dot(A,d)
    rz_old = np.dot(np.transpose(r),z)
    if abs(rz_old) < 1e-10:
        break
    alpha = rz_old / np.dot(np.transpose(d),Ad)
    x = x + alpha * d
    r = r - alpha * Ad
    z = preconditioner()
    beta = np.dot(np.transpose(r),z) / rz_old
    d = z + beta * d
    
    