# 4.2.5 Preconditioned Conjuagate Gradient
# 参考 《迭代方法和预处理技术》谷同祥 等 编著 科技出版社
# 个人水平有限，程序可能有错，仅供参考
import numpy as np
import random
# 初始化矩阵
n = 64
A = np.zeros((n,n))
b = np.zeros((n))
for i in range(n):
    for j in range(i+1,n):
        ran = random.random()
         # 稀疏矩阵，矩阵越稀疏，越适合用不完全Cholesky分解
        if ran < 0.9:
            continue
        A[i,j] = random.random()
        A[j,i] = A[i,j]
    A[i,i] = 4 # 不知道为什么，对角占优越多，求解越快
    b[i] = random.random()

# 不完全Cholesky 分解
L = A.copy()
for k in range(n):
    if L[k,k] < 0:
        L[k,k] = np.sqrt(A[k,k])
    else:
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

# 函数，预处理子
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
            result[i] -= L[j,i] / L[i,i] * result[j]
            
    return result

# 预处理共轭梯度
x = np.zeros((n)) # 待求解的值
p = np.zeros((n)) # 方向
r = b - np.dot(A,x) # 残差
z = preconditioner()
d = z.copy() # 方向
tmax = 1000
for t in range(0,tmax):
    z = np.dot(L,d)
    if max(abs(r.max()),abs(r.min())) < 1e-10:
        break
    rz_old = np.dot(np.transpose(r),z)
    alpha = rz_old / np.dot(np.transpose(d),z)
    x = x + alpha * d
    r = r - alpha * z
    z = preconditioner()
    beta = np.dot(np.transpose(r),z) / rz_old
    z_old = z.copy()
    z = z + beta * d
    d = z_old.copy()
# 精确解
xexact = np.dot(np.linalg.inv(A),b)
    