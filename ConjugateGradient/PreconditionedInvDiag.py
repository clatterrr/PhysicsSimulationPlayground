import numpy as np
# 4.2.5 Preconditioned Conjuagate Gradient
# 参考 《迭代方法和预处理技术》谷同祥 等 编著 科技出版社
# 个人水平有限，程序可能有错，仅供参考
A = np.array([[3,0,1],[0,4,2],[1,2,3]],dtype = float)
b = np.array([1,0,0],dtype = float)
n = 3
Minv = np.zeros((n,n))
for i in range(n):
    Minv[i,i] = 1 / A[i,i]

x = np.zeros((n)) # 待求解的值
p = np.zeros((n)) # 方向
r = b - np.dot(A,x) # 残差
z = np.dot(Minv,r)
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
    z = np.dot(Minv,r)
    beta = np.dot(np.transpose(r),z) / rz_old
    d = z + beta * d
    
    