import numpy as np
# 4.2.8 Conjuagate Gradient Normal Error 共轭梯度残差法
# 参考 《迭代方法和预处理技术》谷同祥 等 编著 科技出版社
# 个人水平有限，程序可能有错，仅供参考
A = np.array([[3,0,1],[0,4,2],[1,2,3]],dtype = float)
b = np.array([1,0,0],dtype = float)
nmax = 3

x = np.zeros((nmax)) # 待求解的值
r = b - np.dot(A,x) # 残差
d = np.dot(np.transpose(A),r)
tmax = 100
for t in range(0,tmax):
    rho = np.dot(np.transpose(r),r)
    if abs(rho) < 1e-10:
        break
    alpha = rho / np.dot(np.transpose(d),d) 
    Ad = np.dot(A, d)
    x = x + alpha * d
    r = r - alpha * Ad
    beta = np.dot(np.transpose(r),r) / rho
    d = np.dot(np.transpose(A),r) + beta * d