import numpy as np
import random
# 4.4.6 Bi Conjuagate Residual 双共轭残差法
# 参考 《迭代方法和预处理技术》谷同祥 等 编著 科技出版社
# 个人水平有限，程序可能有错，仅供参考

n = 128
A = np.zeros((n,n))
xexact = np.zeros((n))
for i in range(n):
    for j in range(i+1,n):
        A[j,i] = A[i,j] = random.random()
    A[i,i] = 8 # 对角占优越明显，越可能收敛
    xexact[i] = random.random()
b = np.dot(A,xexact)

x = np.zeros((n)) # 待求解的值
r = b - np.dot(A,x) # 残差
tmax = 1000
r_star = r.copy() # 只要r_start 和 Ar 的内积不为零就行了
d = np.zeros((n))
d_star = np.zeros((n))

beta = 0
Ad = np.zeros((n))
for t in range(0,tmax):
    Ad = np.dot(A,r)
    if t > 0:
        Ad = Ad + beta * np.dot(A,d)
        d = r + beta * d
        d_star = r_star + beta * d_star
    else:
        d = r.copy()
        d_star = r_star.copy()
    
    rho = np.dot(np.transpose(r_star),np.dot(A,r))
    if (abs(rho) < 1e-10):
        break
    ATd_star = np.dot(np.transpose(A),d_star)
    alpha = rho / np.dot(np.transpose(Ad),ATd_star)
    x = x + alpha * d
    r = r - alpha * Ad
    r_star = r_star - alpha * ATd_star
    beta = np.dot(np.transpose(r_star),np.dot(A,r)) / rho
    
