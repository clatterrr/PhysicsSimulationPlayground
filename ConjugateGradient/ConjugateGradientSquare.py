import numpy as np
import random
# 4.4.3 Conjuagate Gradient Square 平方共轭梯度法
# 参考 《迭代方法和预处理技术》谷同祥 等 编著 科技出版社
# 个人水平有限，程序可能有错，仅供参考

n = 128
A = np.zeros((n,n))
xexact = np.zeros((n))
for i in range(n):
    for j in range(n):
        A[i,j] = random.random()
    A[i,i] = 4 # 对角占优越明显，越可能收敛，这里一般需要3000次循环
    xexact[i] = random.random()
b = np.dot(A,xexact)

x = np.zeros((n)) # 待求解的值
r = b - np.dot(A,x) # 残差
d = np.dot(np.transpose(A),r)
tmax = 10000
r_star = r.copy() # 只要r_start 和 r 的内积不为零就行了
u = r.copy()
q = np.zeros((n))
for t in range(0,tmax):
    
    Ad = np.dot(A,d)
    rho_star = np.dot(np.transpose(r), r_star)
    if abs(rho_star) < 1e-10:
        break
    d_star_dot_Ad = np.dot(np.transpose(r_star), Ad)
    if abs(d_star_dot_Ad) < 1e-10:
        break
    alpha = rho_star / d_star_dot_Ad
    q = u - alpha * Ad
    x = x + alpha * (u + q)
    r_old = r.copy()
    r = r - alpha * np.dot(A, (u + q))
    beta = np.dot(np.transpose(r), r_star) / rho_star
    u = r + beta * q
    d = u + beta * (q + beta * d)
    
    
bFinal = np.dot(A,x)