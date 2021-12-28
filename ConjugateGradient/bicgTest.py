import numpy as np
import random
# 4.4.1 Bi Conjuagate Gradient 双共轭梯度法
# 参考 《迭代方法和预处理技术》谷同祥 等 编著 科技出版社
# 个人水平有限，程序可能有错，仅供参考

n = 128
A = np.zeros((n,n))
xexact = np.zeros((n))
for i in range(n):
    for j in range(n):
        A[i,j] = random.random()
    A[i,i] = 8 # 对角占优越明显，越可能收敛
    xexact[i] = random.random()
b = np.dot(A,xexact)

x = np.zeros((n)) # 待求解的值
r = b - np.dot(A,x) # 残差
d = np.dot(np.transpose(A),r)
tmax = 1000
r_star = r.copy() # 只要r_start 和 r 的内积不为零就行了
d_star = r_star.copy()

for t in range(0,tmax):
    
    Ad = np.dot(A,d)
    rho_star = np.dot(np.transpose(r), r_star)
    if abs(rho_star) < 1e-20:
        break
    d_star_dot_Ad = np.dot(np.transpose(d_star), Ad)
    if abs(d_star_dot_Ad) < 1e-20:
        break
    alpha = rho_star / d_star_dot_Ad
    x = x + alpha * d
    
    r = r - alpha * Ad
    ATd_star = np.dot(np.transpose(A), d_star)
    r_star = r_star - alpha * ATd_star
    beta = np.dot(np.transpose(r), r_star) / rho_star
    
    d = r + beta * d
    d_star = r_star + beta * d_star
    
    
bFinal = np.dot(A,x)

# 剩下的比较容易实现的 4.4.6 双共轭残差
# 4.4.7 稳定的双共轭残差