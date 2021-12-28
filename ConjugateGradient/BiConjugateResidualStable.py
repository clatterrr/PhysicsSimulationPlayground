import numpy as np
import random
# 4.4.7 Bi Conjuagate Residual Stable 稳定的双共轭残差法
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
v = np.zeros((n))
q = np.zeros((n))
u = np.dot(A,r)
rho_old = 1
alpha = 1
omega = 1
for ite in range(0,tmax):
    rho = np.dot(np.transpose(u),r_star)
    delta = rho / rho_old * alpha
    rho_old = rho
    
    beta = delta / omega
    d = r + beta * d - delta * v
    v = u + beta * v - delta * q
    q = np.dot(A,v)
    # 注意r_star 一直不会被更新
    alpha = rho / np.dot(np.transpose(q),r_star)
    s = r - alpha * v
    t = u - alpha * q
    s_dot_t = np.dot(np.transpose(s),t)
    t_dot_t = np.dot(np.transpose(t),t)
    if abs(t_dot_t) < 1e-10:
        break
    omega = s_dot_t / t_dot_t
    x = x + alpha * d + omega * s
    r = s - omega * t
    u = np.dot(A,r)
    
    
