import numpy as np
import random
# 4.4.4 Bi Conjuagate Gradient Stable 稳定的双共轭梯度方法
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

r_star = r.copy() # 只要r_start 和 r 的内积不为零就行了
rho_old = 1
omega = 1
alpha = 1
v = np.zeros((n))
ite_max = 10000
for ite in range(0,ite_max):
    
    rho = np.dot(np.transpose(r), r_star)
    if abs(rho) < 1e-10:
        break
    if ite == 0:
        d = r.copy()
    else:
        beta = (rho / rho_old) * (alpha / omega)
        d = r + beta * (d - omega * v)
    
    v = np.dot(A,d)
    alpha = rho_old / np.dot(np.transpose(v),r_star)
    s = r - alpha * v
    
    if np.linalg.norm(s) < 1e-10:
        x = x + alpha * d
        break
    t = np.dot(A,s)
    omega = np.dot(np.transpose(s),t) / np.dot(np.transpose(t), t)
    x = x + alpha * d + omega * s
    r = s - omega * t
    rho = np.dot(np.transpose(r), r)
    if abs(rho) < 1e-10:
        break
    
bFinal = np.dot(A,x)

