import numpy as np
# 4.4.1 Bi Conjuagate Gradient 双共轭梯度法
# 参考 《迭代方法和预处理技术》谷同祥 等 编著 科技出版社
# 个人水平有限，程序可能有错，仅供参考
A = np.array([[9,0,1,1,2,3,2,1],
              [4,9,1,6,7,8,6,2],
              [4,2,9,4,5,6,7,1],
              [4,6,1,9,3,5,1,6],
              [7,5,3,1,9,6,7,8],
              [1,5,6,7,8,9,2,5],
              [5,3,1,2,5,6,9,8],
              [8,6,5,4,2,1,2,9]],dtype = float)
b = np.array([66,190,169,155,224,205,210,153],dtype = float)
# x = np.array([1,2,3,4,5,6,7,8]) # 正确的解
nmax = 8

x = np.zeros((nmax)) # 待求解的值
r = b - np.dot(A,x) # 残差
d = np.dot(np.transpose(A),r)
tmax = 100
r_star = r.copy() # 只要r_start 和 r 的内积不为零就行了
d_star = r_star.copy()

for t in range(0,tmax):
    rho = np.dot(np.transpose(r),r)
    if abs(rho) < 1e-10:
        break
    Ad = np.dot(A,d)
    rho_star = np.dot(np.transpose(r), r_star)
    alpha = rho_star / np.dot(np.transpose(d_star), Ad)
    x = x + alpha * d
    
    r = r - alpha * Ad
    ATd_star = np.dot(np.transpose(A), d_star)
    r_star = r_star - alpha * ATd_star
    beta = np.dot(np.transpose(r), r_star) / rho_star
    
    d = r + beta * d
    d_star = r_star + beta * d_star
    
    
        