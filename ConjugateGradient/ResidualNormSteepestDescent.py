import numpy as np
# 1.5.4 残差范数最速下降
# 参考 《迭代方法和预处理技术》谷同祥 等 编著 科技出版社
# 个人水平有限，程序可能有错，仅供参考

A = np.array([[3,0,1],[0,4,2],[1,2,3]])
b = np.array([1,0,0])
n = 3

r = np.zeros((3)) # 残差
x = np.zeros((3))

error = 1
eps = 1e-10
ite_max = 100
r = b - np.dot(A,x)
for ite in range(ite_max):
    v = np.dot(np.transpose(A),r)
    Av = np.dot(A, v)
    
    v2 = np.dot(np.transpose(v),v)
    Av2 = np.dot(np.transpose(Av),Av)
    a = v2 / Av2

    x = x + a * v
    r = r - a * Av    
        
    error = np.dot(np.transpose(r),r)
    if(error < eps): # 此时收敛
        break

    