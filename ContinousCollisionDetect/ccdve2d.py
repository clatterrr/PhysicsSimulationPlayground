import numpy as np

# 连续碰撞检测 二维 点-边

x0 = np.array([0.0,0.0]) # 点的位置
x1 = np.array([1.0,0.0]) # 边的顶点1的位置
x2 = np.array([0.0,1.0]) # 边的顶点2的位置
v0 = np.array([1.0,1.0]) # 点的速度
v1 = np.array([0.0,0.0]) # 边的顶点1的速度
v2 = np.array([0.0,0.0]) # 边的顶点2的速度

def cross2d(vec0,vec1):
    return vec0[0]*vec1[1] - vec0[1]*vec1[0]

a = cross2d(v0-v1, v2-v1)
b = cross2d(x0-x1, v2-v1) + cross2d(v0-v1, x2-x1)
c = cross2d(x0-x1, x2-x1)

d = b*b - 4*a*c
result = np.zeros((2))
result_num = 0
if d < 0:
    result_num = 1
    result[1] = - b / (2 * a)
else:
    q = - (b + np.sign(b)*np.sqrt(d)) / 2
    if (abs(a) > 1e-12*abs(q)):
        result[result_num] = q / a
        result_num += 1
    if (abs(q) > 1e-12*abs(c)):
        result[result_num] = c / q
        result_num += 1
    if result_num == 2 and result[0] > result[1]:
        temp = result[0]
        result[0] = result[1]
        result[1] = temp
        
collision = False
for i in range(result_num):
    t = result[i]
    x0new = x0 + t * v0
    x1new = x1 + t * v1
    x2new = x2 + t * v2
    res = cross2d(x0new - x1new, x2new - x1new)
    if abs(res) < 1e-10:
        collision = True