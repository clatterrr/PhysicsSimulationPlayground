import numpy as np
# fail
def crossProduct(a):
    return np.array([[0,-a[2],a[1]],
                     [a[2],0,-a[0]],
                     [-a[1],a[0],0]])

x = np.array([[0,0,0],
              [5,0,0]],dtype = float) # 刚体重心位置
r = np.array([[2,0,0],
              [-2,0,0]],dtype = float) #刚体摆臂向量
# p = x + r 即为碰撞点的位置
m = np.array([1,1],dtype = float) # 刚体的重量
W = np.identity((12))
W[0:3,0:3] = 1.0 / m[0] * np.identity(3)
W[6:9,6:9] = 1.0 / m[1] * np.identity(3)
J = np.zeros((3,12))
J[0,0] = J[1,1] = J[2,2] = -1
J[0,6] = J[1,7] = J[2,8] = 1
J[0:3,3:6] = crossProduct(r[0])
J[0:3,9:12] = crossProduct(-r[1])

time = 0
timeFinal = 100
x_record = np.zeros((timeFinal+1,2,3))
v = np.zeros((12))
dt = 0.01
beta = 20 * np.ones((3))
x_record[0] = x
while time < timeFinal:
    x_record[time+1] = x
    # 1 维情况，K 是一维矩阵，也就是只有一个数字
    K = J @ W @ J.T
    C = np.linalg.norm(x[1] + r[1] - x[0] - r[0])
    dCdt = J @ v
    b = beta / dt * C
    lam =  np.linalg.inv(K) @ (dCdt + b) 
    F_c = - lam @ J
    v = dt * F_c
    x[0] = x[0] + dt * v[0:3]
    r[0] = r[0] + np.cross(v[3:6], r[0]) * dt
    r[0] = r[0] / np.linalg.norm(r[0]) * 2
    x[1] = x[1] + dt * v[6:9]
    r[1] = r[1] + np.cross(v[9:12], r[1]) * dt
    r[1] = r[1] / np.linalg.norm(r[1]) * 2
    time += 1
    
