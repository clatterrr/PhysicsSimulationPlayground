import numpy as np
# box2d
# fail

x0 = np.array([0,0],dtype = float) # 刚体 0 的重心位置
x1 = np.array([6,0],dtype = float)
r0 = np.array([2,0],dtype = float) # 刚体 0 的摆臂向量
r1 = np.array([-2,0],dtype = float)

m0 = 1
m1 = 1
I0 = 1
I1 = 1
v0 = np.zeros((2))
v0[0] = 1
v1 = np.zeros((2))
w0 = 0
w1 = 0
rest_len = 1

J = np.array([[-1,0,r0[1],-r0[0],1,0,-r1[1],r1[0]],
              [0,-1,0,        -1,0,1,0,     1]])

W = np.identity(8)

K = J @ W @ J.T

K[0,0] = 1.0 / m0 + 1.0 / m1 + I0 * r0[1] * r0[1] + I1 * r1[1] * r1[1]
K[0,1] = - I0 * r0[0] * r0[1] - I1 * r1[0] * r1[1]
K[1,0] = K[0,1]
K[1,1] = 1.0 / m0 + 1.0 / m1 + I0 * r0[0] * r0[0] + I1 * r1[0] * r1[0] 

Kinv = np.linalg.inv(K)


def cross2d(a,b):
    return a[0] * b[1] - a[1] * b[0]

def cross2d2(a,b):
    return np.array([-a * b[1], a * b[0]])

time = 0
timeFinal = 100
dt = 0.1
x_record = np.zeros((timeFinal,2,2))
while time < timeFinal:
    x_record[time,0] = x0
    x_record[time,1] = x1
    
    # angular
    angCdot = w1 - w0 # 角速度的差值
    angularMass = 1
    impluseAngular = - angularMass * angCdot
    
    velCdot = v1 - v0 + cross2d2(w1, r1) - cross2d2(w0, r0)
    impluseVelocity = - Kinv @ velCdot
    v0 -= impluseVelocity / m0
    v1 += impluseVelocity / m1
    w0 = w0 - impluseAngular * I0 - I0 * cross2d(r0, impluseVelocity)
    w1 = w1 + impluseAngular * I1 + I1 * cross2d(r1, impluseVelocity)
    x0 = x0 + v0 * dt
    x1 = x1 + v1 * dt
    r0 = r0 + dt * cross2d2(w0, r0)
    r1 = r1 + dt * cross2d2(w1, r1)
    time += 1
    
    
    
    
