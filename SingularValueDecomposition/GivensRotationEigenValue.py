'''
Copyright (C) 2022 胡虎护弧呼 - All Rights Reserved
You may use, distribute and modify this code under the
terms of the GPL license.

3x3 singular value decomposition using givens rotation
tutorials : https://zhuanlan.zhihu.com/p/459369233
'''
import numpy as np

A = np.array([[1,2,3],[4,5,3],[7,8,10]],dtype = float)
AtA = A.T @ A
AtA0 = AtA.copy()
s0 = np.zeros((3))


ei = np.linalg.eig(AtA)
    
def GivensRotate(a11,a12,a22):
    xi = 0.5 * (a11 - a22) / a12
    w = np.sqrt(xi * xi + 1)
    if xi > 0:
    	t = 1 / (xi + w)
    else:
        t = 1 / (xi - w)
    c = 1.0 / np.sqrt(t * t + 1)
    s = - t * c
    return c,s

def computeEigenVector(Ae):
    Ae[1,:] = Ae[1,:] - Ae[0,:] * Ae[1,0] / Ae[0,0]
    res = np.ones((3))
    res[1] = - Ae[1,2] / Ae[1,1]
    res[0] = - (Ae[0,1] * res[1] + Ae[0,2] * res[2]) / Ae[0,0]
    res = res / np.linalg.norm(res)
    return res

for it in range(10):
    
    c,s = GivensRotate(AtA[0,0], AtA[0,1], AtA[1,1])
    Q = np.array([[c,s,0],[-s,c,0],[0,0,1]])
    AtA = Q.T @ AtA @ Q
    
    c,s = GivensRotate(AtA[1,1], AtA[1,2], AtA[2,2])
    Q = np.array([[1,0,0],[0,c,s],[0,-s,c]])
    AtA = Q.T @ AtA @ Q
    
    c,s = GivensRotate(AtA[0,0], AtA[0,2], AtA[2,2])
    Q = np.array([[c,0,s],[0,1,0],[-s,0,c]])
    AtA = Q.T @ AtA @ Q
    
    res = AtA[0,1]**2 + AtA[0,2]**2 + AtA[1,2]**2
    if res < 1e-10:
        break
    
s0[0] = np.sqrt(AtA[0,0])
s0[1] = np.sqrt(AtA[1,1])
s0[2] = np.sqrt(AtA[2,2])

if s0[0] < s0[1]:
    temp = s0[0]
    s0[0] = s0[1]
    s0[1] = temp
if s0[0] < s0[2]:
    temp = s0[0]
    s0[0] = s0[2]
    s0[2] = temp
if s0[1] < s0[2]:
    temp = s0[1]
    s0[1] = s0[2]
    s0[2] = temp 

vt0 = np.zeros((3,3))
vt0[0,:] = computeEigenVector(AtA0 - np.identity(3)*abs(s0[0]*s0[0]))
vt0[1,:] = computeEigenVector(AtA0 - np.identity(3)*abs(s0[1]*s0[1]))
vt0[2,:] = computeEigenVector(AtA0 - np.identity(3)*abs(s0[2]*s0[2]))

u0 = np.zeros((3,3))
u0[:,0] = A @ vt0[0,:] / s0[0]
u0[:,1] = A @ vt0[1,:] / s0[1]
u0[:,2] = A @ vt0[2,:] / s0[2]

u1,s1,vt1 = np.linalg.svd(A)