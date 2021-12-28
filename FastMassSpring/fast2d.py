'''
Copyright (C) 2021 胡虎护弧呼 - All Rights Reserved
You may use, distribute and modify this code under the
terms of the MIT license.

fast implicit mass spring 2d example
tutorials : https://zhuanlan.zhihu.com/p/450792199
'''
import numpy as np
node_num = 4 # 顶点数量
p = np.array([0,0,2,0,0,2,2,2]) # 顶点位置
constraint_num = 6 # 约束数量
A = np.array(([1,-1,0,0],
              [1,0,-1,0],
              [1,0,0,-1],
              [0,1,-1,0],
              [0,1,0,-1],
              [0,0,1,-1])) # 约束
h = 1
k = np.ones((6)) # 每根弹簧的刚度
m = np.ones((4)) # 每个质点的重量

def KroneckerProduct2(mat):
    row = mat.shape[0]
    col = mat.shape[1]
    res = np.zeros((row*2,col*2))
    for i in range(row):
        for j in range(col):
            res[i*2,j*2] = mat[i,j]
            res[i*2+1,j*2+1] = mat[i,j]
    return res

A2 = KroneckerProduct2(A)
d0 = np.dot(A2,p) # 每根弹簧的未变形时的向量
rest = np.zeros((constraint_num))
d = np.zeros((constraint_num * 2))
for i in range(constraint_num):
    rest[i] = np.sqrt(d0[i*2]**2 + d0[i*2+1]**2)

p[0] -=1
time = 2
timeFinal = 100
energy_rec = np.zeros((timeFinal))
ppd = np.dot(A2,p) - d
for i in range(constraint_num):
    energy_rec[0] += k[i] / 2 * np.sqrt(ppd[i*2]*ppd[i*2]  + ppd[i*2+1]*ppd[i*2+1])
energy_rec[1] = energy_rec[0]
M = np.zeros((node_num*2,node_num*2))
for i in range(node_num):
    M[i*2,i*2] = M[i*2+1,i*2+1] = m[i]
f_ext = np.zeros((node_num*2))

p_record = np.zeros((node_num*2,timeFinal)) # 每个时刻都保存的位置
p_record[:,0] = p
p_record[:,1] = p

def multipleAtA(vec):
    row = len(vec)
    res = np.zeros((row,row))
    for i in range(row):
        res[i] = vec[i] * vec[:]
    return res


kAtA = np.zeros((node_num,node_num))
for i in range(constraint_num):
    kAtA += k[i] * multipleAtA(A[i,:])
L = KroneckerProduct2(kAtA)
lhs = M + h*h*L # 左手项

kA = np.zeros((node_num,constraint_num))
for i in range(constraint_num):
    kA[:,i] = k[i] * A[i,:]
J = KroneckerProduct2(kA)


while time < timeFinal:
    y = 2 * p_record[:,time-1] - p_record[:,time-2]
    d0 = np.dot(A2,p) # 每根弹簧的未变形时的向量
    for i in range(constraint_num):
        norm = np.sqrt(d0[i*2]**2 + d0[i*2+1]**2)
        d[i*2] =  rest[i] * d0[i*2] / norm
        d[i*2+1] = rest[i] * d0[i*2+1] / norm
    rhs = h*h*np.dot(J,d) + np.dot(M,y) + h*h*f_ext # 右手项 
    p = np.dot(np.linalg.inv(lhs),rhs) # 求解
    p_record[:,time] = p[:]
    
    ppd = np.dot(A2,p) - d
    for i in range(constraint_num):
        energy_rec[time] += k[i] / 2 * np.sqrt(ppd[i*2]*ppd[i*2]  + ppd[i*2+1]*ppd[i*2+1])
    
    time += 1