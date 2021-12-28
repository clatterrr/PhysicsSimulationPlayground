'''
Copyright (C) 2021 胡虎护弧呼 - All Rights Reserved
You may use, distribute and modify this code under the
terms of the MIT license.

implicit mass-spring system 2d example
tutorials : https://zhuanlan.zhihu.com/p/450792199
'''
import numpy as np
node_row_num = 4
node_num = node_row_num * node_row_num 
dx = 2
node_pos = np.zeros((node_num * 2)) # 顶点位置
for j in range(node_row_num):
    for i in range(node_row_num):
        idx = (j * node_row_num + i) * 2
        node_pos[idx] = i * dx
        node_pos[idx + 1] = j * dx
h = 1
# 只考虑了stretch 和 shear
stretch_num = (node_row_num - 1) * node_row_num * 2
shear_num = (node_row_num - 1) * (node_row_num - 1) * 2
constraint_num = stretch_num + shear_num
k = np.ones((constraint_num)) # 每根弹簧的刚度
m = np.ones((node_num)) # 每个质点的重量

m = np.ones((node_num))
M = np.zeros((node_num*2,node_num*2))
for i in range(node_num):
    M[i*2,i*2] = M[i*2+1,i*2+1] = m[i]
    
d_st = np.zeros((constraint_num),dtype = int)
d_ed = np.zeros((constraint_num),dtype = int)
d_val = np.zeros((constraint_num*2))
rest = np.zeros((constraint_num))
cnt = 0

def ConstructMatrix(st,ed):
    global cnt
    d_st[cnt] = st
    d_ed[cnt] = ed
    st2 = st * 2
    ed2 = ed * 2
    diff_x = node_pos[st2] - node_pos[ed2]
    diff_y = node_pos[st2+1] - node_pos[ed2+1]
    rest[cnt] = np.sqrt(diff_x**2 + diff_y**2)
    cnt += 1
    
for j in range(node_row_num):
    for i in range(node_row_num):
        idx = j * node_row_num + i
        if i < node_row_num - 1:
            ConstructMatrix(idx, idx+1)
        if j < node_row_num - 1:
            ConstructMatrix(idx, idx+node_row_num)
        if i > 0 and j > 0:
            ConstructMatrix(idx - node_row_num - 1, idx)
            ConstructMatrix(idx - node_row_num, idx - 1)
            
node_pos[0] = -10
time = 2
timeFinal = 100
node_pos_rec = np.zeros((node_num*2,timeFinal))
node_pos_rec[:,0] = node_pos_rec[:,1] = node_pos
energy_rec = np.zeros((timeFinal))

f = np.zeros((node_num * 2))
df = np.zeros((node_num * 2, node_num * 2))

def computeAAt(vec):
    size = len(vec)
    res = np.zeros((size,size))
    for i in range(size):
        res[i,:] = vec[i] * vec[:]
    return res

while time < timeFinal:
    energy = 0
    f[:] = 0
    df[:,:] = 0
    for ic in range(constraint_num):
        # 能量
        st = int(d_st[ic] * 2)
        ed = int(d_ed[ic] * 2)
        x = node_pos[st:st+2] - node_pos[ed:ed+2]
        absx = np.sqrt(x[0]*x[0] + x[1]*x[1])
        energy += 0.5 * k[ic] * (absx - rest[ic]) ** 2
        
        # 能量求导
        hatx = x / absx
        jacobian = k[ic] * (absx - rest[ic]) * hatx
        f[st:st+2] -= jacobian
        f[ed:ed+2] += jacobian
        
        # 二阶导
        xxt = computeAAt(hatx)
        hessian = k[ic] * (np.identity(2) - rest[ic] / absx * (np.identity(2) - xxt))
        df[st:st+2,st:st+2] -= hessian
        df[st:st+2,ed:ed+2] += hessian
        df[ed:ed+2,st:st+2] += hessian
        df[ed:ed+2,ed:ed+2] -= hessian
        
    Amat = M - h * h * df
    inertia = 2 * node_pos_rec[:,time-1] - node_pos_rec[:,time-2]
    bvec = np.dot(M,inertia) + h*h*f - h*h*np.dot(df,node_pos)
    node_pos = np.dot(np.linalg.inv(Amat),bvec)
    node_pos_rec[:,time] = node_pos
    energy_rec[time] = energy
    time += 1