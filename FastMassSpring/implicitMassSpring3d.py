'''
Copyright (C) 2021 胡虎护弧呼 - All Rights Reserved
You may use, distribute and modify this code under the
terms of the MIT license.

implicit mass-spring system 3d example
tutorials : https://zhuanlan.zhihu.com/p/450792199
'''
import numpy as np
node_row_num = 2
node_num = node_row_num * node_row_num 
dx = 1
node_pos = np.zeros((node_num * 3)) # 顶点位置
for j in range(node_row_num):
    for i in range(node_row_num):
        idx = (j * node_row_num + i) * 3
        node_pos[idx + 0] = i * dx
        node_pos[idx + 1] = j * dx
        node_pos[idx + 2] = 0
h = 1

stretch_num = (node_row_num - 1) * node_row_num * 2
shear_num = (node_row_num - 1) * (node_row_num - 1) * 2
bend_num = (node_row_num - 2) * node_row_num * 2
constraint_num = stretch_num + shear_num + bend_num
stiffness = np.ones((constraint_num)) # 每根弹簧的刚度
m = np.ones((node_num)) # 每个质点的重量
m = np.ones((node_num))
M = np.zeros((node_num*3,node_num*3))
for i in range(node_num):
    M[i*3,i*3] = M[i*3+1,i*3+1] = M[i*3+2,i*3+2] =m[i]
    
d_st = np.zeros((constraint_num),dtype = int)
d_ed = np.zeros((constraint_num),dtype = int)
d_val = np.zeros((constraint_num*2))
rest = np.zeros((constraint_num))
cnt = 0

def ConstructMatrix(st,ed):
    global cnt
    d_st[cnt] = st
    d_ed[cnt] = ed
    st3 = st * 3
    ed3 = ed * 3
    diff_x = node_pos[st3] - node_pos[ed3]
    diff_y = node_pos[st3+1] - node_pos[ed3+1]
    diff_z = node_pos[st3+2] - node_pos[ed3+2]
    rest[cnt] = np.sqrt(diff_x**2 + diff_y**2 + diff_z**2)
    cnt += 1
    
for j in range(node_row_num):
    for i in range(node_row_num):
        idx = j * node_row_num + i
        if i < node_row_num - 1:
            ConstructMatrix(idx, idx+1)
        if j < node_row_num - 1:
            ConstructMatrix(idx, idx+node_row_num)
        if i < node_row_num - 2:
            ConstructMatrix(idx, idx+2)
        if j < node_row_num - 2:
            ConstructMatrix(idx, idx+node_row_num*2)
        if i < node_row_num - 1 and j < node_row_num - 1:
            ConstructMatrix(idx, idx + node_row_num + 1)
            ConstructMatrix(idx + 1, idx + node_row_num)
            
node_pos[0] = -1
time = 2
timeFinal = 100
node_pos_rec = np.zeros((node_num*3,timeFinal))
node_pos_rec[:,0] = node_pos_rec[:,1] = node_pos
energy_rec = np.zeros((timeFinal))

f = np.zeros((node_num * 3))
df = np.zeros((node_num * 3, node_num * 3))

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
        st = int(d_st[ic] * 3)
        ed = int(d_ed[ic] * 3)
        x = node_pos[st:st+3] - node_pos[ed:ed+3]
        absx = np.sqrt(x[0]*x[0] + x[1]*x[1] + x[2]*x[2])
        energy += 0.5 * stiffness[ic] * (absx - rest[ic]) ** 2
        
        # 能量求导
        hatx = x / absx
        jacobian = stiffness[ic] * (absx - rest[ic]) * hatx
        f[st:st+3] -= jacobian
        f[ed:ed+3] += jacobian
        
        # 二阶导
        xxt = computeAAt(hatx)
        hessian = stiffness[ic] * (np.identity(3) - rest[ic] / absx * (np.identity(3) - xxt))
        df[st:st+3,st:st+3] -= hessian
        df[st:st+3,ed:ed+3] += hessian
        df[ed:ed+3,st:st+3] += hessian
        df[ed:ed+3,ed:ed+3] -= hessian
        
    Amat = M - h * h * df
    inertia = 2 * node_pos_rec[:,time-1] - node_pos_rec[:,time-2]
    term1 = np.dot(M,inertia)
    term2 = np.dot(df,node_pos)
    bvec = np.dot(M,inertia) + h*h*f - h*h*np.dot(df,node_pos)
    node_pos = np.dot(np.linalg.inv(Amat),bvec)
    node_pos_rec[:,time] = node_pos
    energy_rec[time] = energy
    time += 1