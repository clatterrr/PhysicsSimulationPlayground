'''
Copyright (C) 2021 胡虎护弧呼 - All Rights Reserved
You may use, distribute and modify this code under the
terms of the MIT license.

fast implicit mass spring 2d example
tutorials : https://zhuanlan.zhihu.com/p/450792199
'''
import numpy as np
node_row_num = 3
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
stiffness = np.ones((constraint_num)) # 每根弹簧的刚度
m = np.ones((node_num)) # 每个质点的重量
m = np.ones((node_num))
M = np.zeros((node_num*2,node_num*2))
for i in range(node_num):
    M[i*2,i*2] = M[i*2+1,i*2+1] = m[i]
    
L = np.zeros((node_num*2,node_num*2))
J = np.zeros((node_num*2,constraint_num*2))
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
    k2 = stiffness[cnt] * np.identity(2) # Kronecker Product
    
    L[st2:st2+2,st2:st2+2] += k2
    L[st2:st2+2,ed2:ed2+2] -= k2
    L[ed2:ed2+2,st2:st2+2] -= k2
    L[ed2:ed2+2,ed2:ed2+2] += k2
    
    J[st2:st2+2,cnt*2:cnt*2+2] += k2
    J[ed2:ed2+2,cnt*2:cnt*2+2,] -= k2
    
    cnt += 1
    
for j in range(node_row_num):
    for i in range(node_row_num):
        idx = j * node_row_num + i
        if i < node_row_num - 1:
           ConstructMatrix(idx, idx+1)
        if j < node_row_num - 1:
            ConstructMatrix(idx, idx+node_row_num)
        if i < node_row_num - 1 and j < node_row_num - 1:
            ConstructMatrix(idx, idx + node_row_num + 1)
            ConstructMatrix(idx + 1, idx + node_row_num)

Qmat = M + h*h*L
bvec = np.zeros((node_num * 2))
# 打死不用库函数
ch_n = node_num * 2
ch_L = np.zeros((ch_n,ch_n))
ch_v = np.zeros((ch_n))
for j in range(0,ch_n):
    for i in range(j,ch_n):
        ch_v[i] = Qmat[i,j]
        for k in range(0,j):
            # A 矩阵的形式是上三角
            ch_v[i] -= ch_L[j,k]*ch_L[i,k]
        ch_L[i,j] = ch_v[i] / np.sqrt(ch_v[j])
# 运用cholesky分解快速求线性矩阵，复杂度n*n
Qt = np.dot(ch_L,ch_L.T)
def choleskySolver(rhs):
    global ch_L
    resultTemp = np.zeros((ch_n))
    # 求解 L * result = rhs，即 result = L^-1 rhs
    # forward subtitution 前向替换
    for i in range(ch_n):
        resultTemp[i] = rhs[i] / ch_L[i,i]
        for j in range(i):
            resultTemp[i] -= ch_L[i,j] / ch_L[i,i] * resultTemp[j]
    # 求解 L^T * result = rhs 即 result = L^T^-1 rhs
    # backward subtitution 后向替换
    result = np.zeros((ch_n))
    for i in range(ch_n-1,-1,-1):
        result[i] = resultTemp[i] / ch_L[i,i]
        for j in range(i+1,ch_n):
            result[i] -= ch_L[j,i] / ch_L[i,i] * result[j]
    return result

time = 2
timeFinal = 10
node_pos_rec = np.zeros((node_num*2,timeFinal))
energy_rec = np.zeros((timeFinal))
f_ext = np.zeros((node_num * 2))
node_pos[0] = -1
node_pos_rec[:,0] = node_pos
node_pos_rec[:,1] = node_pos

while time < timeFinal:
    # local solver
    for i in range(constraint_num):
        st = node_pos[(d_st[i]*2):(d_st[i]*2+2)]
        ed = node_pos[(d_ed[i]*2):(d_ed[i]*2+2)]
        norm = np.sqrt((st[0] - ed[0])**2 + (st[1] - ed[1])**2)
        d_val[(i*2):(i*2+2)] = rest[i] * (st - ed) / norm
        energy_rec[time] += (norm - rest[i]) * stiffness[i] 
    y = 2 * node_pos_rec[:,time-1] - node_pos_rec[:,time-2]
    rhs = h*h*np.dot(J,d_val) + np.dot(M,y) + h*h*f_ext # 右手项 
    rh1 = np.dot(np.linalg.inv(Qmat),node_pos)
    # global solver
    node_pos= choleskySolver(rhs)
    node_pos_rec[:,time] = node_pos
    
    time += 1
