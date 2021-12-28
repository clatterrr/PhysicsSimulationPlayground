'''
Copyright (C) 2021 胡虎护弧呼 - All Rights Reserved
You may use, distribute and modify this code under the
terms of the MIT license.

implicit mass-spring system 1d example
tutorials : https://zhuanlan.zhihu.com/p/450792199
'''
import numpy as np
node_num = 2
node_pos = np.array([3,1],dtype = float)
k = 1
r = 1
h = 1
Mass = np.identity(2)
time = 2
timeFinal = 100
energy_rec = np.zeros((timeFinal))
node_pos_rec = np.zeros((node_num,timeFinal))
node_pos_rec[:,0] = node_pos_rec[:,1] = node_pos
while time < timeFinal:
    energy = k * (abs(node_pos[0] - node_pos[1]) - r)**2
    energy_rec[time] = energy
    
    # 
    absx = abs(node_pos[0] - node_pos[1])
    energy_jacobian = np.zeros((node_num))
    x = node_pos[0] - node_pos[1]
    energy_jacobian[0] = k * (absx - r) * x / absx
    x = node_pos[1] - node_pos[0]
    energy_jacobian[1] = k * (absx - r) * x / absx
    f = - energy_jacobian.copy()
    
    energy_hessian = np.zeros((node_num,node_num))
    x = node_pos[0] - node_pos[1]
    hatx = x / absx
    energy_hessian[0,0] = k * (1 - r / absx * (1 - hatx * hatx))
    energy_hessian[0,1] = - energy_hessian[0,0]
    energy_hessian[1,0] = - energy_hessian[0,0]
    energy_hessian[1,1] = energy_hessian[0,0]
    df = - energy_hessian.copy()
    
    Amat = Mass - df * h * h
    inertia = 2 * node_pos_rec[:,time-1] - node_pos_rec[:,time-2]
    bvec = np.dot(Mass,inertia) + h*h*f - h*h*np.dot(df,node_pos)
    
    node_pos = np.dot(np.linalg.inv(Amat),bvec)
    node_pos_rec[:,time] = node_pos
    
    time += 1