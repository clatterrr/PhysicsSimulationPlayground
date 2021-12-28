import numpy as np
'''
Copyright (C) 2021 胡虎护弧呼 - All Rights Reserved
You may use, distribute and modify this code under the
terms of the MIT license.

fast implicit mass spring 1d example
tutorials : https://zhuanlan.zhihu.com/p/450792199
'''
# 用户定义
d = 1 # 弹簧初始长度
k = 1 # 弹簧刚度矩阵
h = 1 # 时间步长
m = 1 # 质点质量
p = np.array([4.0,1.0]) # 两个质点现在的一维位置
A = np.array([[1],[-1]]) # 约束条件，也就是p1 - p2
f_ext = np.zeros((2))
# 自动计算
M = np.array([[m,0],[0,m]]) # 质量矩阵
time = 2
timeFinal = 10
p_record = np.zeros((2,timeFinal)) # 每个时刻都保存的位置
p_record[:,0] = p
p_record[:,1] = p
while time < timeFinal:
    L = k * np.dot(A,A.T) 
    lhs = M + h*h*L # 左手项
    J = A[:,0]
    y = 2 * p_record[:,time-1] - p_record[:,time-2]
    rhs = h*h*J*d + np.dot(M,y) + h*h*f_ext # 右手项
    p = np.dot(np.linalg.inv(lhs),rhs) # 求解
    p_record[:,time] = p[:]
    time += 1