import numpy as np
# 初始化三角形初始位置
node_pos = np.array([[0,0],[1,0],[0,1]],dtype = float)
# 顶点的位置梯度
Ds = np.array([[node_pos[1,0] - node_pos[0,0],node_pos[2,0] -
node_pos[0,0]],
[node_pos[1,1] - node_pos[0,1],node_pos[2,1] -
node_pos[0,1]]])
# 求逆，用于准备计算形变梯度
minv = np.linalg.inv(Ds)
# 假设某一时刻，三角形变化到了这样的位置
node_pos = np.array([[0,0],[2,1],[0,2]],dtype = float)

young = 100
nu = 0.4
mu = young / ( 2 * (1 + nu))
la = young * nu / (1 + nu) / (1 - 2 * nu)

time = 0
timeFinal = 1
areat = np.zeros((timeFinal))
while(time < timeFinal):
    time += 1
    # 形变梯度中的分子
    Ds_new = np.array([[node_pos[1,0] - node_pos[0,0],node_pos[2,0]
    - node_pos[0,0]],
    [node_pos[1,1] - node_pos[0,1],node_pos[2,1] -
    node_pos[0,1]]])
    # 形变梯度
    F = np.dot(Ds_new,minv)
    C = np.dot(F.T,F)
    Cinv = np.linalg.inv(C)
    J = np.linalg.det(F)
    logJ = np.log(J)
    la = mu = 2
    # 第一不变量
    Ic = C[0,0] + C[1,1] 
    # 可压缩 neohookean 能量
    energy = mu * 0.5 * (Ic - 2) - mu * logJ + la * 0.5 * logJ * logJ
    # 第二 piola kirchhoff 应力
    piola = mu * (np.identity(2) - Cinv) + la * logJ * Cinv
    # 三角形面积
    area = 0.5
    # 计算力
    H = - area * np.dot(piola,minv.transpose())
    gradC1 = np.array([H[0,0],H[1,0]])
    gradC2 = np.array([H[0,1],H[1,1]])
    gradC0 = - gradC1 - gradC2
    invMass = 1
    dt = 0.1
    # 判断是否收敛
    sumGradC = invMass * (gradC0[0]**2 + gradC0[1]**2)
    sumGradC += invMass * (gradC1[0]**2 + gradC1[1]**2)
    sumGradC += invMass * (gradC2[0]**2 + gradC2[1]**2)
    if sumGradC < 1e-10:
        break
    # 校正位置，方法来源于PositionBasedDynamics
    node_pos[0,:] += dt * energy / sumGradC * invMass * gradC0
    node_pos[1,:] += dt * energy / sumGradC * invMass * gradC1
    node_pos[2,:] += dt * energy / sumGradC * invMass * gradC2
    areat[time - 1] = 0.5 * (node_pos[0,0] * (node_pos[1,1] - node_pos[2,1])
                    + node_pos[1,0] * (node_pos[2,1] - node_pos[0,1]) 
                    + node_pos[2,0] * (node_pos[0,1] - node_pos[1,1]))
    
    