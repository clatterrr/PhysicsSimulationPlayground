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
node_pos = np.array([[0,0],[2,0],[0,1]],dtype = float)
time = 0
timeFinal = 100
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
    # 应力，也就是varepsilon
    strain = (F + F.T) * 0.5 - np.identity(2)
    # lame常数
    mu = 2
    # lame常数
    la = 2
    #
    doubleInner = strain[0,0]*strain[0,0] + strain[1,0]*strain[1,0] + strain[0,1]*strain[0,1] + strain[1,1]*strain[1,1]
    # 线弹性的能量计算公式
    energy = doubleInner * mu + la * 0.5 * np.trace(strain) ** 2
    #first piola kirchhoff stress
    piola = mu * (F + F.T - 2 * np.identity(2)) + la * (F[0,0] - 1 + F[1,1] - 1) * np.identity(2)
    # 三角形面积
    area = 0.5
    # 计算力
    H = - area * np.dot(piola,minv.transpose())
    gradC0 = np.array([H[0,0],H[1,0]])
    gradC1 = np.array([H[0,1],H[1,1]])
    gradC2 = - gradC0 - gradC1
    invMass = 1
    dt = 0.1
    # 判断是否收敛
    sumGradC = invMass * (gradC0[0]**2 + gradC0[1]**2)
    sumGradC += invMass * (gradC1[0]**2 + gradC1[1]**2)
    sumGradC += invMass * (gradC2[0]**2 + gradC2[1]**2)
    if sumGradC < 1e-10:
        break
    # 校正位置，方法来源于PositionBasedDynamics
    node_pos[0,:] += dt * energy / sumGradC * invMass * gradC2
    node_pos[1,:] += dt * energy / sumGradC * invMass * gradC0
    node_pos[2,:] += dt * energy / sumGradC * invMass * gradC1
    areat[time - 1] = 0.5 * (node_pos[0,0] * (node_pos[1,1] - node_pos[2,1])
                    + node_pos[1,0] * (node_pos[2,1] - node_pos[0,1]) 
                    + node_pos[2,0] * (node_pos[0,1] - node_pos[1,1]))
    