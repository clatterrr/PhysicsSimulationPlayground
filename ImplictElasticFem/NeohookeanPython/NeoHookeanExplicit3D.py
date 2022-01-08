import numpy as np

node_pos = np.array([[0,0,0],[1,0,0],[0,1,0],[0,0,1]],dtype = float)
Dm = np.zeros((3,3))
Dm[:,0] = node_pos[1,:] - node_pos[0,:]
Dm[:,1] = node_pos[2,:] - node_pos[0,:]
Dm[:,2] = node_pos[3,:] - node_pos[0,:]
Dminv = np.linalg.inv(Dm)
volume = np.linalg.det(Dm) * 0.1666667

mass = 1
dt = 0.1
invMass = 1 / mass

young = 100
nu = 0.4
mu = young / ( 2 * (1 + nu))
la = young * nu / (1 + nu) / (1 - 2 * nu)

node_pos = np.array([[0,0,0],[0.1,0,0],[0,0.1,0],[0,0,0.1]],dtype = float)
time = 0
timeFinal = 100
volumet = np.zeros((timeFinal))
while(time < timeFinal):
    Ds = np.zeros((3,3))
    Ds[:,0] = node_pos[1,:] - node_pos[0,:]
    Ds[:,1] = node_pos[2,:] - node_pos[0,:]
    Ds[:,2] = node_pos[3,:] - node_pos[0,:]
    # 记录体积
    volumet[time] = np.linalg.det(Ds) * 0.1666667
    # deformation gradient 变形梯度
    F = np.dot(Ds,Dminv)
    FinvT = np.linalg.inv(F).T
    FtF = np.dot(F.T,F)
    
    J = np.linalg.det(F)
    
    logJ = np.log(J)
    # 第一不变量
    Ic = FtF[0,0] + FtF[1,1] + FtF[2,2]
    # 可压缩 neohookean 能量
    energy = mu * 0.5 * (Ic - 3) - mu * logJ + la * 0.5 * logJ * logJ
    # 第一 piola kirchhoff 应力
    piola = mu * F - mu * FinvT + la * logJ * 0.5 * FinvT
    # 面积
    H = - volume * np.dot(piola, Dminv.T)
    gradC = np.zeros((4,3))
    gradC[1,:] = H[:,0]
    gradC[2,:] = H[:,1]
    gradC[3,:] = H[:,2]
    gradC[0,:] = - H[:,0] - H[:,1] - H[:,2]
    
    sumGradC = 0
    for i in range(4):
        for j in range(3):
            sumGradC += invMass*(gradC[i,j]*gradC[i,j])
    if sumGradC < 1e-10:
        break
    for i in range(4):
        node_pos[i,:] += dt * energy / sumGradC * invMass * gradC[i,:]
    
    
    time += 1