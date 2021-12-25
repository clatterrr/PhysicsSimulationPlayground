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

node_pos = np.array([[0,0,0],[1,0,0],[0,5,0],[0,0,2]],dtype = float)
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
    # lame常数
    mu = 2
    # lame常数
    la = 2
    # Stvk的能量计算公式
    Ic = 0
    IIc = 0
    IIIc = np.linalg.det(FtF)
    for i in range(3):
        for j in range(3):
            Ic += F[i,j]*F[i,j]
            IIc += FtF[i,j]*FtF[i,j]
    energy = mu * 0.5 * (Ic - 3) - mu * logJ + la * 0.5 * logJ * logJ
    pJpF = np.zeros((3,3))
    pJpF[:,0] = np.cross(F[:,1], F[:,2])
    pJpF[:,1] = np.cross(F[:,2], F[:,0])
    pJpF[:,2] = np.cross(F[:,0], F[:,1])
    # 第一种写法
    piola = mu * (F - 1.0 / J * pJpF) + la * logJ / J * pJpF
    # 第二种写法
    # piola = mu * F - mu * FinvT + la * np.log(IIIc) * 0.5 * FinvT
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