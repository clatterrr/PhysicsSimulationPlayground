import numpy as np

node_pos = np.array([[0,0,0],[1,0,0],[0,1,0],[0,0,1]],dtype = float)
Dm = np.zeros((3,3))
Dm[:,0] = node_pos[1,:] - node_pos[0,:]
Dm[:,1] = node_pos[2,:] - node_pos[0,:]
Dm[:,2] = node_pos[3,:] - node_pos[0,:]
Dminv = np.linalg.inv(Dm)
volume = np.linalg.det(Dm) * 0.1666667

mass = 1
dt = 1
invMass = 1 / mass

young = 10
nu = 0.4
mu = young / ( 2 * (1 + nu))
la = young * nu / (1 + nu) / (1 - 2 * nu)

node_pos = np.array([[0,0,0],[1,0,0],[0,1,0],[0,0,1.1]],dtype = float)
node_vel = np.zeros((4,3))
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
    Finv = np.linalg.inv(F)
    FinvT = Finv.T
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
    
    node_force = gradC.copy()
    
    dD = np.zeros((12,3,3))
    dD[0] = np.array([[-1,-1,-1],[0,0,0],[0,0,0]])
    dD[1] = np.array([[0,0,0],[-1,-1,-1],[0,0,0]])
    dD[2] = np.array([[0,0,0],[0,0,0],[-1,-1,-1]])
    dD[3] = np.array([[1,0,0],[0,0,0],[0,0,0]])
    dD[4] = np.array([[0,0,0],[1,0,0],[0,0,0]])
    dD[5] = np.array([[0,0,0],[0,0,0],[1,0,0]])
    dD[6] = np.array([[0,1,0],[0,0,0],[0,0,0]])
    dD[7] = np.array([[0,0,0],[0,1,0],[0,0,0]])
    dD[8] = np.array([[0,0,0],[0,0,0],[0,1,0]])
    dD[9] = np.array([[0,0,1],[0,0,0],[0,0,0]])
    dD[10] = np.array([[0,0,0],[0,0,1],[0,0,0]])
    dD[11] = np.array([[0,0,0],[0,0,0],[0,0,1]])
    
    dF = np.zeros((12,3,3))
    dP = np.zeros((12,3,3))
    dH = np.zeros((12,3,3))
    for i in range(12):
        dF[i] = np.dot(dD[i],Dminv) 
        dP[i] = mu * dF[i]
        dP[i] += (mu - la * logJ) * np.dot(np.dot(FinvT,dF[i].T),FinvT)
        term = np.dot(Finv,dF[i])
        dP[i] += la * (term[0,0] + term[1,1] + term[2,2]) * FinvT
        dH[i] = - volume * np.dot(dP[i],Dminv.T)
    
    
    # 四个顶点，每个顶点三个维度
    K = np.zeros((12,12))
    for n in range(4):
        for d in range(3):
            idx = n * 3 + d
            K[3,idx] = dH[idx,0,0]
            K[4,idx] = dH[idx,1,0]
            K[5,idx] = dH[idx,2,0]
            
            K[6,idx] = dH[idx,0,1]
            K[7,idx] = dH[idx,1,1]
            K[8,idx] = dH[idx,2,1]
            
            K[9,idx] = dH[idx,0,2]
            K[10,idx] = dH[idx,1,2]
            K[11,idx] = dH[idx,2,2]
            
            K[0,idx] = - dH[idx,0,0] - dH[idx,0,1] - dH[idx,0,2]
            K[1,idx] = - dH[idx,1,0] - dH[idx,1,1] - dH[idx,1,2]
            K[2,idx] = - dH[idx,2,0] - dH[idx,2,1] - dH[idx,2,2]
    
    A = np.identity(12) - K * dt * dt / mass
    b = np.zeros((12))
    for n in range(4):
        for d in range(3):
            b[n*3+d] = node_vel[n,d] + dt/mass*node_force[n,d]
    x = np.dot(np.linalg.inv(A),b)
    for n in range(4):
        for d in range(3):
            node_vel[n,d] = x[n*3+d] 
            node_pos[n,d] += node_vel[n,d]*dt
    time += 1