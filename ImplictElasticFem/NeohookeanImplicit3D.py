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

def crossMatrix(F,col,scale):
    res = np.array([[0,-scale*F[2,col],scale*F[1,col]],
                    [scale*F[2,col],0,-scale*F[0,col]],
                    [-scale*F[1,col],scale*F[0,col],0]])
    return res

node_pos = np.array([[0,0,0],[1,0,0],[0,5,0],[0,0,2]],dtype = float)
node_vel = np.zeros((4,3))
node_force = np.zeros((4,3))
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
    piola = mu * F + la * (J - 1 - mu / la) * pJpF
    # 第二种写法
    # piola = mu * F - mu * FinvT + la * np.log(IIIc) * 0.5 * FinvT
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
    
    pjpfvec = np.zeros((9))
    pjpfvec[0:3] = np.cross(F[:,1], F[:,2])
    pjpfvec[3:6] = np.cross(F[:,2], F[:,0])
    pjpfvec[6:9] = np.cross(F[:,0], F[:,1])
    
    scale = la * (J - 1 - mu / la)
    ahat = crossMatrix(F, 0, scale)
    bhat = crossMatrix(F, 1, scale)
    chat = crossMatrix(F, 2, scale)
    Fj = np.zeros((9,9))
    Fj[3:6,0:3] = - chat
    Fj[6:9,0:3] = bhat
    Fj[0:3,3:6] = chat
    Fj[6:9,3:6] = - ahat
    Fj[0:3,0:3] = - bhat
    Fj[3:6,0:3] = ahat
    
    pPpF = mu * np.identity(9) + la * np.dot(pjpfvec,pjpfvec.T) + Fj
    dH = np.zeros((12,3,3))
    for i in range(12):
        pFpU = np.dot(dD[i],Dminv)
        dH[i] = - volume * np.dot(np.dot(pFpU.T,pPpF),pFpU)
    
    
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