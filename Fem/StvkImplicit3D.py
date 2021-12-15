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

node_pos = np.array([[0,0,0],[1.1,0,0],[0,1,0],[0,0,1]],dtype = float)
node_vel = np.zeros((4,3))
node_force = np.zeros((4,3))
time = 0
timeFinal = 100
volumet = np.zeros((timeFinal))
while(time < timeFinal):
    node_force[:,:] = 0
    Ds = np.zeros((3,3))
    Ds[:,0] = node_pos[1,:] - node_pos[0,:]
    Ds[:,1] = node_pos[2,:] - node_pos[0,:]
    Ds[:,2] = node_pos[3,:] - node_pos[0,:]
    
    if(time == 50):
        time = 50
    # 记录体积
    volumet[time] = np.linalg.det(Ds) * 0.1666667
    # deformation gradient 变形梯度
    F = np.dot(Ds,Dminv)
    # green strain 
    E = (np.dot(F.T,F)- np.identity(3)) * 0.5
    # lame 常数
    mu = 2
    # lame 常数
    la = 2
    # 双点积
    doubleInner = 0
    for i in range(3):
        for j in range(3):
            doubleInner += E[i,j]*E[i,j]
    # trace 迹
    trE = E[0,0] + E[1,1] + E[2,2]
    # 能量
    energy = doubleInner * mu + la * 0.5 * trE * trE
    # first piola kirchhoff stress
    piola = np.dot(F,2*mu*E + la * trE * np.identity(3))
    # 面积
    H = - volume * np.dot(piola, Dminv.T)
    gradC = np.zeros((4,3))
    gradC[1,:] = H[:,0]
    gradC[2,:] = H[:,1]
    gradC[3,:] = H[:,2]
    gradC[0,:] = - H[:,0] - H[:,1] - H[:,2]
    
    node_force = gradC.copy()
    
    sumGradC = 0
    for i in range(4):
        for j in range(3):
            sumGradC += invMass*(gradC[i,j]*gradC[i,j])
    if sumGradC < 1e-10:
        break

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
    dE = np.zeros((12,3,3))
    dP = np.zeros((12,3,3))
    dH = np.zeros((12,3,3))
    for i in range(12):
        dF[i] = np.dot(dD[i],Dminv)
        dE[i] = (np.dot(dF[i,:,:].T,F) + np.dot(F.T,dF[i,:,:]))*0.5
        trdE = dE[i,0,0] + dE[i,1,1] + dE[i,2,2]
        dP[i] = np.dot(dF[i],2*mu*E + la*trE*np.identity(3))
        dP[i] += np.dot(F,2*mu*dE[i] + la*trdE*np.identity(3))
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