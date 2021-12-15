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

node_pos = np.array([[0,0,0],[2,0,0],[0,2,0],[0,0,2]],dtype = float)
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
    
    sumGradC = 0
    for i in range(4):
        for j in range(3):
            sumGradC += invMass*(gradC[i,j]*gradC[i,j])
    if sumGradC < 1e-10:
        break
    for i in range(4):
        node_pos[i,:] += dt * energy / sumGradC * invMass * gradC[i,:]
    
    
    time += 1