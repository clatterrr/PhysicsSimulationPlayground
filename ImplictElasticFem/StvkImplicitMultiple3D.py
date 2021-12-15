'''
三维隐式弹性有限元
'''
import numpy as np
node_row_num = 4
node_num = node_row_num * node_row_num * node_row_num
node_pos = np.zeros((node_num,3))
node_vel = np.zeros((node_num,3))
node_force = np.zeros((node_num,3))

for k in range(node_row_num):
    for j in range(node_row_num):
        for i in range(node_row_num):
            idx = k*node_row_num*node_row_num+j*node_row_num+i
            node_pos[idx] = np.array([i,j,k])
 
cube_row_num = node_row_num - 1
cube_num = cube_row_num * cube_row_num * cube_row_num
element_num = cube_num * 8
element_idx = np.zeros((element_num,4),dtype = int)
element_volume = np.zeros((element_num))
element_minv = np.zeros((element_num,3,3))
for k in range(cube_row_num):
    for j in range(cube_row_num):
        for i in range(cube_row_num):
            idx = k*cube_row_num*cube_row_num+j*cube_row_num+i
            nidx = k*node_row_num*node_row_num+j*node_row_num+i
            
            LeftBotFront = nidx
            RightBotFront = nidx + 1
            LeftTopFront = nidx + node_row_num
            RightTopFront = nidx + 1 + node_row_num
            
            LeftBotBack = nidx + node_row_num * node_row_num
            RightBotBack = nidx + 1 + node_row_num * node_row_num
            LeftTopBack = nidx + node_row_num + node_row_num * node_row_num
            RightTopBack = nidx + 1 + node_row_num  + node_row_num * node_row_num        
            
            
            element_idx[idx*8+0,0] = LeftBotFront
            element_idx[idx*8+0,1] = RightBotFront
            element_idx[idx*8+0,2] = LeftTopFront
            element_idx[idx*8+0,3] = LeftBotBack
            
            element_idx[idx*8+1,0] = RightBotFront
            element_idx[idx*8+1,1] = RightBotBack
            element_idx[idx*8+1,2] = RightTopFront
            element_idx[idx*8+1,3] = LeftBotFront
            
            element_idx[idx*8+2,0] = RightBotBack
            element_idx[idx*8+2,1] = LeftBotBack
            element_idx[idx*8+2,2] = RightTopBack
            element_idx[idx*8+2,3] = RightBotFront
            
            element_idx[idx*8+3,0] = LeftBotBack
            element_idx[idx*8+3,1] = LeftBotFront
            element_idx[idx*8+3,2] = LeftTopBack
            element_idx[idx*8+3,3] = RightBotBack
            
            element_idx[idx*8+4,0] = LeftTopFront
            element_idx[idx*8+4,1] = LeftBotFront
            element_idx[idx*8+4,2] = RightTopFront
            element_idx[idx*8+4,3] = LeftTopBack
            
            element_idx[idx*8+5,0] = RightTopFront
            element_idx[idx*8+5,1] = RightBotFront
            element_idx[idx*8+5,2] = RightTopBack
            element_idx[idx*8+5,3] = LeftTopFront
            
            element_idx[idx*8+6,0] = RightTopBack
            element_idx[idx*8+6,1] = RightBotBack
            element_idx[idx*8+6,2] = LeftTopBack
            element_idx[idx*8+6,3] = RightTopFront
            
            element_idx[idx*8+7,0] = LeftTopBack
            element_idx[idx*8+7,1] = LeftBotBack
            element_idx[idx*8+7,2] = LeftTopFront
            element_idx[idx*8+7,3] = RightTopBack

for i in range(element_num):
    p0 = node_pos[element_idx[i,0]]
    p1 = node_pos[element_idx[i,1]]
    p2 = node_pos[element_idx[i,2]]
    p3 = node_pos[element_idx[i,3]]
    dX = np.zeros((3,3))
    dX[:,0] = p1 - p0
    dX[:,1] = p2 - p0
    dX[:,2] = p3 - p0
    element_volume[i] = np.linalg.det(dX) * 0.16666667
    element_minv[i] = np.linalg.inv(dX)
    


mass = 1
dt = 1
invMass = 1 / mass
time = 0
timeFinal = 100

krow = node_num * 3
Kmat = np.zeros((krow,krow))
Amat = np.zeros((krow,krow))
bvec = np.zeros((krow))

volumet = np.zeros((element_num,timeFinal))
xvect = np.zeros((krow,timeFinal))
post = np.zeros((krow,timeFinal))

node_pos[node_row_num-1,0] += 10

while(time < timeFinal):
    node_force[:,:] = 0
    Kmat[:,:] = 0
    for ie in range(element_num):
        p0 = node_pos[element_idx[ie,0]]
        p1 = node_pos[element_idx[ie,1]]
        p2 = node_pos[element_idx[ie,2]]
        p3 = node_pos[element_idx[ie,3]]
        dx = np.zeros((3,3))
        dx[:,0] = p1 - p0
        dx[:,1] = p2 - p0
        dx[:,2] = p3 - p0
        
        # 记录体积
        volumet[ie,time] = np.linalg.det(dx) * 0.1666667
        # deformation gradient 变形梯度
        F = np.dot(dx,element_minv[ie])
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
        H = - element_volume[ie] * np.dot(piola, element_minv[ie].T)
        gradC = np.zeros((4,3))
        gradC[1,:] = H[:,0]
        gradC[2,:] = H[:,1]
        gradC[3,:] = H[:,2]
        gradC[0,:] = - H[:,0] - H[:,1] - H[:,2]
        
        node_force[element_idx[ie,0]] += gradC[0,:]
        node_force[element_idx[ie,1]] += gradC[1,:]
        node_force[element_idx[ie,2]] += gradC[2,:]
        node_force[element_idx[ie,3]] += gradC[3,:]
        
        sumGradC = 0
        for i in range(4):
            for j in range(3):
                sumGradC += invMass*(gradC[i,j]*gradC[i,j])
        if sumGradC < 1e-10:
            continue
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
            dF[i] = np.dot(dD[i],element_minv[ie])
            dE[i] = (np.dot(dF[i,:,:].T,F) + np.dot(F.T,dF[i,:,:]))*0.5
            trdE = dE[i,0,0] + dE[i,1,1] + dE[i,2,2]
            dP[i] = np.dot(dF[i],2*mu*E + la*trE*np.identity(3))
            dP[i] += np.dot(F,2*mu*dE[i] + la*trdE*np.identity(3))
            dH[i] = - element_volume[ie] * np.dot(dP[i],element_minv[ie].T)
            
        for n in range(4):
            nidx = element_idx[ie,n]
            for d in range(3):
                kidx = nidx * 3 + d
                didx = n * 3 + d
                
                idx = element_idx[ie,1] * 3 + 0
                Kmat[idx,kidx] += dH[didx,0,0]
                idx = element_idx[ie,1] * 3 + 1
                Kmat[idx,kidx] += dH[didx,1,0]
                idx = element_idx[ie,1] * 3 + 2
                Kmat[idx,kidx] += dH[didx,2,0]
                
                idx = element_idx[ie,2] * 3 + 0
                Kmat[idx,kidx] += dH[didx,0,1]
                idx = element_idx[ie,2] * 3 + 1
                Kmat[idx,kidx] += dH[didx,1,1]
                idx = element_idx[ie,2] * 3 + 2
                Kmat[idx,kidx] += dH[didx,2,1]
                
                idx = element_idx[ie,3] * 3 + 0
                Kmat[idx,kidx] += dH[didx,0,2]
                idx = element_idx[ie,3] * 3 + 1
                Kmat[idx,kidx] += dH[didx,1,2]
                idx = element_idx[ie,3] * 3 + 2
                Kmat[idx,kidx] += dH[didx,2,2]
                
                idx = element_idx[ie,0] * 3 + 0
                Kmat[idx,kidx] += - dH[didx,0,0] - dH[didx,0,1] - dH[didx,0,2]
                idx = element_idx[ie,0] * 3 + 1
                Kmat[idx,kidx] += - dH[didx,1,0] - dH[didx,1,1] - dH[didx,1,2]
                idx = element_idx[ie,0] * 3 + 2
                Kmat[idx,kidx] += - dH[didx,2,0] - dH[didx,2,1] - dH[didx,2,2]
                
        check = 1
    check = 1
    for i in range(krow):
        for j in range(krow):
            if i == j:
                Amat[i,j] = 1 - Kmat[i,j]*dt*dt/mass
            else:
                Amat[i,j] = - Kmat[i,j]*dt*dt/mass
    for i in range(node_num):
        bvec[i*3 + 0] = node_vel[i,0] + dt * node_force[i,0] / mass
        bvec[i*3 + 1] = node_vel[i,1] + dt * node_force[i,1] / mass
        bvec[i*3 + 2] = node_vel[i,2] + dt * node_force[i,2] / mass
    xvec = np.dot(np.linalg.inv(Amat),bvec)
    for i in range(node_num):
        node_vel[i,:] = node_force[i,:] * dt
        if(True):
            node_vel[i,0] = xvec[i * 3 + 0]
            node_vel[i,1] = xvec[i * 3 + 1]
            node_vel[i,2] = xvec[i * 3 + 2]
        xvect[i*3+0,time-1] = xvec[i * 3 + 0]
        xvect[i*3+1,time-1] = xvec[i * 3 + 1]
        xvect[i*3+2,time-1] = xvec[i * 3 + 2]
        
        node_pos[i,:] += node_vel[i,:] * dt
        
        post[i*3+0,time-1] = node_pos[i,0]
        post[i*3+1,time-1] = node_pos[i,1]
        post[i*3+2,time-1] = node_pos[i,2]
    
    check = 1
    time += 1