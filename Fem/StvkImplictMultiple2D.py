import numpy as np

node_row_num = 4
node_num = node_row_num * node_row_num
element_row_num = node_row_num - 1
element_num = element_row_num * element_row_num * 2

element_idx = np.zeros((element_num,3),dtype = int)
element_minv = np.zeros((element_num,2,2))

node_pos = np.zeros((node_num,2),dtype = float)
node_vel = np.zeros((node_num,2),dtype = float)
node_force = np.zeros((node_num,2),dtype = float)

for j in range(node_row_num):
    for i in range(node_row_num):
        node_pos[j*node_row_num+i] = np.array([i,j])
cnt = 0
for j in range(element_row_num):
    for i in range(element_row_num):
        idx = j * node_row_num + i
        element_idx[cnt,0] = idx
        element_idx[cnt,1] = idx + 1
        element_idx[cnt,2] = idx + node_row_num
        cnt += 1
        element_idx[cnt,0] = idx + 1
        element_idx[cnt,1] = idx + node_row_num + 1
        element_idx[cnt,2] = idx + node_row_num
        cnt += 1

for ie in range(element_num):
    p0 = node_pos[element_idx[ie,0]]
    p1 = node_pos[element_idx[ie,1]]
    p2 = node_pos[element_idx[ie,2]]
    dX = np.array([[p1[0] - p0[0],p2[0] - p0[0]],
                  [ p1[1] - p0[1],p2[1] - p0[1]]])
    element_minv[ie] = np.linalg.inv(dX)

krow = node_num * 2
Kmat = np.zeros((krow,krow))
Amat = np.zeros((krow,krow))
bvec = np.zeros((krow))


time = 0
timeFinal = 100
areat = np.zeros((element_num,timeFinal+1))
xvect = np.zeros((krow,timeFinal))
post = np.zeros((krow,timeFinal))

mass = 10
dt = 1
node_pos[node_row_num-1,0] += 1

while(time < timeFinal):
    time += 1
    node_force[:,:] = 0
    for ie in range(element_num):
        p0 = node_pos[element_idx[ie,0]]
        p1 = node_pos[element_idx[ie,1]]
        p2 = node_pos[element_idx[ie,2]]
        dx = np.array([[p1[0] - p0[0],p2[0] - p0[0]],
                      [ p1[1] - p0[1],p2[1] - p0[1]]])
        # 面积
        areat[ie,time-1] = np.linalg.det(dx)*0.5
        # 形变梯度
        F = np.dot(dx,element_minv[ie])
        # Green Strain，也就是E
        E = (np.dot(F.T,F)- np.identity(2)) * 0.5
        # lame常数
        mu = 2
        # lame常数
        la = 2
        # 应力
        piola = np.dot(F, 2 * mu * E + la * (E[0,0] + E[1,1]) * np.identity(2))
        
        area = 0.5
        doubleInner = E[0,0]*E[0,0] + E[1,0]*E[1,0] + E[0,1]*E[0,1] + E[1,1]*E[1,1]
        energy = doubleInner * mu + la / 2 * (E[0,0] + E[1,1])**2
        
        H = - area * np.dot(piola,element_minv[ie].transpose())
        
        gradC0 = np.array([H[0,0],H[1,0]])
        gradC1 = np.array([H[0,1],H[1,1]])
        gradC2 = - gradC0 - gradC1
        
        node_force[element_idx[ie,0],:] += gradC2
        node_force[element_idx[ie,1],:] += gradC0
        node_force[element_idx[ie,2],:] += gradC1
        
        dD = np.zeros((6,2,2))
        dD[0,:,:] = np.array([[-1,-1],[0,0]])
        dD[1,:,:] = np.array([[0,0],[-1,-1]])
        dD[2,:,:] = np.array([[1,0],[0,0]])
        dD[3,:,:] = np.array([[0,0],[1,0]])
        dD[4,:,:] = np.array([[0,1],[0,0]])
        dD[5,:,:] = np.array([[0,0],[0,1]])
        # 变形梯度求导
        dF = np.zeros((6,2,2))
        # Green 应变求导
        dE = np.zeros((6,2,2))
        # piola 求导
        dP = np.zeros((6,2,2))
        # Hessian 求导
        dH = np.zeros((6,2,2))
        for i in range(6):
            dF[i,:,:] = np.dot(dD[i,:,:],element_minv[ie]) 
            d_F = dF[i,:,:]
            dE[i,:,:] = (np.dot(d_F.T,F) + np.dot(F.T,d_F))*0.5
            d_E = dE[i,:,:]
            dP[i,:,:] = np.dot(d_F,2 * mu * E + la * (E[0,0] + E[1,1]) * np.identity(2))
            dP[i,:,:] += np.dot(F,2 * mu * d_E + la * (d_E[0,0] + d_E[1,1]) * np.identity(2))
            dH[i,:,:] = - area * np.dot(dP[i,:,:],element_minv[ie].T)
        for n in range(3):
            nidx = element_idx[ie,n]
            for d in range(2):
                kidx = nidx * 2 + d
                didx = n * 2 + d
                idx = element_idx[ie,1] * 2
                Kmat[idx,kidx] += dH[didx,0,0]
                idx = element_idx[ie,1] * 2 + 1
                Kmat[idx,kidx] += dH[didx,1,0]
                idx = element_idx[ie,2] * 2
                Kmat[idx,kidx] += dH[didx,0,1]
                idx = element_idx[ie,2] * 2 + 1
                Kmat[idx,kidx] += dH[didx,1,1]
                idx = element_idx[ie,0] * 2
                Kmat[idx,kidx] += - dH[didx,0,0] - dH[didx,0,1]
                idx = element_idx[ie,0] * 2 + 1
                Kmat[idx,kidx] += - dH[didx,1,0] - dH[didx,1,1]
        
        check = 1
                
    for i in range(krow):
        for j in range(krow):
            if i == j:
                Amat[i,j] = mass - Kmat[i,j]*dt*dt
            else:
                Amat[i,j] = - Kmat[i,j]*dt*dt
    for i in range(node_num):
        bvec[i*2 + 0] = mass * node_vel[i,0] + dt * node_force[i,0]
        bvec[i*2 + 1] = mass * node_vel[i,1] + dt * node_force[i,1]
        
    xvec = np.dot(np.linalg.inv(Amat),bvec)
    for i in range(node_num):
        node_vel[i,0] = node_force[i,0] * dt
        node_vel[i,1] = node_force[i,1] * dt
        if(True):
            node_vel[i,0] = xvec[i * 2 + 0]
            node_vel[i,1] = xvec[i * 2 + 1]
        xvect[i*2+0,time-1] = xvec[i * 2 + 0]
        xvect[i*2+1,time-1] = xvec[i * 2 + 1]
        node_pos[i,:] += node_vel[i,:] * dt
        post[i*2+0,time-1] = node_pos[i,0]
        post[i*2+1,time-1] = node_pos[i,1]
    
for ie in range(element_num):
    p0 = node_pos[element_idx[ie,0]]
    p1 = node_pos[element_idx[ie,1]]
    p2 = node_pos[element_idx[ie,2]]
    dx = np.array([[p1[0] - p0[0],p2[0] - p0[0]],
                  [ p1[1] - p0[1],p2[1] - p0[1]]])
    # 面积
    areat[ie,time] = np.linalg.det(dx)*0.5