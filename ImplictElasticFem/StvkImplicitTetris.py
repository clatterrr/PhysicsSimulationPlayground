'''
三维隐式弹性有限元
'''
import numpy as np

file_name = "tetrisCube.1"
tetNum = 2

file_f = open("E:\\vspro\\ImplicitCreeper\\Assets\\TetrisModel\\" + file_name + ".node","r")
lines = file_f.readlines()
node_num = int(lines[0].split()[0])
node_pos = np.zeros((node_num * tetNum,3))
node_vel = np.zeros((node_num * tetNum,3))
node_force = np.zeros((node_num * tetNum,3))
cnt = -1
for line in lines:
    if cnt == -1:
        cnt = 0
        continue
    x = line.split()
    node_pos[cnt,0] = float(x[1])
    node_pos[cnt,1] = float(x[2])
    node_pos[cnt,2] = float(x[3])
    for te in range(1,tetNum):
        node_pos[te * node_num, 0] = node_pos[cnt,0]
        node_pos[te * node_num, 1] = node_pos[cnt,1]
        node_pos[te * node_num, 2] = node_pos[cnt,2]
    cnt += 1
    if cnt == node_num:
        break

file_f = open("E:\\vspro\\ImplicitCreeper\\Assets\\TetrisModel\\" + file_name + ".ele","r")
lines = file_f.readlines()
elem_num = int(lines[0].split()[0])
elem_idx = np.zeros((elem_num * tetNum,4),dtype=int)
elem_minv = np.zeros((elem_num * tetNum ,3,3))
elem_volume = np.zeros((elem_num * tetNum))
cnt = -1
for line in lines:
    if cnt == -1:
        cnt = 0
        continue
    x = line.split()
    elem_idx[cnt,0] = int(x[1])
    elem_idx[cnt,1] = int(x[2])
    elem_idx[cnt,2] = int(x[3])
    elem_idx[cnt,3] = int(x[4])
    for te in range(1,tetNum):
        elem_idx[te * node_num, 0] = elem_idx[cnt,0]
        elem_idx[te * node_num, 1] = elem_idx[cnt,1]
        elem_idx[te * node_num, 2] = elem_idx[cnt,2]
        elem_idx[te * node_num, 3] = elem_idx[cnt,3]
    cnt += 1
    if cnt == elem_num:
        break
file_f.close()


for i in range(elem_num):
    p0 = node_pos[elem_idx[i,0]]
    p1 = node_pos[elem_idx[i,1]]
    p2 = node_pos[elem_idx[i,2]]
    p3 = node_pos[elem_idx[i,3]]
    dX = np.zeros((3,3))
    dX[:,0] = p1 - p0
    dX[:,1] = p2 - p0
    dX[:,2] = p3 - p0
    elem_volume[i] = np.linalg.det(dX) * 0.16666667
    elem_minv[i] = np.linalg.inv(dX)
   
        
    


mass = 1
dt = 1
invMass = 1 / mass
time = 0
timeFinal = 100

krow = node_num * 3
Kmat = np.zeros((krow,krow))
Amat = np.zeros((krow,krow))
bvec = np.zeros((krow))
xvec = np.zeros((krow))
resvec = np.zeros((krow))
dvec = np.zeros((krow))
Advec = np.zeros((krow))

volumet = np.zeros((elem_num,timeFinal))
xvect = np.zeros((krow,timeFinal))
post = np.zeros((krow,timeFinal))

while(time < timeFinal):
    node_force[:,:] = 0
    Kmat[:,:] = 0
    for ie in range(elem_num):
        p0 = node_pos[elem_idx[ie,0]]
        p1 = node_pos[elem_idx[ie,1]]
        p2 = node_pos[elem_idx[ie,2]]
        p3 = node_pos[elem_idx[ie,3]]
        dx = np.zeros((3,3))
        dx[:,0] = p1 - p0
        dx[:,1] = p2 - p0
        dx[:,2] = p3 - p0
        
        # 记录体积
        volumet[ie,time] = np.linalg.det(dx) * 0.1666667
        # deformation gradient 变形梯度
        F = np.dot(dx,elem_minv[ie])
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
        H = - elem_volume[ie] * np.dot(piola, elem_minv[ie].T)
        gradC = np.zeros((4,3))
        gradC[1,:] = H[:,0]
        gradC[2,:] = H[:,1]
        gradC[3,:] = H[:,2]
        gradC[0,:] = - H[:,0] - H[:,1] - H[:,2]
        
        node_force[elem_idx[ie,0]] += gradC[0,:]
        node_force[elem_idx[ie,1]] += gradC[1,:]
        node_force[elem_idx[ie,2]] += gradC[2,:]
        node_force[elem_idx[ie,3]] += gradC[3,:]
        
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
            dF[i] = np.dot(dD[i],elem_minv[ie])
            dE[i] = (np.dot(dF[i,:,:].T,F) + np.dot(F.T,dF[i,:,:]))*0.5
            trdE = dE[i,0,0] + dE[i,1,1] + dE[i,2,2]
            dP[i] = np.dot(dF[i],2*mu*E + la*trE*np.identity(3))
            dP[i] += np.dot(F,2*mu*dE[i] + la*trdE*np.identity(3))
            dH[i] = - elem_volume[ie] * np.dot(dP[i],elem_minv[ie].T)
            
        for n in range(4):
            nidx = elem_idx[ie,n]
            for d in range(3):
                kidx = nidx * 3 + d
                didx = n * 3 + d
                
                idx = elem_idx[ie,1] * 3 + 0
                Kmat[idx,kidx] += dH[didx,0,0]
                idx = elem_idx[ie,1] * 3 + 1
                Kmat[idx,kidx] += dH[didx,1,0]
                idx = elem_idx[ie,1] * 3 + 2
                Kmat[idx,kidx] += dH[didx,2,0]
                
                idx = elem_idx[ie,2] * 3 + 0
                Kmat[idx,kidx] += dH[didx,0,1]
                idx = elem_idx[ie,2] * 3 + 1
                Kmat[idx,kidx] += dH[didx,1,1]
                idx = elem_idx[ie,2] * 3 + 2
                Kmat[idx,kidx] += dH[didx,2,1]
                
                idx = elem_idx[ie,3] * 3 + 0
                Kmat[idx,kidx] += dH[didx,0,2]
                idx = elem_idx[ie,3] * 3 + 1
                Kmat[idx,kidx] += dH[didx,1,2]
                idx = elem_idx[ie,3] * 3 + 2
                Kmat[idx,kidx] += dH[didx,2,2]
                
                idx = elem_idx[ie,0] * 3 + 0
                Kmat[idx,kidx] += - dH[didx,0,0] - dH[didx,0,1] - dH[didx,0,2]
                idx = elem_idx[ie,0] * 3 + 1
                Kmat[idx,kidx] += - dH[didx,1,0] - dH[didx,1,1] - dH[didx,1,2]
                idx = elem_idx[ie,0] * 3 + 2
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
        bvec[i*3 + 0] = node_vel[i,0]
        xvec[i*3 + 1] = node_vel[i,1]
        xvec[i*3 + 2] = node_vel[i,2]
        bvec[i*3 + 0] = node_vel[i,0] + dt * node_force[i,0] / mass
        bvec[i*3 + 1] = node_vel[i,1] + dt * node_force[i,1] / mass
        bvec[i*3 + 2] = node_vel[i,2] + dt * node_force[i,2] / mass
    # 库函数求解
    # xvec = np.dot(np.linalg.inv(Amat),bvec)
    # 用咱自己的共轭梯度法
    rho1 = 1
    rho = 0
    for i in range(krow):
        Ax = 0
        for j in range(krow):
            Ax += Amat[i,j] * xvec[j]
        resvec[i] = bvec[i] - Ax
        rho += resvec[i] * resvec[i]
    cgit_max = 100
    for cgit in range(cgit_max):
        rho = np.dot(resvec.T,resvec)
        if rho < 1e-10:
            break
        beta = 0
        if time > 0:
            beta = rho / rho1
        rho1 = rho
        dvec = beta * dvec + resvec
        Advec = np.dot(Amat,dvec)
        AdA = np.dot(dvec.T,Advec)
        alpha = rho1 / AdA
        xvec = xvec + alpha * dvec
        resvec = resvec - alpha * Advec
        
    xxvec = np.dot(np.linalg.inv(Amat),bvec)  
    # 再把结果加上去
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