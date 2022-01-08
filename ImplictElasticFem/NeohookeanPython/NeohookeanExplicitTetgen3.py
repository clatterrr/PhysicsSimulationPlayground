'''
三维隐式弹性有限元
'''
import numpy as np

file_name = "cubeHigh.1"

file_f = open("E:\\vspro\\UnityProject\\Assets\\TetModel\\" + file_name + ".node","r")
lines = file_f.readlines()
node_num = int(lines[0].split()[0])
node_pos = np.zeros((node_num,3))
node_vel = np.zeros((node_num,3))
node_force = np.zeros((node_num,3))
cnt = -1
for line in lines:
    if cnt == -1:
        cnt = 0
        continue
    x = line.split()
    node_pos[cnt,0] = float(x[1])
    node_pos[cnt,1] = float(x[2])
    node_pos[cnt,2] = float(x[3])
    cnt += 1
    if cnt == node_num:
        break

file_f = open("E:\\vspro\\UnityProject\\Assets\\TetModel\\" + file_name + ".ele","r")
lines = file_f.readlines()
elem_num = int(lines[0].split()[0])
elem_idx = np.zeros((elem_num,4),dtype=int)
elem_minv = np.zeros((elem_num,3,3))
elem_volume = np.zeros((elem_num))
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


node_pos = node_pos * 0.3

mass = 0.1
dt = 0.05
invMass = 1 / mass
time = 0
timeFinal = 300

mu = 0.2
la = 0.2

volumet = np.zeros((elem_num,timeFinal))
xvect = np.zeros((node_num,3,timeFinal))
post = np.zeros((node_num,3,timeFinal))

while(time < timeFinal):
    node_force[:,:] = 0
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
        C = np.dot(F.T,F)
        Cinv = np.linalg.inv(C)
        J = max(np.linalg.det(F),0.01)
        logJ = np.log(J)
        # 第一不变量
        Ic = C[0,0] + C[1,1] + C[1,1]
        # 可压缩 neohookean 能量
        energy = mu * 0.5 * (Ic - 3) - mu * logJ + la * 0.5 * logJ * logJ
        # 第二 piola kirchhoff 应力
        piola = mu * (np.identity(3) - Cinv) + la * logJ * Cinv
        # 面积
        H = - elem_volume[ie] * np.dot(piola, elem_minv[ie].T)
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
            continue
        node_force[elem_idx[ie,0]] += gradC[0,:] 
        node_force[elem_idx[ie,1]] += gradC[1,:]   
        node_force[elem_idx[ie,2]] += gradC[2,:]  
        node_force[elem_idx[ie,3]] += gradC[3,:]   
    for i in range(node_num):
        node_pos[i] += dt * node_force[i]
    post[:,:,time] = node_pos
    time += 1
    
volumeError = np.zeros((elem_num))
for ie in range(elem_num):
    volumeError[ie] = abs(volumet[ie,timeFinal-1] - elem_volume[ie]) / elem_volume[ie]