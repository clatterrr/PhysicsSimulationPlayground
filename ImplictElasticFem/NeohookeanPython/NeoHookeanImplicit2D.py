import numpy as np

Top = True
# 初始化三角形初始位置
node_pos = np.array([[0,0],[1,0],[0,1]],dtype = float)
if Top == True:
    node_pos = np.array([[1,0],[1,1],[0,1]],dtype = float)
# 顶点的位置梯度
Ds = np.array([[node_pos[1,0] - node_pos[0,0],node_pos[2,0] -
node_pos[0,0]],
[node_pos[1,1] - node_pos[0,1],node_pos[2,1] -
node_pos[0,1]]])

young = 100
nu = 0.4
mu = young / ( 2 * (1 + nu))
la = young * nu / (1 + nu) / (1 - 2 * nu)


# 求逆，用于准备计算形变梯度
minv = np.linalg.inv(Ds)
# 假设某一时刻，三角形变化到了这样的位置
node_pos = np.array([[0,0],[100,0],[0,1]],dtype = float)
if Top == True:
    node_pos = np.array([[2,0],[1,1],[0,1]],dtype = float)
node_vel = np.zeros((3,2))
node_force = np.zeros((3,2))
time = 0
timeFinal = 100
areat = np.zeros((timeFinal))
while(time < timeFinal):

    
    node_force[:,:] = 0
    
    time += 1
    # 形变梯度中的分子
    Ds_new = np.array([[node_pos[1,0] - node_pos[0,0],node_pos[2,0]
    - node_pos[0,0]],
    [node_pos[1,1] - node_pos[0,1],node_pos[2,1] -
    node_pos[0,1]]])
    # 形变梯度
    F = np.dot(Ds_new,minv)
    Finv = np.linalg.inv(F)
    FinvT = Finv.T
    FtF = np.dot(F.T,F)
    
    J = np.linalg.det(F)
    
    logJ = np.log(J)
    # 第一不变量
    Ic = FtF[0,0] + FtF[1,1] 
    # 可压缩 neohookean 能量
    energy = mu * 0.5 * (Ic - 2) - mu * logJ + la * 0.5 * logJ * logJ
    # 第一 piola kirchhoff 应力
    piola = mu * F - mu * FinvT + la * logJ * 0.5 * FinvT
    area = 0.5
    H = - area * np.dot(piola,minv.transpose())
    
    gradC0 = np.array([H[0,0],H[1,0]])
    gradC1 = np.array([H[0,1],H[1,1]])
    gradC2 = - gradC0 - gradC1
    
    node_force[0,:] += gradC2
    node_force[1,:] += gradC0
    node_force[2,:] += gradC1
    
    dD = np.zeros((6,2,2))
    dD[0,:,:] = np.array([[-1,-1],[0,0]])
    dD[1,:,:] = np.array([[0,0],[-1,-1]])
    dD[2,:,:] = np.array([[1,0],[0,0]])
    dD[3,:,:] = np.array([[0,0],[1,0]])
    dD[4,:,:] = np.array([[0,1],[0,0]])
    dD[5,:,:] = np.array([[0,0],[0,1]])
    # 变形梯度求导
    dF = np.zeros((6,2,2))
    # piola 求导
    dP = np.zeros((6,2,2))
    # Hessian 求导
    dH = np.zeros((6,2,2))
    for i in range(6):
        dF[i] = np.dot(dD[i],minv) 
        dP[i] = mu * dF[i]
        dP[i] += (mu - la * logJ) * np.dot(np.dot(FinvT,dF[i].T),FinvT)
        term = np.dot(Finv,dF[i])
        dP[i] += la * (term[0,0] + term[1,1]) * FinvT
        dH[i] = - area * np.dot(dP[i],minv.T)
    
    K = np.zeros((6,6))
    # 3 个顶点
    for n in range(3):
        # 2 个维度
        for d in range(2):
            # 第 idx 列，每列3 x 2 个元素
            idx = n * 2 + d
            # 先填写第一第二个顶点，第零个顶点之后填
            K[2,idx] += dH[idx,0,0]
            K[3,idx] += dH[idx,1,0]
            K[4,idx] += dH[idx,0,1]
            K[5,idx] += dH[idx,1,1]
            
            K[0,idx] += - dH[idx,0,0] - dH[idx,0,1]
            K[1,idx] += - dH[idx,1,0] - dH[idx,1,1]
            
    mass = 0.1
    dt = 1
    A = mass * np.identity(6) -  K  * dt * dt
    b = np.zeros((6))
    for n in range(3):
        for d in range(2):
            b[n*2+d] = mass * node_vel[n,d] + dt * node_force[n,d]
            
    x = np.dot(np.linalg.inv(A), b)
    for n in range(3):
        for d in range(2):
            node_vel[n,d] = x[n*2+d]
            node_pos[n,d] += node_vel[n,d]*dt
            
    areat[time - 1] = 0.5 * (node_pos[0,0] * (node_pos[1,1] - node_pos[2,1])
                    + node_pos[1,0] * (node_pos[2,1] - node_pos[0,1]) 
                    + node_pos[2,0] * (node_pos[0,1] - node_pos[1,1]))
    
    