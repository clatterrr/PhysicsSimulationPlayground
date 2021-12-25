import numpy as np
# 初始化三角形初始位置
node_pos = np.array([[0,0],[1,0],[2,0],[3,0],
                     [0,1],[1,1],[2,1],[3,1],
                     [0,2],[1,2]],dtype = float)
node_num = 10
node_force = np.zeros((node_num,2))
elem_num = 8
elem_idx = np.array([[0,4,1],[1,4,5],
                     [1,5,2],[2,5,6],
                     [2,6,3],[3,6,7],
                     [4,8,5],[5,8,9]])
elem_minv = np.zeros((elem_num,2,2))
for ie in range(elem_num):
    p0x = node_pos[elem_idx[ie,0],0]
    p0y = node_pos[elem_idx[ie,0],1]
    p1x = node_pos[elem_idx[ie,1],0]
    p1y = node_pos[elem_idx[ie,1],1]
    p2x = node_pos[elem_idx[ie,2],0]
    p2y = node_pos[elem_idx[ie,2],1]
    dX = np.array([[p1x - p0x,p2x - p0x],
                   [p1y - p0y,p2y - p0y]])
    elem_minv[ie] = np.linalg.inv(dX)
    
time = 0
dt = 0.1
timeFinal = 1000
areat = np.zeros((elem_num,timeFinal))
node_pos_t = np.zeros((timeFinal,node_num,2))
piola_el_t = np.zeros((timeFinal,elem_num,4))

def cross2d(vec0,vec1):
    return vec0[0]*vec1[1] - vec0[1]*vec1[0]

while(time < timeFinal):
    time += 1
    node_force[:,:] = 0
    for ie in range(elem_num):
        
        # 形变梯度中的分子
        p0x = node_pos[elem_idx[ie,0],0]
        p0y = node_pos[elem_idx[ie,0],1]
        p1x = node_pos[elem_idx[ie,1],0]
        p1y = node_pos[elem_idx[ie,1],1]
        p2x = node_pos[elem_idx[ie,2],0]
        p2y = node_pos[elem_idx[ie,2],1]
        dx = np.array([[p1x - p0x,p2x - p0x],
                       [p1y - p0y,p2y - p0y]])
        
        areat[ie,time - 1] = 0.5 * abs(np.linalg.det(dx))
        # 形变梯度
        F = np.dot(dx,elem_minv[ie])
        
        E = (np.dot(F.T,F)- np.identity(2)) * 0.5
        # lame 常数
        mu = 100
        # lame 常数
        la = 100
        # 双点积
        doubleInner = 0
        for i in range(2):
            for j in range(2):
                doubleInner += E[i,j]*E[i,j]
        # trace 迹
        trE = E[0,0] + E[1,1]
        # 能量
        energy = doubleInner * mu + la * 0.5 * trE * trE
        # first piola kirchhoff stress
        piola = np.dot(F,2*mu*E + la * trE * np.identity(2))

        
        # 三角形面积
        area = 0.5
        # 计算力
        H = - area * np.dot(piola,elem_minv[ie].T)
        dforce = np.zeros((3,2))
        dforce[1,:] = np.array([H[0,0],H[1,0]])
        dforce[2,:]= np.array([H[0,1],H[1,1]])
        dforce[0,:] = - dforce[1,:] - dforce[2,:]
        
        piola_el_t[time-1,ie,0] = H[0,0]
        piola_el_t[time-1,ie,1] = H[0,1]
        piola_el_t[time-1,ie,2] = H[1,0]
        piola_el_t[time-1,ie,3] = H[1,1]
        invMass = 0.1
        
        sumdforce = 0
        for i in range(3):
            for j in range(2):
                sumdforce += invMass*dforce[i,j] * dforce[i,j]
        dforce[:,1] -= 1
        # 判断是否收敛
        if sumdforce < 1e-10:
            continue
        # 校正位置，方法来源于PositionBasedDynamics
        node_force[elem_idx[ie,0],:] += energy * dforce[0,:] / sumdforce * invMass
        node_force[elem_idx[ie,1],:] += energy * dforce[1,:] / sumdforce * invMass
        node_force[elem_idx[ie,2],:] += energy * dforce[2,:] / sumdforce * invMass
    gravity = np.array([0,-0.2])

    for ip in range(node_num):
        x0 = node_pos[ip,:]
        x1 = np.array([-1,0],dtype = float)
        x2 = np.array([5,0],dtype =float)
        v0 = node_force[ip,:] + gravity
        v1 = np.zeros((2))
        v2 = np.zeros((2))
        a = cross2d(v0-v1, v2-v1)
        b = cross2d(x0-x1, v2-v1) + cross2d(v0-v1, x2-x1)
        c = cross2d(x0-x1, x2-x1)
        
        d = b*b - 4*a*c
        result = np.zeros((2))
        result_num = 0
        if d < 0:
            result_num = 1
            result[1] = - b / (2 * a)
        else:
            q = - (b + np.sign(b)*np.sqrt(d)) / 2
            if (abs(a) > 1e-12*abs(q)):
                result[result_num] = q / a
                result_num += 1
            if (abs(q) > 1e-12*abs(c)):
                result[result_num] = c / q
                result_num += 1
            if result_num == 2 and result[0] > result[1]:
                temp = result[0]
                result[0] = result[1]
                result[1] = temp
                
        collision = False
        for i in range(result_num):
            t = result[i]
            if t >= 0 and t < dt:
                x0new = x0 + t * v0
                x1new = x1 + t * v1
                x2new = x2 + t * v2
                res = cross2d(x0new - x1new, x2new - x1new)
                if abs(res) < 1e-10:
                    collision = True
        if collision == True:
            check = 1
        else:
            node_pos[ip,:] += (node_force[ip,:] + gravity)*dt
        node_pos_t[time-1,ip,:] = node_pos[ip,:]