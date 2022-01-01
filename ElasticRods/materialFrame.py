import numpy as np

node_num = 100
elem_num = node_num - 1
node_pos = np.zeros((node_num,3))
for i in range(node_num):
    theta = i / node_num * 360 / 180 * np.pi
    node_pos[i,0] = np.cos(theta)
    node_pos[i,1] = 0
    node_pos[i,2] = np.sin(theta)
    
tangent = np.zeros((elem_num,3))
axis  = np.zeros((elem_num,3))
bishop_t  = np.zeros((elem_num,3))
bishop_u = np.zeros((elem_num,3))
bishop_v = np.zeros((elem_num,3))

for i in range(elem_num):
    tangent[i] = node_pos[i+1] - node_pos[i]
    tangent[i] /= np.linalg.norm(tangent[i])
    bishop_t[i] = tangent[i]
    
bishop_u[0] = np.array([0,1,0])
bishop_v[0] = np.cross(bishop_t[0], bishop_u[0])

def quaternion_mult(q,r):
    return np.array([r[0]*q[0]-r[1]*q[1]-r[2]*q[2]-r[3]*q[3],
            r[0]*q[1]+r[1]*q[0]-r[2]*q[3]+r[3]*q[2],
            r[0]*q[2]+r[1]*q[3]+r[2]*q[0]-r[3]*q[1],
            r[0]*q[3]-r[1]*q[2]+r[2]*q[1]+r[3]*q[0]])
# parallel transport
for i in range(1,elem_num):
    axis[i] = np.cross(bishop_t[i-1], bishop_t[i]) / (1 + np.dot(bishop_t[i-1], bishop_t[i]))
    mag = np.dot(axis[i],axis[i])
    cosphi = np.sqrt(4 / (4 + mag))
    sinphi = np.sqrt(mag / (4 + mag))
    # 四元数旋转
    r = np.array([0,bishop_u[i-1,0],bishop_u[i-1,1],bishop_u[i-1,2]])
    q = np.array([cosphi,sinphi*axis[i,0],
                     sinphi*axis[i,1],sinphi*axis[i,2]])
    q_conj = np.array([q[0],-1*q[1],-1*q[2],-1*q[3]])
    bishop_u[i] = quaternion_mult(quaternion_mult(q,r),q_conj)[1:]
    bishop_v[i] = np.cross(bishop_t[i], bishop_u[i])
    
theta = np.zeros((elem_num))
matFrme_t  = np.zeros((elem_num,3))
matFrame_m1 = np.zeros((elem_num,3))
matFrame_m2 = np.zeros((elem_num,3))

for i in range(elem_num):
    matFrme_t[i] = bishop_t[i]
    cosTheta = np.cos(theta[i])
    sinTheta = np.sin(theta[i])
    matFrame_m1[i] = cosTheta * bishop_u[i] + sinTheta * bishop_v[i]
    matFrame_m2[i] = - sinTheta * bishop_u[i] + cosTheta * bishop_v[i]
    

kb = np.zeros((elem_num,3))
kappa_rest = np.zeros((elem_num,2))
curvature_method1 = np.zeros((elem_num))
curvature_method2 = np.zeros((elem_num))
omega_rest = np.zeros((node_num,2))
for i in range(1,elem_num):
    kb[i] = 2 * np.cross(bishop_t[i-1], bishop_t[i]) / (1 + np.dot(bishop_t[i-1], bishop_t[i]))
    curvature_method1[i] = 2 * np.linalg.norm(np.cross(bishop_t[i-1], bishop_t[i])) / (1 + np.dot(bishop_t[i-1], bishop_t[i]))
    kappa_rest[i,0] = 0.5 * np.dot((matFrame_m2[i] + matFrame_m2[i]),kb[i])
    kappa_rest[i,1] = - 0.5 * np.dot((matFrame_m1[i] + matFrame_m1[i]),kb[i])
    curvature_method2[i] = np.sqrt(kappa_rest[i,0] ** 2 + kappa_rest[i,1] ** 2)
    omega_rest[i,0] = np.dot(kb[i],matFrame_m2[i])
    omega_rest[i,1] = - np.dot(kb[i],matFrame_m1[i])
    
time = 0
timeFinal = 100
kappa = np.zeros((elem_num,2))
omega = np.zeros((elem_num,2))

bendingEnergy = np.zeros((elem_num))
stretchEnergy = np.zeros((elem_num))
twistEnergy = np.zeros((elem_num))
bendingForce = np.zeros((elem_num,3))
stretchForce = np.zeros((elem_num,3))
twistForce = np.zeros((elem_num),3)
bendingHessian = np.zeros((elem_num,3))
stretchHessian = np.zeros((elem_num,3))
twistHessian = np.zeros((elem_num),3)
while time < timeFinal:
    
    # update bishop frame
    for i in range(1,elem_num):
        tangent[i] = node_pos[i+1] - node_pos[i]
        tangent[i] /= np.linalg.norm(tangent[i])
        bishop_t[i] = tangent[i]
    # parallel transport
    for i in range(1,elem_num):
        axis[i] = np.cross(bishop_t[i-1], bishop_t[i]) / (1 + np.dot(bishop_t[i-1], bishop_t[i]))
        mag = np.dot(axis[i],axis[i])
        cosphi = np.sqrt(4 / (4 + mag))
        sinphi = np.sqrt(mag / (4 + mag))
        # 四元数旋转
        r = np.array([0,bishop_u[i-1,0],bishop_u[i-1,1],bishop_u[i-1,2]])
        q = np.array([cosphi,sinphi*axis[i,0],
                         sinphi*axis[i,1],sinphi*axis[i,2]])
        q_conj = np.array([q[0],-1*q[1],-1*q[2],-1*q[3]])
        bishop_u[i] = quaternion_mult(quaternion_mult(q,r),q_conj)[1:]
        bishop_v[i] = np.cross(bishop_t[i], bishop_u[i])
    # update material frame
    for i in range(elem_num):
        matFrme_t[i] = bishop_t[i]
        cosTheta = np.cos(theta[i])
        sinTheta = np.sin(theta[i])
        matFrame_m1[i] = cosTheta * bishop_u[i] + sinTheta * bishop_v[i]
        matFrame_m2[i] = - sinTheta * bishop_u[i] + cosTheta * bishop_v[i]
    # bending energy 
    for i in range(1,elem_num):
        kb[i] = 2 * np.cross(bishop_t[i-1], bishop_t[i]) / (1 + np.dot(bishop_t[i-1], bishop_t[i]))
        kappa[i,0] = 0.5 * np.dot((matFrame_m2[i] + matFrame_m2[i]),kb[i])
        kappa[i,1] = - 0.5 * np.dot((matFrame_m1[i] + matFrame_m1[i]),kb[i])
        omega[i,0] = np.dot(kb[i],matFrame_m2[i])
        omega[i,1] = - np.dot(kb[i],matFrame_m1[i])
    for i in range(elem_num):
        bendingEnergy[i] = 0.5
    time += 1
