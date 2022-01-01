import numpy as np

node_num = 100
elem_num = node_num - 1
node_pos = np.zeros((node_num,3))
for i in range(node_num):
    theta = i / 100 * 360 / 180 * np.pi
    node_pos[i,0] = i
    node_pos[i,1] = 0
    node_pos[i,2] = i
    
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