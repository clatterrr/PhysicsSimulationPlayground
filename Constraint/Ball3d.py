import numpy as np





hingeAxis_body1 = np.array([0,0,1])
hingeAxis_body2 = np.array([0,0,1])

def cross(veca,vecb):
    return np.array([veca[1]*vecb[2] - veca[2]*vecb[1],
                     veca[2]*vecb[0] - veca[0]*vecb[2],
                     veca[0]*vecb[1] - veca[1]*vecb[0]])

def getRotationMatrix(veca,vecb):
    v = cross(veca,vecb)
    s = np.linalg.norm(v)
    c = veca[0]*vecb[0] + veca[1]*vecb[1] + veca[2]*vecb[2]
    vx = np.array([[0,-v[2],v[1]],
                   [v[2],0,-v[0]],
                   [-v[1],v[0],0]])
    return np.identity(3) + vx + np.dot(vx,vx) / (1 + c)
    
# veca = np.array([1,0,0])
# vecb = np.array([0.707,0.707,0])
# R = getRotationMatrix(veca, vecb)
# veca = np.dot(R,veca)




def getNormalizedPerpendicuar(mf):
    if abs(mf[0]) > abs(mf[1]):
        leng = np.sqrt(mf[0]*mf[0] + mf[2]*mf[2])
        return np.array([mf[2],0,-mf[0]])/leng
    else:
        leng = np.sqrt(mf[1]*mf[1] + mf[2]*mf[2])
        return np.array([0,mf[2],-mf[1]])/leng
    
length_body = 2
height_body = 2
width_body = 2

def getInverseInertiaBox():
    Inertia = np.zeros((3,3))
    Inertia[0,0] = 12 / (height_body * height_body + width_body * width_body)
    Inertia[1,1] = 12 / (length_body * length_body + width_body * width_body)
    Inertia[2,2] = 12 / (height_body * height_body + length_body * length_body)
    return Inertia
    
    
mb2 = getNormalizedPerpendicuar(hingeAxis_body2)
mc2 = cross(hingeAxis_body1,mb2)


Jtrans = np.zeros((3,12))
Jtrans[:,0:3] = - np.identity(3)
Jtrans[:,6:9] =  np.identity(3)

Jtrans[0,4] = - hingeAxis_body1[2]
Jtrans[0,5] = hingeAxis_body1[1]
Jtrans[0,10] = - hingeAxis_body2[2]
Jtrans[0,11] = hingeAxis_body2[1]

Jtrans[1,3] = hingeAxis_body1[2]
Jtrans[1,5] = - hingeAxis_body1[0]
Jtrans[1,9] = hingeAxis_body2[2]
Jtrans[1,11] = - hingeAxis_body2[0]

Jtrans[2,3] = - hingeAxis_body1[1]
Jtrans[2,4] = hingeAxis_body1[0]
Jtrans[2,9] = - hingeAxis_body2[1]
Jtrans[2,10] = hingeAxis_body2[0]
# Jv 的 v
# Ball

step = 100
vel_body1 = np.array([0,0,0],dtype = float)
vel_body2 = np.array([0,0,0],dtype = float)
ang_body1 = np.array([0,0,0],dtype = float)
ang_body2 = np.array([0,0,0],dtype = float)

v = np.concatenate((vel_body1,ang_body1,vel_body2,ang_body2),0)
mass_body1 = 0.19
mass_body2 = 0.19


r_body1 = np.array([1,0,0])
r_body2 = np.array([-1,0,0])
len_body1 = 2
len_body2 = 2

pos_body1 = np.array([0,0,0])
pos_body2 = np.array([5,5,0])
r_body1_rest = np.array([1,0,0])
r_body2_rest = np.array([-1,0,0])

pos_body1_record = np.zeros((step,3))
pos_body2_record = np.zeros((step,3))

rot_body1 = getRotationMatrix(r_body1_rest, r_body1)
rot_body2 = getRotationMatrix(r_body2_rest, r_body2)
dt = 2
for i in range(step):
    pos_body1_record[i,:] = pos_body1[:]
    pos_body2_record[i,:] = pos_body2[:]
    
    InvInertia = getInverseInertiaBox() # 逆转动惯量
    mInvI1 = np.dot(rot_body1,np.dot(InvInertia,rot_body1)) # 矩阵 M 的一部分
    mInvI2 = np.dot(rot_body2,np.dot(InvInertia,rot_body2)) # 矩阵 M 的一部分
    Ktrans = np.identity(3) / mass_body1 + np.identity(3) / mass_body2 + mInvI1 + mInvI2 # 计算矩阵K = J M^-1 J^T
    connector_body1 = np.dot(rot_body1,r_body1 * len_body1) + pos_body1 # 对于物体一来说的连接处
    connector_body2 = np.dot(rot_body2,r_body2 * len_body2) + pos_body2 # 对于物体二来说的连接处
    lamb = - np.dot(np.linalg.inv(Ktrans),connector_body1 - connector_body2) # 拉格朗日乘子 K lamba = Jv
    corr_vel_body1 = np.dot(rot_body1,lamb) / mass_body1 * dt
    corr_vel_body2 = - np.dot(rot_body2,lamb) / mass_body2 * dt
    pos_body1 = pos_body1 + corr_vel_body1
    pos_body2 = pos_body2 + corr_vel_body2



