import numpy as np


_gamma = 5.828427124 # FOUR_GAMMAt_sQUARED = sqrt(8)+3;
_cstar = 0.923879532 # cos(pi/8)
_sstar = 0.3826834323 # sin(pi/8)
EPSILON = 1e-6


def ApproxGivens(a11,a12,a22):
    ch = 2 * (a11 - a22)
    sh = a12
    w = 1.0 / np.sqrt(ch * ch + sh * sh)
    if _gamma * sh * sh < ch * ch:
        return w * ch, w * sh
    else:
        return _cstar,_sstar

def QRGivens(a1,a2):
    rho = np.sqrt(a1 * a1 + a2 * a2)
    sh = 0
    if rho > EPSILON:
        sh = a2
    ch = abs(a1) + max(rho,EPSILON)
    if a1 < 0:
        temp = sh
        sh = ch
        ch = temp
    w = 1.0 / np.sqrt(ch * ch + sh * sh)
    return ch * w, sh * w

def CondWap(c,x,y):
    z = x
    if c == True:
        return y,z
    else:
        return x,y
    
def CondNegWap(c,x,y):
    z = -x
    if c == True:
        return y,z
    else:
        return x,y

def quatToMat(qV):
    w = qV[3]
    x = qV[0]
    y = qV[1]
    z = qV[2]
    qxx = x * x
    qyy = y * y
    qzz = z * z
    qxz = x * z
    qxy = x * y
    qyz = y * z
    qwx = w * x
    qwy = w * y
    qwz = w * z
    return np.array([[1 - 2 * (qyy + qzz),2 * (qxy - qwz),2 * (qxz + qwy)],
                     [2 * (qxy + qwz), 1 - 2 * (qxx + qzz), 2 * (qyz - qwx)],
                     [2 * (qxz - qwy), 2 * (qyz + qwx), 1 - 2 * (qxx + qyy)]])
    
def jacobiConjugation(x,y,z,
                      s11,
                      s21,s22,
                      s31,s32,s33,qV):
    ch,sh = ApproxGivens(s11, s21, s22)
    scale = ch * ch + sh * sh
    a = (ch *ch - sh * sh) / scale
    b = 2 * sh * ch / scale
    
    t_s11 = s11
    t_s21 = s21
    t_s22 = s22
    t_s31 = s31
    t_s32 = s32
    t_s33 = s33
    
    s11 = a*(a*t_s11 + b*t_s21) + b*(a*t_s21 + b*t_s22)
    s21 = a*(-b*t_s11 + a*t_s21) + b*(-b*t_s21 + a*t_s22)
    s22 = -b*(-b*t_s11 + a*t_s21) + a*(-b*t_s21 + a*t_s22)
    s31 = a*t_s31 + b*t_s32
    s32 = -b*t_s31 + a*t_s32
    s33 = t_s33
    
    tmp = np.array([qV[0],qV[1],qV[2]]) * sh
    sh = qV[3] * sh
    
    qV *= ch
    qV[z] += sh
    qV[3] -= tmp[z]
    qV[x] += tmp[y]
    qV[y] -= tmp[x]
    
    return s22,s32,s33,s21,s31,s11,qV
    
# Cyclic Jacobi requires that we always 
# pick |θ| < π/4 to ensure convergence, an option that
# is always possible as illustrated next
def jacobiEigenAnlysis(s11,s21,s22,s31,s32,s33):
    qV = np.array([0,0,0,1],dtype = float)
    for i in range(10):
        S = np.array([[s11,s21,s31],[s21,s22,s32],[s31,s32,s33]])
        s11,s21,s22,s31,s32,s33,qV = jacobiConjugation(0,1,2,s11,s21,s22,s31,s32,s33,qV)
        S = np.array([[s11,s21,s31],[s21,s22,s32],[s31,s32,s33]])
        s11,s21,s22,s31,s32,s33,qV = jacobiConjugation(1,2,0,s11,s21,s22,s31,s32,s33,qV)
        S = np.array([[s11,s21,s31],[s21,s22,s32],[s31,s32,s33]])
        s11,s21,s22,s31,s32,s33,qV = jacobiConjugation(2,0,1,s11,s21,s22,s31,s32,s33,qV)
    return qV
    
def dist(x,y,z):
    return x * x + y * y + z * z

def sortSingluar(B,V):
    b11,b12,b13 = B[0,0],B[0,1],B[0,2]
    b21,b22,b23 = B[1,0],B[1,1],B[1,2]
    b31,b32,b33 = B[2,0],B[2,1],B[2,2]
    v11,v12,v13 = V[0,0],V[0,1],V[0,2]
    v21,v22,v23 = V[1,0],V[1,1],V[1,2]
    v31,v32,v33 = V[2,0],V[2,1],V[2,2]
    rho1 = dist(b11, b21, b31)
    rho2 = dist(b12, b22, b23)
    rho3 = dist(b13, b23, b33)
    c = rho1 < rho2
    b11,b12 = CondNegWap(c, b11, b12)
    b21,b22 = CondNegWap(c, b21, b22)
    b31,b32 = CondNegWap(c, b31, b32)
    v11,v12 = CondNegWap(c, v11, v12)
    v21,v22 = CondNegWap(c, v21, v22)
    v31,v32 = CondNegWap(c, v31, v32)
    rho1,rho2 = CondWap(c, rho1, rho2)
    
    c = rho1 < rho3
    b11,b13 = CondNegWap(c, b11, b13)
    b21,b23 = CondNegWap(c, b21, b23)
    b31,b33 = CondNegWap(c, b31, b33)
    v11,v13 = CondNegWap(c, v11, v13)
    v21,v23 = CondNegWap(c, v21, v23)
    v31,v33 = CondNegWap(c, v31, v33)
    rho1,rho3 = CondWap(c, rho1, rho3)
    
    c = rho2 < rho3
    b12,b13 = CondNegWap(c, b12, b13)
    b22,b23 = CondNegWap(c, b22, b23)
    b32,b33 = CondNegWap(c, b32, b33)
    v12,v13 = CondNegWap(c, v12, v13)
    v22,v23 = CondNegWap(c, v22, v23)
    v32,v33 = CondNegWap(c, v32, v33)
    
    B = np.array([[b11,b12,b13],[b21,b22,b23],[b31,b32,b33]])
    V = np.array([[v11,v12,v13],[v21,v22,v23],[v31,v32,v33]])
    return B,V
    
def QRdecomp(B):
    b11,b12,b13 = B[0,0],B[0,1],B[0,2]
    b21,b22,b23 = B[1,0],B[1,1],B[1,2]
    b31,b32,b33 = B[2,0],B[2,1],B[2,2]
    
    ch1,sh1 = QRGivens(b11,b21)
    a = 1 - 2 * sh1 * sh1
    b = 2 * ch1 * sh1
    r11 = a * b11 + b * b21
    r12 = a * b12 + b * b22
    r13 = a * b13 + b * b23
    r21 = -b * b11 + a * b21
    r22 = -b * b12 + a * b22
    r23 = -b * b13 + a * b23
    r31 = b31
    r32 = b32
    r33 = b33

    ch2,sh2 = QRGivens(r11,r31)
    a = 1 - 2 * sh2 * sh2
    b = 2 * ch2 * sh2
    b11 = a * r11 + b * r31
    b12 = a * r12 + b * r32
    b13 = a * r13 + b * r33
    b21 = r21
    b22 = r22
    b23 = r23;
    b31 = -b * r11 + a * r31
    b32 = -b * r12 + a * r32
    b33 = -b * r13 + a * r33
    
    ch3,sh3 = QRGivens(b22,b32)
    a = 1 - 2 * sh3 * sh3
    b = 2 * ch3 * sh3
    r11 = b11
    r12 = b12
    r13 = b13
    r21 = a * b21 + b * b31
    r22 = a * b22 + b * b32
    r23 = a * b23 + b * b33
    r31 = -b * b21 + a * b31
    r32 = -b * b22 + a * b32
    r33 = -b * b23 + a * b33
    
    sh12 = sh1 * sh1
    sh22 = sh2 * sh2
    sh32 = sh3 * sh3
    q11 = (-1 + 2 * sh12) * (-1 + 2 * sh22)
    q12 = 4 * ch2 * ch3 * (-1 + 2 * sh12) * sh2 * sh3 + 2 * ch1 * sh1 * (-1 + 2 * sh32)
    q13 = 4 * ch1 * ch3 * sh1 * sh3 - 2 * ch2 * (-1 + 2 * sh12) * sh2 * (-1 + 2 * sh32)

    q21 = 2 * ch1 * sh1 * (1 - 2 * sh22)
    q22 = -8 * ch1 * ch2 * ch3 * sh1 * sh2 * sh3 + (-1 + 2 * sh12) * (-1 + 2 * sh32)
    q23 = -2 * ch3 * sh3 + 4 * sh1 * (ch3 * sh1 * sh3 + ch1 * ch2 * sh2 * (-1 + 2 * sh32))

    q31 = 2 * ch2 * sh2
    q32 = 2 * ch3 * (1 - 2 * sh22) * sh3
    q33 = (-1 + 2 * sh22) * (-1 + 2 * sh32)
    
    R = np.array([[r11,r12,r13],[r21,r22,r23],[r31,r32,r33]])
    Q = np.array([[q11,q12,q13],[q21,q22,q23],[q31,q32,q33]])
    return Q,R
    

def svd(A):
    symmMat = np.dot(A.T,A)
    s11 = symmMat[0,0]
    s21 = symmMat[1,0]
    s22 = symmMat[1,1]
    s31 = symmMat[2,0]
    s32 = symmMat[2,1]
    s33 = symmMat[2,2]
    # 计算特征值，并且返回四元数形式的特征值向量
    qV = jacobiEigenAnlysis(s11, s21, s22, s31, s32, s33)
    # 计算右奇异矩阵
    V = quatToMat(qV)
    # 计算左侧 B = UΣ
    B = np.dot(A,V)
    # 重新排序以便之后的QR分解
    B,V = sortSingluar(B, V)
    # QR分解以计算左奇异矩阵和中间的对角矩阵
    U,S = QRdecomp(B)
    return U,S,V
    
    
    

A = np.array([[5,3,4],[5,7,3],[2,6,9]])
u1,s1,v1 = np.linalg.svd(A)
u2,s2,v2 = svd(A)
    
        