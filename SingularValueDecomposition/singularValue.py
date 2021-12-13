import numpy as np

def QRdecomposition(A):
    m, n = A.shape
    Q = np.eye(m)
    for i in range(n - (m == n)):
        H = np.eye(m)
        H[i:, i:] = householder(A[i:, i])
        Q = np.dot(Q, H)
        A = np.dot(H, A)
    return Q, A
 
def householder(a):
    v = a / (a[0] + np.copysign(np.linalg.norm(a), a[0]))
    v[0] = 1
    H = np.eye(a.shape[0])
    inner = np.dot(v, v)
    outter = np.dot(v[:, None], v[None, :])
    H -= (2 / np.dot(v, v)) * np.dot(v[:, None], v[None, :])
    return H
 
def computeEigenValue(A):
    A_new = A.copy()
    R_old = np.zeros((3,3))
    for i in range(100):
        Q,R = QRdecomposition(A_new)
        term0 = R[0,0] + R[1,1] + R[2,2]
        term1 = R_old[0,0] + R_old[1,1] + R_old[2,2]
        R_old = R.copy()
        if abs(term0 - term1)<1e-10:
            break
        A_new = np.dot(R,Q)
    return R_old
        
def computeEigenVector(A):
    A[1,:] = A[1,:] - A[0,:] * A[1,0] / A[0,0]
    res = np.ones((3))
    res[1] = - A[1,2] / A[1,1]
    res[0] = - (A[0,1] * res[1] + A[0,2] * res[2]) / A[0,0]
    res = res / np.linalg.norm(res)
    return res
    

def svd3x3(A):
    
    if abs(A[0,0] * A[1,1] * A[2,2] - np.linalg.det(A)):
        return np.identity(3),A,np.identity(3)
    
    AtA = np.dot(A.T,A)
    eigenValue = computeEigenValue(AtA)
    eiv,eivv = np.linalg.eig(AtA)
    eigenVector = np.zeros((3,3))
    eigenVector[0,:] = computeEigenVector(AtA - np.identity(3)*abs(eigenValue[0,0]))
    eigenVector[1,:] = computeEigenVector(AtA - np.identity(3)*abs(eigenValue[1,1]))
    eigenVector[2,:] = computeEigenVector(AtA - np.identity(3)*abs(eigenValue[2,2]))
    sigma = np.zeros((3,3))
    sigma[0,0] = np.sqrt(abs(eigenValue[0,0]))
    sigma[1,1] = np.sqrt(abs(eigenValue[1,1]))
    sigma[2,2] = np.sqrt(abs(eigenValue[2,2]))
    svd_s = sigma
    svd_v = eigenVector
    svd_u = np.zeros((3,3))
    svd_u[:,0] = np.dot(A,svd_v[0,:]) / sigma[0,0]
    svd_u[:,1] = np.dot(A,svd_v[1,:]) / sigma[1,1]
    svd_u[:,2] = np.dot(A,svd_v[2,:]) / sigma[2,2]
    return svd_u,svd_s,svd_v

# 初始矩阵
Amat = np.array([[0,1,1],[1.414,2,0],[0,1,1]])
Amat = np.array([[1,0,0],[0,1,0],[0,0,2]])
import datetime
# 奇异值分解
time0 = datetime.datetime.now()
u0,s0,v0t = svd3x3(Amat)
time1 = datetime.datetime.now()

# 库函数验证
u1,s,v1t = np.linalg.svd(Amat)
time2 = datetime.datetime.now()
s1 = np.zeros((3,3))
s1[0,0] = s[0]
s1[1,1] = s[1]
s1[2,2] = s[2]

error0 = np.dot(np.dot(u0,s0),v0t) - Amat
error1 = np.dot(np.dot(u1,s1),v1t) - Amat

print(time1 - time0)
print(time2 - time1)