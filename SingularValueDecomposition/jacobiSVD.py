import numpy as np

# 算法来源于Eigen的JacobiSVD，注释也一样，完成，数据一样，但是sigma哪里去了？
N = 3

A = np.array([[1,2,3],[4,5,6],[7,8,10]],dtype = float)

finished = False
considerAsZero = 1.17549435e-38
precision = 2.38418579e-07
maxDiagEntry = 1

workMatrix = A.copy()
maxElement = A.max()
workMatrix /= maxElement

U = np.array([[1,0,0],[0,1,0],[0,0,1]],dtype = float)
V = np.array([[1,0,0],[0,1,0],[0,0,1]],dtype = float)
def makeJacobi(x,y,z):
    deno = 2 * abs(y)
    m_c = 1
    m_s = 0
    if deno > considerAsZero:
        tau = (x - z) / deno
        w = np.sqrt(tau * tau + 1)
        t = 0
        if tau > 0:
            t = 1 / (tau + w)
        else:
            t = 1 / (tau - w)
        n = 1 / np.sqrt(t*t + 1)
        m_s = - np.sign(t) * y / abs(y) * abs(t) * n
        m_c = n
    return np.array([m_c,m_s])
    
def rotate(a,b,c,s):
    a0 = c * a + s * b # - 0.45 + 
    b0 = -s * a + c * b
    return a0,b0

def rotatePlaneLeft(m,p,q,c,s):
    m_rotate = m.copy()
    n = m.shape[0]
    for j in range(n):
        m_rotate[p,j],m_rotate[q,j] = rotate(m[p,j], m[q,j], c, s)
    return m_rotate

def rotatePlaneRight(m,p,q,c,s):
    m_rotate = m.copy()
    n = m.shape[0]
    for i in range(n):
        m_rotate[i,p],m_rotate[i,q] = rotate(m[i,p], m[i,q], c, s)
    return m_rotate

def real_2x2_jacobi_svd(matrix,p,q):
    m = np.array([[matrix[p,p],matrix[p,q]],[matrix[q,p],matrix[q,q]]])
    t = m[0,0] + m[1,1]
    d = m[1,0] - m[0,1]
    rotate_s = 0
    rotate_c = 0
    if abs(d) < considerAsZero:
        rotate_s = 0
        rotate_c = 1
    else:
        u = t / d
        tmp = np.sqrt(1 + u*u)
        rotate_s = 1 / tmp
        rotate_c = u / tmp
    m = rotatePlaneLeft(m,0,1,rotate_c,rotate_s)
    j_right = makeJacobi(m[0,0], m[0,1], m[1,1])
    j_left = np.zeros((2))
    j_left[0] = rotate_c * j_right[0] + rotate_s * j_right[1]
    j_left[1] = rotate_s * j_right[0] - rotate_c * j_right[1]
    return j_left,j_right

# step 2. The main Jacobi SVD iteration.
while(finished == False):
    finished = True
    # do a sweep: for all index pairs (p,q), 
    # perform SVD of the corresponding 2x2 sub-matrix
    for p in range(1,N):
        for q in range(0,p):
            # if this 2x2 sub-matrix is not diagonal already...
            # notice that this comparison will evaluate to false if any NaN is involved, 
            # ensuring that NaN's don't keep us iterating forever. 
            # Similarly, small denormal numbers are considered zero.
            threshold = max(considerAsZero,precision*maxDiagEntry)
            if abs(workMatrix[p,q]) > threshold or abs(workMatrix[q,p]) > threshold:
                finished = False
            j_left,j_right = real_2x2_jacobi_svd(workMatrix, p, q)
            workMatrix = rotatePlaneLeft(workMatrix, p, q, j_left[0], j_left[1])
            U = rotatePlaneRight(U, p, q, j_left[0], j_left[1])
            workMatrix = rotatePlaneRight(workMatrix, p, q, j_right[0], -j_right[1])
            V = rotatePlaneRight(V, p, q, j_right[0], -j_right[1])
            maxDiagEntry = max(maxDiagEntry,workMatrix[p,p],workMatrix[q,q])
            
# 
            
#  step 3. The work matrix is now diagonal, 
# so ensure it's positive so its diagonal entries are the singular values
            
singularValues = np.zeros((N))
for i in range(N):
    a = workMatrix[i,i]
    if a < 0:
        workMatrix[i,i] = -a
        U[:,i] = - U[:,i]
    singularValues[i] = abs(a) * maxElement
    
#  step 4. Sort singular values in descending order and compute the number of nonzero singular values
for i in range(N):
    maxValue = 0
    maxpos = -1
    for j in range(i,N):
        if maxValue < singularValues[j]:
            maxpos = N - 1 - j # tail 倒着来
            maxValue = singularValues[j]
    if maxValue == 0:
        break
    if maxpos > 0:
        maxpos += i
        
        tmp = V[:,i].copy()
        V[:,i] = V[:,maxpos].copy()
        V[:,maxpos] = tmp.copy()
        
        tmp = U[:,i].copy()
        U[:,i] = U[:,maxpos].copy()
        U[:,maxpos] = tmp.copy()
        
        temp = 0
            