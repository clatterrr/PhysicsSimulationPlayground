import numpy as np

def G2(c,s):
    return np.array([[c,s],[-s,c]])

def PolarDecompostion(A):
    R = np.zeros((2,2))
    S = np.zeros((2,2))
    x = A[0,0] + A[1,1]
    y = A[1,0] - A[0,1]
    d = np.sqrt(x*x + y*y)
    R = G2(1,0)
    if abs(d) > 1e-10:
        d = 1 / d
        R = G2(x*d,-y*d)
    S = np.dot(R.T,A)
    return R,S

def SVD2x2(A):
    U = np.zeros((2,2))
    D = np.zeros((2))
    V = np.zeros((2,2))
    R,S = PolarDecompostion(A)
    c = 1
    s = 0
    if abs(S[0,1]) < 1e-10:
        D[0] = S[0,0]
        D[1] = S[1,1]
    else:
        taw = 0.5 * (S[0,0] - S[1,1])
        w = np.sqrt(taw * taw + S[0,1]*S[0,1])
        if taw > 0:
            t = S[0,1] / (taw + w)
        else:
            t = S[0,1] / (taw - w)
        c = 1.0 / np.sqrt(t * t + 1)
        s = - t * c
        D[0] = c*c*S[0,0] - 2*c*s*S[0,1] + s*s*S[1,1]
        D[1] = s*s*S[0,0] + 2*c*s*S[0,1] + c*c*S[1,1]
    if D[0] < D[1]:
        temp = D[0]
        D[0] = D[1]
        D[1] = temp
        V = G2(-s,c)
    else:
        V = G2(c,s)
    U = np.dot(R,V)
    return U,D,V

A = np.array([[3.0,2.0],[1.0,4.0]])
eiv,eigvv = np.linalg.eig(A)
AtA = np.dot(A.T,A)
u0,s0,vt0 = SVD2x2(A)
u1,s1,vt1 = np.linalg.svd(A)
