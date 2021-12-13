import numpy as np

EPSILON = 1e-10

def G2(c,s):
    return np.array([[c,s],[-s,c]])

def G2_Con(a,b):
    d = a * a + b * b
    c = 1
    s = 0
    if abs(d) > 0:
        t = 1.0 / np.sqrt(d)
        c = a * t
        s = - b * t
    return c,s

def G2_unCon(a,b):
    d = a * a + b * b
    c = 1
    s = 0
    if abs(d) > 0:
        t = 1.0 / np.sqrt(d)
        c = a * t
        s = b * t
    return c,s

def G3_12(c,s,con = True):
    if con == True:
        c,s = G2_Con(c, s)
    else:
        c,s = G2_unCon(c, s)
    return np.array([[c,s,0],[-s,c,0],[0,0,1]])

def G3_12_Direct(c,s):
    return np.array([[c,s,0],[-s,c,0],[0,0,1]])
    

def G3_23(c,s,con = True):
    if con == True:
        c,s = G2_Con(c, s)
    else:
        c,s = G2_unCon(c, s)
    return np.array([[1,0,0],[0,c,s],[0,-s,c]])

def G3_23_Direct(c,s):
    return np.array([[1,0,0],[0,c,s],[0,-s,c]])

def G3_13(c,s,con = True):
    if con == True:
        c,s = G2_Con(c, s)
    else:
        c,s = G2_unCon(c, s)
    return np.array([[c,0,s],[0,1,0],[-s,0,c]])

def G3_13_Direct(c,s):
    return np.array([[c,0,s],[0,1,0],[-s,0,c]])

def PolarDecompostion(A):
    R = np.zeros((2,2))
    S = np.zeros((2,2))
    x = A[0,0] + A[1,1]
    y = A[1,0] - A[0,1]
    d = np.sqrt(x*x + y*y)
    R = G2(1,0)
    if abs(d) > EPSILON:
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
    if abs(S[0,1]) < EPSILON:
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

def ZeroChasing(U,A,V):
    G = G3_12(A[0,0],A[1,0])
    A = np.dot(G.T,A)
    U = np.dot(U,G)
    c = A[0,1]
    s = A[0,2]
    if abs(A[1,0] > EPSILON):
        c = A[0,0] * A[0,1] + A[1,0] * A[1,1]
        s = A[0,0] * A[0,2] + A[1,0] * A[1,2]
    G = G3_23(c,s)
    A = np.dot(A,G)
    V = np.dot(V,G)
    
    G = G3_23(A[1,1],A[2,1])
    A = np.dot(G.T,A)
    U = np.dot(U,G)
    return U,A,V

def Bidiag(U,A,V):
    G = G3_23(A[1,0],A[2,0])
    A = np.dot(G.T,A)
    U = np.dot(U,G)
    return ZeroChasing(U, A, V)

def FrobeniusNorm(B):
    ret = 0
    for i in range(3):
        for j in range(3):
            ret += B[i,j] * B[i,j]
    return ret

def FlipSign(idx,mat,sigma):
    mat[0,idx] = - mat[0,idx]
    mat[1,idx] = - mat[1,idx]
    mat[2,idx] = - mat[2,idx]
    sigma[idx] = - sigma[idx]
    return mat,sigma

def FlipSignColumn(idx,mat):
    mat[0,idx] = - mat[0,idx]
    mat[1,idx] = - mat[1,idx]
    mat[2,idx] = - mat[2,idx]
    return mat

def SwapColumn(a,col_a,col_b):
    acopy = a.copy()
    a[:,col_a] = a[:,col_b]
    a[:,col_b] = acopy[:,col_a]
    return a

def SortWithTopLeft(U,sigma,V):
    if abs(sigma[1]) >= abs(sigma[2]):
        if sigma[1] < 0:
            U,sigma = FlipSign(1, U, sigma)
            U,sigma = FlipSign(2, U, sigma)
        return U,sigma,V
    if sigma[2] < 0:
        U,sigma = FlipSign(1, U, sigma)
        U,sigma = FlipSign(2, U, sigma)
    temp = sigma[1]
    sigma[1] = sigma[2]
    sigma[2] = temp
    U = SwapColumn(U, 1, 2)
    V = SwapColumn(V, 1, 2)
    if sigma[1] > sigma[0]:
        temp = sigma[0]
        sigma[0] = sigma[1]
        sigma[1] = temp
        U = SwapColumn(U, 0, 1)
        V = SwapColumn(V, 0, 1)
    else:
        U = FlipSignColumn(U, 2)
        V = FlipSignColumn(V, 2)
    return U,sigma,V

def SortWithBotRight(U,sigma,V):
    if abs(sigma[0]) >= abs(sigma[1]):
        if sigma[0] < 0:
            U,sigma = FlipSign(0, U, sigma)
            U,sigma = FlipSign(2, U, sigma)
        return U,sigma,V
    temp = sigma[0]
    sigma[0] = sigma[1]
    sigma[1] = temp
    U = SwapColumn(U, 0, 1)
    V = SwapColumn(V, 0, 1)
    if abs(sigma[1]) < abs(sigma[2]):
        temp = sigma[1]
        sigma[1] = sigma[2]
        sigma[2] = temp
        U = SwapColumn(U, 1, 2)
        V = SwapColumn(V, 1, 2)
    else:
        U = FlipSignColumn(U, 2)
        V = FlipSignColumn(V, 2)
    
    if sigma[1] < 0:
        U,sigma = FlipSign(1, U, sigma)
        U,sigma = FlipSign(2, U, sigma)
    return U,sigma,V

def SolveReducedTopLeft(B,U,V):
    s3 = B[2,2]
    top_left = B[0:2,0:2]
    U2,D2,V2 = SVD2x2(top_left)
    u3 = G3_12_Direct(U2[0,0], U2[0,1])
    v3 = G3_12_Direct(U2[0,0], U2[0,1])
    U = np.dot(U,u3)
    V = np.dot(V,v3)
    sigma = np.array([D2[0],D2[1],s3])
    return U,sigma,V

def SolveReducedBotRight(B,U,V):
    s1 = B[0,0]
    botRight = B[1:3,1:3]
    U2,D2,V2 = SVD2x2(botRight)
    u3 = G3_23_Direct(U2[0,0], U2[0,1])
    v3 = G3_23_Direct(U2[0,0], U2[0,1])
    U = np.dot(U,u3)
    V = np.dot(V,v3)
    sigma = np.array([s1,D2[0],D2[1]])
    return U,sigma,V
    
def PostProcess(B,U,V,alpha1,alpha2,alpha3
                ,beta1,beta2,gamma1,gamma2,tao):
    if abs(beta2) <= tao:
        U,sigma,V = SolveReducedTopLeft(B, U, V)
        U,sigma,V = SortWithTopLeft(U,sigma,V)
    elif abs(beta1) <= tao:
        U,sigma,V = SolveReducedBotRight(B, U, V)
        U,sigma,V = SortWithBotRight(U,sigma,V)
    elif abs(alpha2) <= tao:
        G = G3_23(B[1,2],B[2,2],False)
        B = np.dot(G.T,B)
        U = np.dot(U,G)
        U,sigma,V = SolveReducedTopLeft(B, U, V)
        U,sigma,V = SortWithTopLeft(U,sigma,V)
    elif abs(alpha3) <= tao:
        G = G3_23(B[1,1],B[1,2])
        B = np.dot(B,G)
        V = np.dot(V,G)
        G = G3_13(B[0,0],B[0,2])
        B = np.dot(B,G)
        V = np.dot(V,G)
        U,sigma,V = SolveReducedTopLeft(B, U, V)
        U,sigma,V = SortWithTopLeft(U,sigma,V)
    elif abs(alpha1) <= tao:
       G = G3_12(B[0,1],B[1,1],False)
       B = np.dot(G.T,B)
       U = np.dot(U,G)
       G = G3_13(B[0,2],B[2,2],False)
       B = np.dot(G.T,B)
       U = np.dot(U,G)
       U,sigma,V = SolveReducedBotRight(B, U, V)
       U,sigma,V = SortWithBotRight(U,sigma,V)
    return U,sigma,V
        
def SVD3x3(A):
    U = np.identity(3)
    V = np.identity(3)
    D = np.zeros(3)
    U,B,V = Bidiag(U, A, V)
    alpha1,alpha2,alpha3 = B[0,0],B[1,1],B[2,2]
    beta1,beta2 = B[0,1],B[1,2]
    gamma1,gamma2 = alpha1 * beta1, alpha2 * beta2
    tol = 128 * EPSILON
    tao = tol * max(0.5 * FrobeniusNorm(B),1)
    while(abs(beta1) > tao and abs(beta2) > tao
          and abs(alpha1) > tao and abs(alpha2) > tao
          and abs(alpha3) > tao):
        a1 = alpha2 * alpha2 + beta1 * beta1
        a2 = alpha3 * alpha3 + beta1 * beta1
        b1 = gamma1
        d = (a1 - a2) * 0.5
        mu = (b1 * b1) / (abs(d) + np.sqrt(d*d + b1*b1))
        if d < 0:
            mu = -mu
        mu = a2 - mu
        G = G3_12(alpha1 * alpha1 - mu,gamma1)
        B = np.dot(B,G)
        V = np.dot(V,G)
        U,B,V = Bidiag(U, B, V)
        alpha1,alpha2,alpha3 = B[0,0],B[1,1],B[2,2]
        beta1,beta2 = B[0,1],B[1,2]
        gamma1,gamma2 = alpha1 * beta1, alpha2 * beta2
    count = 0
    return PostProcess(B, U, V, alpha1, alpha2, alpha3, beta1, beta2, gamma1, gamma2, tao)
        
        
A = np.array([[3,4,2],[4,5,2],[2,5,3]])
U0,D0,V0 = SVD3x3(A)
U1,D1,V1 = np.linalg.svd(A)