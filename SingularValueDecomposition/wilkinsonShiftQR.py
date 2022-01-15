import numpy as np
n = 3
A = np.array([[3,1,2],[2,7,2],[2,1,3]])
A = np.dot(A.T,A)

def wilksonShiftQR(A, iterations=100):
    Ak = np.copy(A)
    n = Ak.shape[0]
    QQ = np.eye(n)
    for k in range(iterations):
        b11 = A[n-2,n-2]
        b22 = A[n-1,n-1]
        b12 = A[n-2,n-1]
        b21 = A[n-1,n-2]
        delta = (b11 - b22) / 2
        sign = 0
        if abs(delta) < 1e-10:
            sign = - 1
        else:
            sign = np.sign(delta)
        shift = b22 - (sign*b21*b21) / (abs(delta) + np.sqrt(delta*delta + b21*b21))
        smult = shift * np.eye(n)
        Q, R = np.linalg.qr(Ak - smult)
        Ak = np.dot(R,Q) + smult
    return Ak

eiv0 =  wilksonShiftQR(A)
eiv1 = np.linalg.eig(A)[0]