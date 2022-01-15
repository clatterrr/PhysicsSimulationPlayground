import numpy as np

# A is a square random matrix of size n
n = 3
A = np.array([[3,1,2],[2,7,2],[2,1,3]])
A = np.dot(A.T,A)

def eigen_qr_practical(A, iterations=100):
    Ak = np.copy(A)
    n = Ak.shape[0]
    QQ = np.eye(n)
    for k in range(iterations):
        # s_k is the last item of the first diagonal
        s = Ak.item(n-1, n-1)
        smult = s * np.eye(n)
        # pe perform qr and subtract smult
        Aks = np.subtract(Ak, smult)
        Q, R = np.linalg.qr(np.subtract(Ak, smult))
        # we add smult back in
        Ak = np.add(R @ Q, smult)
        QQ = QQ @ Q
    return Ak, QQ

#Print results
ak,qq = eigen_qr_practical(A)