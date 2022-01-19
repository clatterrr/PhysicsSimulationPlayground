import numpy as np
'''
Copyright (C) 2022 胡虎护弧呼 - All Rights Reserved
You may use, distribute and modify this code under the
terms of the GPL license.

shift qr decomposition
tutorials : https://zhuanlan.zhihu.com/p/459369233
'''
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