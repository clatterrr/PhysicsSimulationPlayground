import numpy as np
node_num = 3
nei_num = 2
node_pos = np.array([[0,0],
              [1,1],
              [2,0]],dtype = float)
node_nei = np.array([[1,2],[2,0],[0,1]],dtype = int)
P = np.zeros((node_num,nei_num,2))
Dweight = np.zeros((node_num,nei_num,nei_num))
for i in range(node_num):
    for j in range(nei_num):
        pi = i
        pj = node_nei[i,j]
        P[pi,j,:] = node_pos[pi] - node_pos[pj]
        Dweight[pi,j,j] = 0.5
R = np.zeros((node_num,2,2))
L = np.array([[1,0,0],[-0.5,1,-0.5],[-0.5,-0.5,1]])
node_pos_prime = np.array([[0,0],
              [4,4],
              [4,0]],dtype = float)
time = 0
timeFinal = 100
node_pos_prime_t = np.zeros((timeFinal,node_num,2))
while time < timeFinal:
    Pprime = np.zeros((node_num,nei_num,2))
    for i in range(node_num):
        for j in range(nei_num):
            pi = i
            pj = node_nei[i,j]
            Pprime[pi,j,:] = node_pos_prime[pi] - node_pos_prime[pj]
            Dweight[pi,j,j] = 0.5
        S = np.dot(np.dot(P[i],Dweight[i]),Pprime[i].T)
        U,sigma,Vt = np.linalg.svd(S)
        R[i] = np.dot(Vt.T,U.T)
    rhs = np.zeros((node_num,2))
    rhs[0,:] = node_pos[0,:]
    for i in range(1,node_num):
        for j in range(nei_num):
            pi = i
            pj = node_nei[i,j]
            rhs[pi,:] += Dweight[pi,j,j] * 0.5 * np.dot((R[pi] + R[pj]),node_pos[pi] - node_pos[pj])
    node_pos_prime = np.dot(np.linalg.inv(L),rhs)
    node_pos_prime_t[time] = node_pos_prime
    time += 1