import numpy as np
import math

cA = np.cos(-45.0 / 180.0 * np.pi)
sA = np.sin(-45.0 / 180.0 * np.pi)
A = np.array([[cA,-sA],[sA,cA]])
A = np.array([[1,0.1],[0.1,1]])
Su = A @ A.T

phi = 0.5 * math.atan2(Su[0,1]+Su[1,0], Su[0,0] - Su[1,1])
Cphi = np.cos(phi)
Sphi = np.sin(phi)
U = np.array([[Cphi,-Sphi],[Sphi,Cphi]])
 
Sw = A.T @ A
theta = 0.5 * math.atan2(Sw[0,1]+Sw[1,0], Sw[0,0]-Sw[1,1])
Ctheta = np.cos(theta)
Stheta = np.sin(theta)
W = np.array([[Ctheta,-Stheta],[Stheta,Ctheta]])

SUsum = Su[0,0] + Su[1,1]
SUdif = np.sqrt((Su[0,0] - Su[1,1])**2 + 4*Su[0,1]*Su[1,0])
svals = np.array([[np.sqrt((SUsum + SUdif)/2),0],
                  [0,np.sqrt((SUsum - SUdif)/2)]])

S = U.T @ A @ W
C = np.array([[np.sign(S[0,0]),0],
              [0,np.sign(S[1,1])]])
V = W @ C
Ao = U @ svals @ V
u0,s0,v0 = np.linalg.svd(A)