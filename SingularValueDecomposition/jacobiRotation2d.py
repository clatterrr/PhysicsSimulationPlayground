import numpy as np

A = np.array([[1,2],[0,1.0]],dtype = float)
AtA = A.T @ A

def GivensRotate(a11,a12,a22):
    xi = 0.5 * (a11 - a22) 
    if abs(a12) < 1e-10:
        return 1,0
    w = np.sqrt(xi * xi + a12 * a12)
    if xi > 0:
    	t = a12 / (xi + w)
    else:
        t = a12 / (xi - w)
    c = 1.0 / np.sqrt(t * t + 1)
    s = - t * c
    return c,s

c,s = GivensRotate(AtA[0,0], AtA[0,1], AtA[1,1])
Q = np.array([[c,s],[-s,c]])
AtA = Q.T @ AtA @ Q