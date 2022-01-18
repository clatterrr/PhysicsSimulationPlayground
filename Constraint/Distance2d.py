import numpy as np
# box2d
# success
x = np.array([[0,0],
              [5,1]],dtype = float)
r = np.array([[2,0],
              [-2,0]],dtype = float)

m0 = 0.1
m1 = 0.1
I0 = 1
I1 = 1
v0 = np.zeros((2))
v1 = np.zeros((2))
w0 = 0
w1 = 0
rest_len = 1
def cross2d(a,b):
    return a[0] * b[1] - a[1] * b[0]

def cross2d2(a,b):
    return np.array([-a * b[1], a * b[0]])

time = 0
timeFinal = 100
x_record = np.zeros((timeFinal,2,2))
while time < timeFinal:
    x_record[time] = x
    m_u = x[1] + r[1] - x[0] - r[0]
    cur_len = np.linalg.norm(m_u)
    cr0 = cross2d(r[0], m_u)
    cr1 = cross2d(r[1], m_u)
    invMass = 1.0 / m0 + 1.0 / m1 + I0 * cr0 * cr0 + I1 * cr1 * cr1
    C = cur_len - rest_len
    damping = 1
    stiffness = 1
    dt = 0.1
    gamma = dt * (damping + dt * stiffness)
    bias = C * dt * stiffness / gamma
    softMass = invMass + bias
    if softMass != 0:
        softMass = 1.0 / softMass
    vp0 = v0 + cross2d2(w0, r[0])
    vp1 = v1 + cross2d2(w1, r[1])
    Cdot = np.dot(m_u,vp1 - vp0)
    impluse = - softMass * (Cdot + bias)
    P = impluse * m_u
    v0 -= 1.0 / m0 * P
    v1 += 1.0 / m1 * P
    w0 = I0 * cross2d(r[0], P)
    w1 = I1 * cross2d(r[1], P)
    x[0] = x[0] + dt * v0
    x[1] = x[1] + dt * v1
    r[0] = r[0] + dt * cross2d2(w0, r[0])
    r[1] = r[1] + dt * cross2d2(w1, r[1])
    time += 1
    
    
    
