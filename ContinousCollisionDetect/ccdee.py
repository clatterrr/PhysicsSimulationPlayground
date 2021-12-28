import numpy as np

# 连续碰撞检测 三维 边-边
# 参考：arcsim库 http://graphics.berkeley.edu/resources/ARCSim/

def cross(vec0,vec1):
    res = np.zeros((3))
    res[0] = vec0[1]*vec1[2] - vec0[2]*vec1[1]
    res[1] = vec0[2]*vec1[0] - vec0[0]*vec1[2]
    res[2] = vec0[0]*vec1[1] - vec0[1]*vec1[0]
    return res

def dot(vec0,vec1):
    res = 0
    for i in range(len(vec0)):
        res += vec0[i]*vec1[i]
    return res

# 时刻 0 的位置
pos0 = np.array([0,0,0]) # 边0 顶点 0
pos1 = np.array([2,0,0]) # 边0 顶点 1
pos2 = np.array([1,1,-1]) # 边1 顶点 0
pos3 = np.array([1,1,1]) # 边1 顶点 1

vel0 = np.array([0,0,0]) 
vel1 = np.array([0,0,0]) 
vel2 = np.array([0,-2,0])
vel3 = np.array([0,-2,0]) 

x20 = pos2 - pos0
x10 = pos1 - pos0
x32 = pos3 - pos2

v20 = vel2 - vel0
v10 = vel1 - vel0
v32 = vel3 - vel2

coeff_a = dot(v20,cross(v10,v32))
coeff_b = dot(x20,cross(v10,v32)) + dot(v20,cross(v10,x32)) + dot(v20,cross(x10,v32))
coeff_c = dot(x20,cross(v10,x32)) + dot(x20,cross(x10,v32)) + dot(v20,cross(x10,x32))
coeff_d = dot(x20,cross(x10,x32))

# 接下来解方程a^3 t + b^2 t + c t + d 

# 解三次方程用牛顿牛顿迭代法 x^{n+1} = x^n - frac{F}{F'}

def solve_cubic(a,b,c,d,x):
    xc = np.zeros((2))
    solutionNum = solve_quadratic(3*a, 2*b, c, xc)
    if solutionNum == 0:
        x[0] = newtowns_method(a,b,c,d,xc[0],0)
        return 1
    elif solutionNum == 1:
        return solve_quadratic(b,c,d, x)
    else:
        yc = np.array([a*xc[0]**3 + b*xc[0]**2 + c*xc[0] + d,
                       a*xc[1]**3 + b*xc[1]**2 + c*xc[1] + d])
        i = 0
        if yc[0] * a >= 0:
            x[i+1] = newtowns_method(a, b, c, d, xc[0], -1)
            i += 1
        if yc[0]*yc[1] <= 0:
            closer = -1
            if abs(yc[0]) >= abs(yc[1]):
                closer = 1
            x[i+1] = newtowns_method(a, b, c, d, xc[1], closer)
            i += 1
        if yc[1] *a <= 0:
            x[i+1] = newtowns_method(a, b, c, d, xc[1], 1)
            i += 1
        return i

# 解法来自 http://en.wikipedia.org/wiki/Quadratic_formula#Floating_point_implementation
def solve_quadratic(a,b,c,x):
    d = b*b - 4*a*c
    if d < 0:
        return 0
    if a == 0:
        if b == 0:
            return np.inf
        else:
            return  - c / b
    x[0] = - b / (2 * a)
    q = -(b + np.sign(b) + np.sqrt(d)) / 2
    i = 0
    if abs(a) > 1e-12 * abs(q):
        x[i+1] = q / a
        i += 1
    if abs(q) > 1e-12 * abs(c):
        x[i+1] = c / q
        i += 1
    if i == 2 and x[0] > x[1]:
        temp = x[0]
        x[0] = x[1]
        x[1] = temp
    return i
    
def newtowns_method(a,b,c,d,x0,init_dir):
    if init_dir != 0:
        y0 = a*x0**3 + b*x0**2 + c*x0 + d
        ddy0 = 6*a*x0 + 2*b
        if ddy0 != 0:
            x0 += init_dir * np.sqrt(abs(2 * y0 / ddy0))
    for ite in range(100):
        y = a*x0**3 + b*x0**2 + c*x0 + d
        dy = 3*a*x0**2 + 2*b*x0 + c
        if dy == 0:
            return x0
        x1 = x0 - y / dy
        if abs(x0 - x1) < 1e-6:
            return x0
        x0 = x1
    return x0

def signed_ee_distance(x0,x1,y0,y1):
    normal = np.zeros((3))
    w = np.zeros((4))
    x10 = x1 - x0
    y10 = y1 - y0
    # x10 = x10 / np.linalg.norm(x10)
    x10 = x10 / np.sqrt(dot(x10,x10))
    y10 = y10 / np.sqrt(dot(y10,y10))
    normal = cross(x10,y10)
    norm = np.sqrt(dot(normal,normal))
    if norm < 1e-6:
        return np.array([1,1,1,1])
    normal = normal / norm
    h = dot(x0 - y0,normal)
    
    a0 = dot(y1 - x1,cross(y0 - x1,normal))
    a1 = dot(y0 - x0,cross(y1 - x0,normal))
    b0 = dot(x0 - y1,cross(x1 - y1,normal))
    b1 = dot(x1 - y0,cross(x0 - y0,normal))
    w = np.zeros((5))
    w[0] = a0 / (a0 + a1)
    w[1] = a1 / (a0 + a1)
    w[2] = - b0 / (b0 + b1)
    w[3] = - b1 / (b0 + b1)
    w[4] = h
    return w
                
# 主程序继续
t = np.zeros((4)) # 解
nsol = solve_cubic(coeff_a, coeff_b, coeff_c, coeff_d, t)
t[nsol] = 1

IntersectionResult = False

for i in range(nsol):
    if t[i] < 0 or t[i] > 1:
        continue
    
    # 因为求出的解尽管符合那个三次方程，但不一定有物理意义
    # 所以还要检测一遍在这个时间点，两条线段是否真的相交
    # 本篇的代码的例子就是如此 
    pos0_new = pos0 + t[i] * vel0
    pos1_new = pos1 + t[i] * vel1
    pos2_new = pos2 + t[i] * vel2
    pos3_new = pos3 + t[i] * vel3
    
    w = signed_ee_distance(pos0_new,pos1_new,pos2_new,pos3_new)
    
    d = abs(w[4])
    
    if min(min(w[0],w[1]),min(-w[2],-w[3])) >= -1e-6:
        IntersectionResult = True
        

    
    