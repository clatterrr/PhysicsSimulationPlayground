import numpy as np

# 连续碰撞检测 点-面
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
pos0 = np.array([1,1,0]) # 点P的位置 
pos1 = np.array([0,0,1]) # 三角形顶点 1 的位置
pos2 = np.array([1,0,1]) # 三角形顶点 2 的位置
pos3 = np.array([0,1,1]) # 三角形顶点 3 的位置

vel0 = np.array([0,0,0]) # 点 P 的速度
vel1 = np.array([0,0,-1]) # 三角形顶点 1 的速度
vel2 = np.array([-1,1,-1]) # 三角形顶点 2 的速度
vel3 = np.array([1,-1,-1]) # 三角形顶点 3 的速度

v01 = vel0 - vel1
v21 = vel2 - vel1
v31 = vel3 - vel1

x01 = pos0 - pos1
x21 = pos2 - pos1
x31 = pos3 - pos1

coeff_a = dot(v01,cross(v21,v31))
coeff_b = dot(x01,cross(v21,v31)) + dot(v01,cross(v21,x31)) + dot(v01,cross(x21,v31))
coeff_c = dot(x01,cross(v21,x31)) + dot(x01,cross(x21,v31)) + dot(v01,cross(x21,x31))
coeff_d = dot(x01,cross(x21,x31))

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

def signed_vf_distance(px,py0,py1,py2):
    normal = np.zeros((3))
    w = np.zeros((4))
    y10 = py1 - py0
    y10 /= np.sqrt(y10[0]**2 + y10[1]**2 + y10[2]**2)
    y20 = py2 - py0
    y20 /= np.sqrt(y20[0]**2 + y20[1]**2 + y20[2]**2)
    normal = cross(y10, y20)
    norm = np.sqrt(normal[0]**2 + normal[1]**2 + normal[2]**2) # 三角形的法向量
    if norm < 1e-6:
        return np.array([1,1,1,1])
    normal /= norm
    # 下面的公式来自 Fast and Exact Continuous Collision Detection
    # with Bernstein Sign Classification 的公式(10)
    h = dot(py0 - px,normal)
    # 如果 点 p 
    b0 = dot(py1 - px,cross(py2 - px,normal))
    b1 = dot(py2 - px,cross(py0 - px,normal))
    b2 = dot(py0 - px,cross(py1 - px,normal))
    w[0] = h
    w[1] = b0 / (b0 + b1 + b2)
    w[2] = b1 / (b0 + b1 + b2)
    w[3] = b2 / (b0 + b1 + b2)
    return w
                
# 主程序继续
t = np.zeros((4)) # 解
nsol = solve_cubic(coeff_a, coeff_b, coeff_c, coeff_d, t)
t[nsol] = 1

pointInTriangleResult = False

for i in range(nsol):
    if t[i] < 0 or t[i] > 1:
        continue
    
    # 因为求出的解尽管符合那个三次方程，但不一定有物理意义
    # 所以还要检测一遍在这个时间点，点 p 是否真的在三角形里
    # 本篇的代码的例子就是如此 
    pos0_new = pos0 + t[i] * vel0
    pos1_new = pos1 + t[i] * vel1
    pos2_new = pos2 + t[i] * vel2
    pos3_new = pos3 + t[i] * vel3
    
    w = signed_vf_distance(pos0_new,pos1_new,pos2_new,pos3_new)
    
    if abs(min(min(w[1],w[2]),w[3])) < 1e-6:
        # 点在边上
        pointInTriangleResult = True

    if abs(w[0]) < 1e-6:
        # 点在三角形内
        pointInTriangleResult = True
        

    
    