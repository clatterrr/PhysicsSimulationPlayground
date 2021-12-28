import numpy as np

def solve_quadratic(a,b,c):
    d = b*b - 4*a*c
    result = np.zeros((2))
    result_num = 0
    if d < 0:
        result_num = 0
        return result_num,result
    q = - (b + np.sign(b)*np.sqrt(d)) / 2
    if (abs(a) > 1e-12*abs(q)):
        result[result_num] = q / a
        result_num += 1
    if (abs(q) > 1e-12*abs(c)):
        result[result_num] = c / q
        result_num += 1
    if result_num == 2 and result[0] > result[1]:
        temp = result[0]
        result[0] = result[1]
        result[1] = temp
    return result_num,result

def solve_cubic(a,b,c,d):
    # 如果a是0，那么直接解二次方程即可
    if abs(a) < 1e-20:
        return solve_quadratic(b,c,d)
    segment_point_num = 0
    segment_point = np.zeros((4))
    # 计算极值点，把三次方程分成三段
    point_num,point = solve_quadratic(3*a, 2*b, c)
    
    # 没有极值点
    if point_num == 0:
        f0 = d
        f1 = a + b + c + d
        if f0 * f1 < 0:
            segment_point[0] = 0
            segment_point[1] = 1
            segment_point_num = 2
    # 一个极值点
    elif point_num == 1:
        segment_point[0] = 0
        segment_point[1] = point[0]
        segment_point[2] = 1
        segment_point_num = 3
    # 两个极值点
    else:
        segment_point[0] = 0
        segment_point[1] = point[0]
        segment_point[2] = point[1]
        segment_point[3] = 1
        segment_point_num = 4
    
    t_num = 0
    t = np.zeros((3))
    
    for i in range(segment_point_num - 1):
        tLow = segment_point[i]
        tHigh = segment_point[i+1]
        fLow = a*tLow*tLow*tLow + b*tLow*tLow + c*tLow + d
        if abs(fLow) < 1e-10:
            t[i] = tLow
            t_num += 1
            continue
        fHigh = a*tHigh*tHigh*tHigh + b*tHigh*tHigh + c*tHigh + d
        if abs(fHigh) < 1e-10:
            t[i] = tHigh
            t_num += 1
            continue
        
        if tLow > 0:
            temp = tLow
            tLow = tHigh
            tHigh = temp
            
        dx = abs(tHigh - tLow)
        tMid = (tLow + tHigh) / 2
        f = a*tMid*tMid*tMid + b*tMid*tMid + c*tMid + d
        df = 3*a*tMid*tMid + 2*b*tMid + c
        
        for ite in range(100):
            fLow = f - df * (tMid - tLow)
            fHigh = f - df * (tMid - tHigh)
            # 如果解可能隔得近，就用牛顿法
            if fLow * fHigh < 0:
                dx = f / df
                tMid = tMid - dx
                # 如果迭代过头了，就换用二分法
                if tMid >= max(tHigh,tLow) or tMid <= min(tHigh,tLow):
                    dx = (tHigh - tLow) / 2
                    tMid = (tHigh + tLow) / 2
            # 如果解可能隔得远，就用二分法
            else:
                dx = (tHigh - tLow) / 2
                tMid = (tHigh + tLow) / 2
            f = a*tMid*tMid*tMid + b*tMid*tMid + c*tMid + d
            if abs(f) < 1e-10:
                t[i] = tMid
                t_num += 1
                break
            df = 3*a*tMid*tMid + 2*b*tMid + c
            if f < 0:
                tLow = tMid
            else:
                tHigh = tMid
    return t_num,t

a = 10
b = -14
c = 5
d = -0.2
t_num,t = solve_cubic(a,b,c,d)
error = np.zeros((t_num))
for i in range(t_num):
    error[i] = a*t[i]**3 + b*t[i]**2 + c*t[i] + d
