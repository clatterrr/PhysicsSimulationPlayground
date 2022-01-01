import numpy as np
# 翻译自 https://atmos.uw.edu/academics/classes/2012Q2/581/lect/matlab/pois_FFT.m
Nx = 32
Ny = 32
L = 1 # 求解域长度
dx = 1 / Nx
dy = 1 / Ny

# 泊松方程等式右边项
f = np.zeros((Nx,Ny))
rsq = np.zeros((Nx,Ny))
ft = 0
for i in range(Nx):
    for j in range(Ny):
        rsq[i,j] = (i*dx - L/2)**2 + (j*dx - L/2)**2
        sigsq = 0.01
        f[i,j] = np.exp(-rsq[i,j] / (2*sigsq))*(rsq[i,j] - 2*sigsq)/(sigsq**2)
f_hat = np.fft.fft2(f)

k = np.zeros((Nx))
l = np.zeros((Nx))
for i in range(0,Nx):
    if i < Nx/2:
        l[i] = k[i] = 2*np.pi/L*i
    else:
        l[i] = k[i] = 2*np.pi/L*(i - Nx)

# 泊松等式左边项
lhs = np.zeros((Nx,Ny),dtype = complex)
for i in range(Nx):
    for j in range(Ny):
        lhs[i,j] = - (k[i]**2 + l[j]**2)
lhs[0,0] = 1# 防止被0除

u_hat = f_hat / lhs
u = np.fft.ifft2(u_hat).real # 得到正确的结果，也就是压力泊松方程的压力
u -= u[0,0]

# 检验结果，用解析及和数值解比较
analytic = np.zeros((Nx,Ny))
error = np.zeros((Nx,Ny)) 
for i in range(1,Nx-1):
    for j in range(1,Ny-1):
        analytic[i,j] = np.exp(-rsq[i,j] / (2 * sigsq))
        error[i,j] = abs(analytic[i,j] - u[i,j])