import numpy as np

def DFT1D(fx):
    fx = np.asarray(fx, dtype=complex)
    M = fx.shape[0]
    fu = fx.copy()

    for i in range(M):
        u = i
        sum = 0
        for j in range(M):
            x = j
            tmp = fx[x]*np.exp(-2j*np.pi*x*u*np.divide(1, M, dtype=complex))
            sum += tmp
        # print(sum)
        fu[u] = sum
    # print(fu)

    return fu


def inverseDFT1D(fu):
    fu = np.asarray(fu, dtype=complex)
    M = fu.shape[0]
    fx = np.zeros(M, dtype=complex)

    for i in range(M):
        x = i
        sum = 0
        for j in range(M):
            u = j
            tmp = fu[u]*np.exp(2j*np.pi*x*u*np.divide(1, M, dtype=complex))
            sum += tmp
        fx[x] = np.divide(sum, M, dtype=complex)

    return fx


def FFT1D(fx):
    """ use recursive method to speed up"""
    fx = np.asarray(fx, dtype=complex)
    M = fx.shape[0]
    minDivideSize = 4

    if M % 2 != 0:
        raise ValueError("the input size must be 2^n")

    if M <= minDivideSize:
        return DFT1D(fx)
    else:
        fx_even = FFT1D(fx[::2])  # compute the even part
        fx_odd = FFT1D(fx[1::2])  # compute the odd part
        W_ux_2k = np.exp(-2j * np.pi * np.arange(M) / M)

        f_u = fx_even + fx_odd * W_ux_2k[:M//2]

        f_u_plus_k = fx_even + fx_odd * W_ux_2k[M//2:]

        fu = np.concatenate([f_u, f_u_plus_k])

    return fu


def inverseFFT1D(fu):
    """ use recursive method to speed up"""
    fu = np.asarray(fu, dtype=complex)
    fu_conjugate = np.conjugate(fu)

    fx = FFT1D(fu_conjugate)

    fx = np.conjugate(fx)
    fx = fx / fu.shape[0]

    return fx

L = 2
N = 32
x = np.zeros((N))
k = np.zeros((N))
exact = np.zeros((3,N))
fourier = np.zeros((3,N))
error = np.zeros((3,N))
for i in range(0,N):
    x[i] = L/N*(i - N/2)
    if i < N/2:
        k[i] = 2*np.pi/L*i
    else:
        k[i] = 2*np.pi/L*(i - N)
u = np.exp(np.sin(np.pi*x))
ut = FFT1D(u)
exact[0,:] = np.pi*np.cos(np.pi*x)*u#解析一阶导
exact[1,:] = np.pi*np.pi*(np.cos(np.pi*x)**2 - np.sin(np.pi*x))*u
exact[2,:] = np.pi**3 * np.cos(np.pi*x)*(np.cos(np.pi*x)**2 - 3*np.sin(np.pi*x)-1)*u
fourier[0,:] = inverseFFT1D(complex(0,1)*k*ut).real#谱方法一阶导
fourier[1,:] = inverseFFT1D((complex(0,1)*k)**2*ut).real
fourier[2,:] = inverseFFT1D((complex(0,1)*k)**3*ut).real
error[:,:] = exact[:,:] - fourier[:,:]