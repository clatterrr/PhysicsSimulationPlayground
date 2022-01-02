import numpy as np
import scipy.fftpack as fft
import matplotlib.pyplot as pl

# Domain and discretization
nx = 512
xl = 0.
xh = 1.

dx = (xh - xl) / nx
x = np.linspace(xl + dx, xh - dx, nx)
xlb, xhb = xl, xh


# RHS function
def rhs_leveque(x):
    a = 0.5
    phi = 20. * np.pi * x ** 3
    phip = 60. * np.pi * x ** 2
    phipp = 120. * np.pi * x
    return -20. + a * phipp * np.cos(phi) - a * (phip * phip) * np.sin(phi)


# Dirichlet boundary conditions (solution)
def bc_leveque(x):
    a = 0.5
    phi = 20. * np.pi * x ** 3
    return 1. + 12. * x - 10. * x * x + a * np.sin(phi)

# Create RHS with boundary conditions
rhs = rhs_leveque(x) * dx * dx
rhs[0] -= bc_leveque(xlb)
rhs[-1] -= bc_leveque(xhb)

# FFT RHS
rhs_p = fft.dst(rhs, type=1, norm='ortho')
# Construct solution in Fourier space
l = np.arange(1, nx + 1)
cosl = np.cos(np.pi * l / nx)
sol_p = rhs_p / (2. * (cosl - 1.))

# Inverse FFT to get real solution
sol = fft.idst(sol_p, type=1, norm='ortho')
# Plot solution
fig1 = pl.figure(figsize=(8, 6))
pl.plot(sol, label='DST solver')
pl.plot(bc_leveque(x), label='analytic')
pl.legend()
# fig1.savefig('sol_fft_leveque.png')
