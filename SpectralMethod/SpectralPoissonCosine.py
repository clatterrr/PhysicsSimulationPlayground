import numpy as np
import scipy.fftpack as fft
import matplotlib.pyplot as pl

# Chose between the original Leveque function or a modified one
# in which the Neumann boundary conditions are zero
zero_bc = False

# Plot gradient
plot_grad_u = False

# RHS function
def rhs_leveque(x):
    a = 0.5
    phi = 20. * np.pi * x ** 3
    phip = 60. * np.pi * x ** 2
    phipp = 120. * np.pi * x
    if zero_bc:
        return 6 - 12 * x + a * phipp * np.sin(phi) + a * phip * phip * np.cos(phi)
    else:
        return -20. + a * phipp * np.cos(phi) - a * phip * phip * np.sin(phi)


# Neumann boundary conditions
def bc_leveque(x):
    a = 0.5
    phi = 20. * np.pi * x ** 3
    phip = 60. * np.pi * x ** 2
    if zero_bc:
        return 6. * x * (1 - x) + a * phip * np.sin(phi)
    else:
        return 12. - 20. * x + a * phip * np.cos(phi)


# Solution
def sol_leveque(x):
    a = 0.5
    phi = 20. * np.pi * x ** 3
    if zero_bc:
        return 3. * x * x - 2. * x * x * x - a * np.cos(phi)
    else:
        return 1. + 12. * x - 10. * x * x + a * np.sin(phi)




# Domain and discretization
# Use dct_norm = None so that we can normalize ourselves
nx = 512
xl = 0.
xh = 1.
dct_type = 2
dct_norm = None

dx = (xh - xl) / (nx)
x = np.linspace(xl + 0.5 * dx, xh - 0.5 * dx, nx)
xlb, xhb = xl, xh

# Create RHS with boundary conditions
rhs = rhs_leveque(x) * dx * dx
rhs[0] += dx * bc_leveque(xlb)
rhs[-1] -= dx * bc_leveque(xhb)

# FFT RHS
rhs_p = fft.dct(rhs, type=dct_type, norm=dct_norm)
rhs_p *= 1. / nx

# Construct solution in Fourier space
l = np.arange(0, nx)
cosl = np.cos(np.pi * l / nx)
sol_p = rhs_p / (2. * (cosl - 1.))

# Normalization is directly obtainable in the case of cosine transform (type 2)
sol_p[0] = sol_leveque(0) - np.sum(sol_p[1:])
sol_p[1:] *= 0.5

# Inverse FFT to get real solution
sol = fft.idct(sol_p, type=dct_type, norm=dct_norm)

# Plot solution
fig1 = pl.figure(figsize=(8, 6))
pl.plot(sol, '.', label='DCT solver')
pl.plot(sol_leveque(x), label='analytic')
pl.legend()
fig1.savefig('sol_dct_leveque.png')

if plot_grad_u:
    fig2 = pl.figure(figsize=(8, 6))
    pl.plot(np.gradient(sol, dx), '.', label='grad DCTsolver')
    pl.plot(bc_leveque(x), label='analytic')
    pl.plot(np.gradient(sol_leveque(x), dx), '--', label='grad anal. sol.')
    pl.legend()
    fig2.savefig('bc_dct_leveque.png')