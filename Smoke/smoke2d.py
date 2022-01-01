import numpy as np
import math
hbar = 0.1
grid_row_num = 4
phase = np.zeros((grid_row_num,grid_row_num))
psi1 = np.zeros((grid_row_num,grid_row_num),dtype = complex)
psi2 = np.zeros((grid_row_num,grid_row_num),dtype = complex)
for i in range(grid_row_num):
    for j in range(grid_row_num):
        phase[i,j] = i * -5
        psi1[i,j] = np.exp(1j * phase[i,j])
        psi2[i,j] = 0.01 * np.exp(1j * phase[i,j])
        psinorm = np.sqrt(psi1[i,j]*psi1[i,j] + psi2[i,j]*psi2[i,j])
        psi1[i,j] /= psinorm
        psi2[i,j] /= psinorm
        
grid_vel = np.zeros((grid_row_num,grid_row_num,2))
dx = 1
dy = 1
grid_div = np.zeros((grid_row_num,grid_row_num))
for i in range(grid_row_num):
    for j in range(grid_row_num):
        ip = (i + 1) % grid_row_num
        jp = (j + 1) % grid_row_num
        psi1_conj = - psi1[i,j]
        psi2_conj = - psi2[i,j]
        
        cx = psi1_conj * psi1[ip,j] +  psi2_conj * psi2[ip,j]
        cy = psi1_conj * psi1[i,jp] +  psi2_conj * psi2[i,jp]
        
        vx = math.atan2(cx.imag, cx.real)
        vy = math.atan2(cy.imag, cy.real)
        grid_vel[i,j] = np.array([cx,cy])
for i in range(grid_row_num):
    for j in range(grid_row_num):     
        im = (i - 1 + grid_row_num) % grid_row_num
        jm = (j - 1 + grid_row_num) % grid_row_num
        grid_div[i,j] = (grid_vel[i,j,0] - grid_vel[im,j,0])/dx/dx
        grid_div[i,j] += (grid_vel[i,j,0] - grid_vel[i,jm,0])/dy/dy