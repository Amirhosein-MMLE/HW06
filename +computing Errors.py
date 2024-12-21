import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Flux function
def flux(u):
    return u**2 / (u**2 + (1 - u)**2)

# Godunov flux
def godunov_flux(uL, uR):
    if uL <= uR:  # Rarefaction
        return min(flux(uL), flux(uR))
    else:  # Shock
        return max(flux(uL), flux(uR))

# Godunov method
def godunov_method(u0, dx, dt, t_end):
    u = u0.copy()
    N = len(u)
    steps = int(t_end / dt)
    
    for n in range(steps):
        u_new = u.copy()
        for i in range(1, N-1):
            F_right = godunov_flux(u[i], u[i+1])
            F_left = godunov_flux(u[i-1], u[i])
            u_new[i] = u[i] - (dt / dx) * (F_right - F_left)
        u = u_new.copy()
    
    return u

# Roe's method
def roe_average(uL, uR):
    if uL == uR:
        return uL
    else:
        return (flux(uR) - flux(uL)) / (uR - uL)

def roe_method(u0, dx, dt, t_end):
    u = u0.copy()
    N = len(u)
    steps = int(t_end / dt)

    for n in range(steps):
        u_new = u.copy()
        for i in range(1, N-1):
            a_half = roe_average(u[i], u[i+1])
            F_plus = 0.5 * (flux(u[i]) + flux(u[i+1])) - 0.5 * abs(a_half) * (u[i+1] - u[i])
            F_minus = 0.5 * (flux(u[i-1]) + flux(u[i])) - 0.5 * abs(roe_average(u[i-1], u[i])) * (u[i] - u[i-1])
            u_new[i] = u[i] - (dt / dx) * (F_plus - F_minus)
        u = u_new.copy()

    return u

# Lax-Friedrichs method
def lax_friedrichs_method(u0, dx, dt, t_end):
    u = u0.copy()
    N = len(u)
    steps = int(t_end / dt)
    
    for n in range(steps):
        u_new = u.copy()
        for i in range(1, N-1):
            u_new[i] = 0.5 * (u[i-1] + u[i+1]) - (dt / (2 * dx)) * (flux(u[i+1]) - flux(u[i-1]))
        u = u_new.copy()
    
    return u

# Local Lax-Friedrichs method
def local_lax_friedrichs(u0, x, dx, CFL, t_end):
    u = u0.copy()
    N = len(u)
    t = 0.0
    
    while t < t_end:
        wave_speeds = np.abs(u)
        dt = CFL * dx / np.max(wave_speeds + 1e-10)
        if t + dt > t_end:
            dt = t_end - t
        
        u_new = u.copy()
        for i in range(1, N-1):
            u_new[i] = 0.5 * (u[i-1] + u[i+1]) - (dt / (2 * dx)) * (flux(u[i+1]) - flux(u[i-1]))
        
        u = u_new.copy()
        t += dt
    
    return u

# Compute error
def compute_error(numerical, exact, dx, norm='L2'):
    error = numerical - exact
    if norm == 'L1':
        return np.sum(np.abs(error)) * dx
    elif norm == 'L2':
        return np.sqrt(np.sum(error**2) * dx)
    elif norm == 'Linf':
        return np.max(np.abs(error))
    else:
        raise ValueError("Unsupported norm type. Choose from 'L1', 'L2', or 'Linf'.")

# Domain and discretization
x_min, x_max = -5, 5
N = 1000
dx = (x_max - x_min) / N
x = np.linspace(x_min, x_max, N)

# Initial condition
u0 = np.zeros(N)
for i in range(N):
    if -1 <= x[i] < 0:
        u0[i] = -1
    elif 0 <= x[i] <= 1:
        u0[i] = 1

# Exact solution (if available, for now assuming initial profile remains unchanged)
u_exact = u0.copy()

# Time parameters
t_end = 2.0
dt = 0.9 * dx / max(abs(u0) + 1e-10)
CFL = 0.9

# Solve using methods
u_godunov = godunov_method(u0, dx, dt, t_end)
u_roe = roe_method(u0, dx, dt, t_end)
u_lax_friedrichs = lax_friedrichs_method(u0, dx, dt, t_end)
u_local_lax_friedrichs = local_lax_friedrichs(u0, x, dx, CFL, t_end)

# Plot results
plt.figure(figsize=(8, 5))
plt.plot(x, u_godunov, label="Godunov Method", linestyle='--', color='r')
plt.plot(x, u_roe, label="Roe's Method", linestyle='-', color='g')
plt.plot(x, u_lax_friedrichs, label="Lax-Friedrichs Method", linestyle=':', color='b')
plt.plot(x, u_local_lax_friedrichs, label="Local Lax-Friedrichs Method", linestyle='-.', color='purple')
plt.title("Comparison of Numerical Methods for Nonlinear Conservation Law")
plt.xlabel("x")
plt.ylabel("u(x, t=2)")
plt.legend()
plt.grid()
plt.show()

# Compute errors
error_data = {
    'Method': ['Godunov', "Roe's Method", 'Lax-Friedrichs', 'Local Lax-Friedrichs'],
    'L1 Error': [compute_error(u_godunov, u_exact, dx, norm='L1'),
                 compute_error(u_roe, u_exact, dx, norm='L1'),
                 compute_error(u_lax_friedrichs, u_exact, dx, norm='L1'),
                 compute_error(u_local_lax_friedrichs, u_exact, dx, norm='L1')],
    'L2 Error': [compute_error(u_godunov, u_exact, dx, norm='L2'),
                 compute_error(u_roe, u_exact, dx, norm='L2'),
                 compute_error(u_lax_friedrichs, u_exact, dx, norm='L2'),
                 compute_error(u_local_lax_friedrichs, u_exact, dx, norm='L2')],
    'Linf Error': [compute_error(u_godunov, u_exact, dx, norm='Linf'),
                   compute_error(u_roe, u_exact, dx, norm='Linf'),
                   compute_error(u_lax_friedrichs, u_exact, dx, norm='Linf'),
                   compute_error(u_local_lax_friedrichs, u_exact, dx, norm='Linf')],
}

# Save errors to Excel
df = pd.DataFrame(error_data)
df.to_excel('numerical_errors.xlsx', index=False)
print("Errors have been saved to numerical_errors.xlsx!")
import os
print("File saved at:", os.path.abspath('numerical_errors.xlsx'))
