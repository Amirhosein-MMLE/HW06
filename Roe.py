import numpy as np
import matplotlib.pyplot as plt

def flux(u):
    """
    Physical flux function: f(u) = u^2 / 2.
    """
    return u**2 / (u**2 + (1 - u)**2)

def roe_average(uL, uR):
    """
    Compute the Roe average speed at the interface.
    """
    if uL == uR:
        return uL
    else:
        return (flux(uR) - flux(uL)) / (uR - uL)

def roe_method(u0, dx, dt, t_end):
    """
    Roe's method for solving the nonlinear conservation law.
    u0: Initial condition array
    dx: Spatial grid size
    dt: Time step
    t_end: Final time
    """
    u = u0.copy()
    N = len(u)
    steps = int(t_end / dt)  # Total number of time steps

    for n in range(steps):
        u_new = u.copy()
        for i in range(1, N-1):
            # Compute Roe average speed at the interface
            a_half = roe_average(u[i], u[i+1])
            
            # Compute fluxes at the interfaces
            F_plus = 0.5 * (flux(u[i]) + flux(u[i+1])) - 0.5 * abs(a_half) * (u[i+1] - u[i])
            F_minus = 0.5 * (flux(u[i-1]) + flux(u[i])) - 0.5 * abs(roe_average(u[i-1], u[i])) * (u[i] - u[i-1])

            # Update the solution using Roe's method
            u_new[i] = u[i] - (dt / dx) * (F_plus - F_minus)
        
        u = u_new.copy()

    return u

# Domain and discretization
x_min, x_max = -5, 5
N = 1000  # Number of grid points
dx = (x_max - x_min) / N
x = np.linspace(x_min, x_max, N)

# Initial condition
u0 = np.zeros(N)
for i in range(N):
    if -1 <= x[i] < 0:
        u0[i] = -1
    elif 0 <= x[i] <= 1:
        u0[i] = 1

# Time parameters
t_end = 2.0  # Final time
dt = 0.9 * dx / max(abs(u0) + 1e-10)  # CFL condition

# Solve using Roe's method
u_roe = roe_method(u0, dx, dt, t_end)

# Plot the solution
plt.figure(figsize=(8, 5))
plt.plot(x, u_roe, label="Roe's Method at t=2s", color='g')
plt.title("Solution of Nonlinear Conservation Law using Roe's Method")
plt.xlabel("x")
plt.ylabel("u(x, t=2)")
plt.grid(True)
plt.legend()
plt.show()
