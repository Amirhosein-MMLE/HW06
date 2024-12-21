import numpy as np
import matplotlib.pyplot as plt

def flux(u):
    """
    Flux function for the conservation law: f(u) = u^2 / 2.
    """
    return u**2 / (u**2 + (1 - u)**2)

def local_lax_friedrichs(u0, x, dx, CFL, t_end):
    """
    Local Lax-Friedrichs method for the nonlinear conservation law.
    u0: Initial condition array
    x: Spatial grid
    dx: Spatial grid size
    CFL: CFL condition number
    t_end: Final time
    """
    u = u0.copy()
    N = len(u)
    t = 0.0
    
    while t < t_end:
        # Calculate local wave speeds and time step
        wave_speeds = np.abs(u)
        dt = CFL * dx / np.max(wave_speeds + 1e-10)  # Small epsilon to avoid division by zero
        if t + dt > t_end:
            dt = t_end - t  # Adjust final time step
        
        u_new = u.copy()
        for i in range(1, N-1):
            # Local Lax-Friedrichs update
            u_new[i] = 0.5 * (u[i-1] + u[i+1]) - (dt / (2 * dx)) * (flux(u[i+1]) - flux(u[i-1]))
        
        u = u_new.copy()
        t += dt
    
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
CFL = 0.9    # CFL condition number

# Solve using Local Lax-Friedrichs method
u_local_lax_friedrichs = local_lax_friedrichs(u0, x, dx, CFL, t_end)

# Plot the solution
plt.figure(figsize=(8, 5))
plt.plot(x, u_local_lax_friedrichs, label="Local Lax-Friedrichs Method at t=2s", color='b')
plt.title("Solution of Nonlinear Conservation Law using Local Lax-Friedrichs Method")
plt.xlabel("x")
plt.ylabel("u(x, t=2)")
plt.grid(True)
plt.legend()
plt.show()
