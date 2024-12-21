import numpy as np
import matplotlib.pyplot as plt

def flux(u):
    """
    Flux function for the conservation law: f(u) = u^2 / 2.
    """
    return u**2 / (u**2 + (1 - u)**2)

def godunov_flux(uL, uR):
    """
    Compute the Godunov flux for a Riemann problem at the interface.
    uL: Left state
    uR: Right state
    """
    if uL <= uR:  # Rarefaction
        return min(flux(uL), flux(uR))
    else:  # Shock
        return max(flux(uL), flux(uR))

def godunov_method(u0, dx, dt, t_end):
    """
    Godunov finite volume method for the nonlinear conservation law.
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
            # Compute Godunov fluxes at interfaces
            F_right = godunov_flux(u[i], u[i+1])
            F_left = godunov_flux(u[i-1], u[i])

            # Update using the finite volume scheme
            u_new[i] = u[i] - (dt / dx) * (F_right - F_left)
        u = u_new.copy()

    return u

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

def lax_friedrichs_method(u0, dx, dt, t_end):
    """
    Lax-Friedrichs method for solving the nonlinear conservation law.
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
            # Lax-Friedrichs update formula
            u_new[i] = 0.5 * (u[i-1] + u[i+1]) - (dt / (2 * dx)) * (flux(u[i+1]) - flux(u[i-1]))
        u = u_new.copy()

    return u

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
dt = 1.2 * dx / max(abs(u0) + 1e-10)  # CFL condition
CFL = 1.2

# Solve using different methods
u_godunov = godunov_method(u0, dx, dt, t_end)
u_roe = roe_method(u0, dx, dt, t_end)
u_lax_friedrichs = lax_friedrichs_method(u0, dx, dt, t_end)
u_local_lax_friedrichs = local_lax_friedrichs(u0, x, dx, CFL, t_end)

# Plot the solutions
plt.figure(figsize=(10, 6))
plt.plot(x, u_godunov, label="Godunov Method", color='r',linewidth=2, markersize=2, linestyle='--')
plt.plot(x, u_roe, label="Roe's Method", color='g')
plt.plot(x, u_lax_friedrichs, label="Lax-Friedrichs Method", color='b',linewidth=3,linestyle='--')
plt.plot(x, u_local_lax_friedrichs, label="Local Lax-Friedrichs Method", color='m')
plt.title("Comparison of Numerical Methods for Nonlinear Conservation Law")
plt.xlabel("x")
plt.ylabel("u(x, t=2)")
plt.grid(True)
plt.legend()
plt.show()
