# Code modified from GeeksforGeeks and numpy.org
import numpy as np
from scipy.integrate import solve_ivp

# Parameters
a = 100  # Starting x-coordinate (miles east of airport)
w = 44   # Wind speed (towards north)
v_a = 88  # Plane's velocity in the air

# dy/dx
def dydx(x, y):
    k = w / v_a
    if abs(x) < 1e-6:
        return 0
    ratio = y / x
    return ratio - k * np.sqrt(1 + ratio**2)

# Runge_kutta
def runge_kutta(f, x0, y0, h, x_end):
    x_values = [x0]
    y_values = [y0]
    x = x0
    y = y0

    while x > x_end:
        dy_val = f(x, y)
        if not np.isfinite(dy_val):
            print(f"RK stopped: non-finite dy/dx at x = {x:.2f}, y = {y:.2f}")
            break

        h_adjusted = h if abs(dy_val) < 1000 else max(h / 10, 0.001)
        if x - h_adjusted < x_end:
            h_adjusted = x - x_end

        k1 = h_adjusted * dy_val
        k2 = h_adjusted * f(x - h_adjusted/2, y - k1/2)
        k3 = h_adjusted * f(x - h_adjusted/2, y - k2/2)
        k4 = h_adjusted * f(x - h_adjusted, y - k3)

        y = y - (k1 + 2*k2 + 2*k3 + k4) / 6
        x = x - h_adjusted

        x_values.append(x)
        y_values.append(y)

        if abs(y) > 1e6:
            print(f"RK stopped: y too large at x = {x:.2f}, y = {y:.2e}")
            break

    return x_values, y_values

# Euler method
def euler_method(f, x0, y0, h, x_end):
    x_values = [x0]
    y_values = [y0]
    x = x0
    y = y0

    while x > x_end:
        h_step = h if x - h > x_end else x - x_end
        y = y - h_step * f(x, y)
        x = x - h_step
        x_values.append(x)
        y_values.append(y)

        if not np.isfinite(y) or abs(y) > 1e6:
            print(f"Euler stopped at x = {x:.2f}, y = {y:.2e}")
            break

    return x_values, y_values

# solve_ivp solution
def solve_with_scipy():
    def system(t, vars):
        x, y = vars
        r = np.sqrt(x**2 + y**2)
        dx_dt = -v_a * x / r
        dy_dt = -v_a * y / r + w
        return [dx_dt, dy_dt]

    def reach_airport(t, vars):
        x, y = vars
        return x**2 + y**2 - 0.01
    reach_airport.terminal = True
    reach_airport.direction = -1

    est_time = 3 * a / v_a
    sol = solve_ivp(
        system, 
        (0, est_time), 
        [a, 0], 
        method='RK45', 
        events=reach_airport,
        dense_output=True,
        rtol=1e-6, 
        atol=1e-8
    )
    return sol.t, sol.y[0], sol.y[1]

# Test methods
def test_trajectory_methods():
    h = 0.01  # Step size
    x_rk, y_rk = runge_kutta(dydx, a, 0, h, 0)
    rk_final = y_rk[-1]
    x_euler, y_euler = euler_method(dydx, a, 0, h, 0)
    euler_final = y_euler[-1]
    _, x_scipy, y_scipy = solve_with_scipy()
    scipy_final = y_scipy[-1]

    # Print final landing results
    print("\nFinal Landing Positions:")
    print(f"  RK lands at y ≈ {rk_final:.4f}")
    print(f"  Euler lands at y ≈ {euler_final:.4f}")
    print(f"  Scipy lands at y ≈ {scipy_final:.4f}")

    # Print trajectory points
    print("\nRK Trajectory Points:")
    for i in np.linspace(0, len(x_rk)-1, 10, dtype=int):
        print(f"x = {x_rk[i]:.2f}, y = {y_rk[i]:.4f}")

test_trajectory_methods()
