import numpy as np
from scipy.integrate import solve_bvp, quad
import matplotlib.pyplot as plt

alpha = 0.0072973525693

def ode(r, y):
    f, fp = y
    sin_f = np.sin(f)
    cos_f = np.cos(f)
    sin2 = sin_f*sin_f
    denom = r*r + 2*sin2
    term1 = 2*sin_f*cos_f*(1 - fp*fp + sin2/(r*r))
    term2 = (alpha/2)*r*r*sin_f
    term3 = -2*r*fp
    fpp = (term1 + term2 + term3)/denom
    return np.vstack((fp, fpp))

def bc(ya, yb):
    return np.array([ya[0] - 3*np.pi, yb[0]])

r = np.logspace(-4, 2, 1000)
f_init = 3*np.pi * (1 - r/100)
fp_init = -3*np.pi/100 * np.ones_like(r)
y_init = np.vstack((f_init, fp_init))

sol = solve_bvp(ode, bc, r, y_init, tol=1e-6, max_nodes=50000, verbose=2)

if sol.success:
    print("BVP успешно решено")
    
    # Вычисление интегралов
    def integrand_I2(r):
        y = sol.sol(r)
        return r*r*y[1]**2 + 2*np.sin(y[0])**2
    
    def integrand_I4(r):
        y = sol.sol(r)
        sin_f = np.sin(y[0])
        sin_f_r = sin_f / r if r > 1e-8 else -y[1]
        return sin_f**2 * (2*y[1]**2 + sin_f_r**2)
    
    def integrand_I0(r):
        y = sol.sol(r)
        return (1 - np.cos(y[0])) * r*r
    
    r_min = 1e-4
    r_max = 100.0
    I2, err2 = quad(integrand_I2, r_min, r_max, limit=2000, epsabs=1e-12, epsrel=1e-12)
    I4, err4 = quad(integrand_I4, r_min, r_max, limit=2000, epsabs=1e-12, epsrel=1e-12)
    I0, err0 = quad(integrand_I0, r_min, r_max, limit=2000, epsabs=1e-12, epsrel=1e-12)
    
    alpha_calc = (I4 - I2) / (3 * I0)
    
    print("\n" + "="*60)
    print("РЕЗУЛЬТАТЫ ДЛЯ ПРОТОНА")
    print("="*60)
    print(f"I2 = {I2:.10f} ± {err2:.2e}")
    print(f"I4 = {I4:.10f} ± {err4:.2e}")
    print(f"I0 = {I0:.10f} ± {err0:.2e}")
    print(f"Δ = I4 - I2 = {I4 - I2:.10f}")
    print(f"α из интегралов = {alpha_calc:.10f}")
    print(f"Использованное α = {alpha:.10f}")
    print(f"Отклонение = {abs(alpha_calc - alpha):.2e}")
    print("="*60)
    
    # Построение графика
    r_plot = np.logspace(-4, 2, 1000)
    f_plot = sol.sol(r_plot)[0]
    plt.plot(r_plot, f_plot)
    plt.xscale('log')
    plt.xlabel(r'$\rho$')
    plt.ylabel(r'$f(\rho)$')
    plt.title('Профиль протона (трилистник)')
    plt.grid()
    plt.show()
    
else:
    print("Решение не найдено")
