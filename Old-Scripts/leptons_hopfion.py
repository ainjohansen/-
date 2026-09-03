import numpy as np
from scipy.integrate import solve_bvp, quad
import matplotlib.pyplot as plt

# ============================================================
# ПАРАМЕТРЫ
# ============================================================
alpha = 0.0072973525693   # фиксированная α нашей Вселенной
r_min = 1e-4
r_max = 120.0
n_nodes = 1000

# Числа навивки для лептонов
n_values = [1, 6, 15]
energies = {}

# ============================================================
# ОПРЕДЕЛЕНИЕ СИСТЕМЫ ОДУ С УЧЁТОМ n
# ============================================================
# В радиальной редукции эффект числа навивки n проявляется
# в дополнительном центробежном члене ~ n^2 / r^2.
# Модифицируем уравнение, добавив n^2 sin^2 f в знаменатель.
def make_ode(n):
    def ode_system(r, y):
        f, fp = y
        sin_f = np.sin(f)
        cos_f = np.cos(f)
        sin2 = sin_f**2
        
        # Добавка от навивки: n^2 * sin^2 f
        denom = r**2 + 2 * sin2 + n**2 * sin2
        
        term1 = 2 * sin_f * cos_f * (1 - fp**2 + sin2/r**2)
        term2 = (alpha / 2) * r**2 * sin_f
        term3 = -2 * r * fp
        
        fpp = (term1 + term2 + term3) / denom
        return np.vstack((fp, fpp))
    return ode_system

def boundary_conditions(ya, yb):
    return np.array([ya[0] - np.pi, yb[0] - 0.0])

# ============================================================
# РЕШЕНИЕ ДЛЯ КАЖДОГО n
# ============================================================
for n in n_values:
    print(f"\n--- РЕШЕНИЕ ДЛЯ n = {n} ---")
    r_nodes = np.logspace(np.log10(r_min), np.log10(r_max), n_nodes)
    
    # Начальное приближение (анзац)
    A_guess = 0.8168
    B_guess = np.sqrt(alpha)
    u_guess = (A_guess / r_nodes) * np.exp(-B_guess * r_nodes)
    f_guess = 2 * np.arctan(u_guess)
    fp_guess = (2 * u_guess / (1 + u_guess**2)) * (-1/r_nodes - B_guess)
    y_guess = np.vstack((f_guess, fp_guess))
    
    sol = solve_bvp(make_ode(n), boundary_conditions, r_nodes, y_guess, tol=1e-8, max_nodes=50000)
    
    if sol.success:
        # Интегралы
        def I2_int(r):
            y = sol.sol(r)
            return r**2 * y[1]**2 + 2 * np.sin(y[0])**2
        def I4_int(r):
            y = sol.sol(r)
            sin_f = np.sin(y[0])
            sin_f_r = sin_f / r if r > 1e-8 else -y[1]
            return sin_f**2 * (2 * y[1]**2 + sin_f_r**2)
        def I0_int(r):
            y = sol.sol(r)
            return (1 - np.cos(y[0])) * r**2
        
        I2, _ = quad(I2_int, r_min, r_max, limit=300)
        I4, _ = quad(I4_int, r_min, r_max, limit=300)
        I0, _ = quad(I0_int, r_min, r_max, limit=300)
        E = I2 + I4 + 3 * alpha * I0
        energies[n] = E
        print(f"I2 = {I2:.6f}, I4 = {I4:.6f}, I0 = {I0:.6f}")
        print(f"Энергия E = {E:.6f}")
    else:
        print(f"Решение для n={n} не сошлось.")
        energies[n] = np.nan

# ============================================================
# СРАВНЕНИЕ С ПРЕДСКАЗАНИЕМ n^3
# ============================================================
print("\n" + "="*60)
print("СРАВНЕНИЕ ОТНОШЕНИЙ ЭНЕРГИЙ С ПРЕДСКАЗАНИЕМ n^3")
print("="*60)
E1 = energies[1]
for n in n_values:
    if not np.isnan(energies[n]):
        ratio = energies[n] / E1
        pred = n**3
        print(f"n={n:2d}: E/E1 = {ratio:.4f}, n^3 = {pred:.4f}, откл. = {abs(ratio-pred)/pred*100:.2f}%")
