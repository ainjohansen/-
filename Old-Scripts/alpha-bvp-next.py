import numpy as np
from scipy.integrate import solve_bvp, quad
import matplotlib.pyplot as plt

# ============================================================
# 1. ОДУ С НЕИЗВЕСТНОЙ ALPHA
# ============================================================

def ode_system(r, y):
    f = y[0]
    fp = y[1]
    alpha = y[2][0]  # глобальный параметр

    sin_f = np.sin(f)
    cos_f = np.cos(f)
    sin2 = sin_f**2

    denom = r**2 + 2 * sin2

    term1 = 2 * sin_f * cos_f * (1 - fp**2 + sin2 / r**2)
    term2 = (alpha / 2) * r**2 * sin_f
    term3 = -2 * r * fp

    fpp = (term1 + term2 + term3) / denom

    return np.vstack((fp, fpp, np.zeros_like(r)))  # alpha' = 0


# ============================================================
# 2. ГРАНИЧНЫЕ УСЛОВИЯ
# ============================================================

def bc(ya, yb):
    f0, fp0, alpha0 = ya
    fR, fpR, alphaR = yb

    return np.array([
        f0 - np.pi,     # ядро
        fR - 0.0,       # вакуум
        fpR             # затухание хвоста
    ])


# ============================================================
# 3. НАЧАЛЬНОЕ ПРИБЛИЖЕНИЕ
# ============================================================

r_min = 1e-4
r_max = 100
r = np.logspace(np.log10(r_min), np.log10(r_max), 800)

# анзац
alpha_guess = 0.007
A = 0.8
B = np.sqrt(alpha_guess)

u = (A / r) * np.exp(-B * r)
f_guess = 2 * np.arctan(u)
fp_guess = (2 * u / (1 + u**2)) * (-1/r - B)

alpha_arr = np.full_like(r, alpha_guess)

y_guess = np.vstack((f_guess, fp_guess, alpha_arr))


# ============================================================
# 4. ИТЕРАЦИЯ С ЛОГОМ
# ============================================================

print("\n=== START EIGENVALUE SEARCH ===\n")

sol = solve_bvp(
    ode_system,
    bc,
    r,
    y_guess,
    tol=1e-6,
    max_nodes=50000,
    verbose=2
)

# ============================================================
# 5. ПРОВЕРКА
# ============================================================

if not sol.success:
    print("❌ НЕ СОШЛОСЬ:", sol.message)
    exit()

print("\n✔ СХОДИМОСТЬ ДОСТИГНУТА")
alpha_found = sol.y[2][0]
print(f"НАЙДЕННАЯ alpha = {alpha_found:.10f}")


# ============================================================
# 6. ИНТЕГРАЛЫ
# ============================================================

def I2_integrand(r):
    y = sol.sol(r)
    return r**2 * y[1]**2 + 2 * np.sin(y[0])**2

def I4_integrand(r):
    y = sol.sol(r)
    sin_f = np.sin(y[0])
    sin_f_r = sin_f / r if r > 1e-8 else -y[1]
    return sin_f**2 * (2 * y[1]**2 + sin_f_r**2)

def I0_integrand(r):
    y = sol.sol(r)
    return (1 - np.cos(y[0])) * r**2


I2, _ = quad(I2_integrand, r_min, r_max, limit=500)
I4, _ = quad(I4_integrand, r_min, r_max, limit=500)
I0, _ = quad(I0_integrand, r_min, r_max, limit=500)

alpha_check = (I4 - I2) / (3 * I0)

print("\n--- ПРОВЕРКА ДЕРРИКА ---")
print(f"I2 = {I2:.6f}")
print(f"I4 = {I4:.6f}")
print(f"I0 = {I0:.6f}")
print(f"alpha (из баланса) = {alpha_check:.10f}")

print("\nΔalpha =", abs(alpha_check - alpha_found))


# ============================================================
# 7. ВИЗУАЛИЗАЦИЯ
# ============================================================

r_plot = np.linspace(0.001, 10, 1000)
y_plot = sol.sol(r_plot)

plt.figure(figsize=(8,5))
plt.plot(r_plot, y_plot[0], label="f(r)")
plt.title(f"Eigen-solution, alpha={alpha_found:.6f}")
plt.legend()
plt.grid()
plt.show()
