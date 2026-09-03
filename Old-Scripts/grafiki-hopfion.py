"""
Точный скрипт для генерации рисунков к статье
"Электрон как топологический хопфион: конечная энергия и происхождение постоянной тонкой структуры"

Выполняет:
1. Решение BVP для профиля f(r) с α = 1/137.036
2. Построение f(r) и f'(r) → profile.png
3. Вычисление эталонных интегралов и проверка баланса I4 - I2 = 3α I0
4. Масштабирование профиля и построение энергетической ямы E(a) → energy_well.png
5. Сохранение профиля в bvp_exact_profile.npz для последующего использования
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_bvp, simpson
from scipy.interpolate import interp1d
import os

# ----------------------------------------------------------------------
# Параметры
# ----------------------------------------------------------------------
alpha = 1.0 / 137.035999084   # постоянная тонкой структуры
r_min, r_max = 1e-4, 120.0    # интервал интегрирования
n_nodes = 1000                # число узлов начальной сетки

# ----------------------------------------------------------------------
# 1. Определение системы ОДУ и граничных условий
# ----------------------------------------------------------------------
def ode_system(r, y):
    f, fp = y
    sin_f = np.sin(f)
    cos_f = np.cos(f)
    sin2 = sin_f**2
    denom = r**2 + 2 * sin2
    term1 = 2 * sin_f * cos_f * (1 - fp**2 + sin2/r**2)
    term2 = (alpha / 2) * r**2 * sin_f
    term3 = -2 * r * fp
    fpp = (term1 + term2 + term3) / denom
    return np.vstack((fp, fpp))

def boundary_conditions(ya, yb):
    return np.array([ya[0] - np.pi, yb[0] - 0.0])

# ----------------------------------------------------------------------
# 2. Начальное приближение и решение BVP
# ----------------------------------------------------------------------
print("Решение краевой задачи для хопфиона...")
r_nodes = np.logspace(np.log10(r_min), np.log10(r_max), n_nodes)

A_guess = 0.8168
B_guess = np.sqrt(alpha)
u_guess = (A_guess / r_nodes) * np.exp(-B_guess * r_nodes)
f_guess = 2 * np.arctan(u_guess)
fp_guess = (2 * u_guess / (1 + u_guess**2)) * (-1/r_nodes - B_guess)
y_guess = np.vstack((f_guess, fp_guess))

sol = solve_bvp(ode_system, boundary_conditions, r_nodes, y_guess,
                tol=1e-8, max_nodes=50000)

if not sol.success:
    raise RuntimeError("BVP не сошёлся: " + sol.message)

print("BVP успешно решён.")

# Извлекаем решение на более мелкой сетке для точных интегралов
r_fine = np.linspace(r_min, r_max, 5000)
f_fine = sol.sol(r_fine)[0]
fp_fine = sol.sol(r_fine)[1]

# ----------------------------------------------------------------------
# 3. Сохранение профиля для будущего использования
# ----------------------------------------------------------------------
np.savez('bvp_exact_profile.npz', r=r_fine, f=f_fine, fp=fp_fine)
print("Профиль сохранён в 'bvp_exact_profile.npz'.")

# ----------------------------------------------------------------------
# 4. Вычисление эталонных интегралов
# ----------------------------------------------------------------------
def compute_integrals(r, f, fp):
    sin_f = np.sin(f)
    # избегаем деления на ноль в нуле
    sin_f_r = np.where(r < 1e-8, -fp, sin_f / r)

    I2_dens = r**2 * fp**2 + 2 * sin_f**2
    I4_dens = sin_f**2 * (2 * fp**2 + sin_f_r**2)
    I0_dens = (1 - np.cos(f)) * r**2

    I2 = simpson(4 * np.pi * I2_dens, x=r)
    I4 = simpson(4 * np.pi * I4_dens, x=r)
    I0 = simpson(4 * np.pi * I0_dens, x=r)

    # Топологическая плотность заряда и эффективный радиус
    rho_top = (1/(4*np.pi)) * 2 * fp * sin_f**2 / (r**2 + 1e-12)
    Q = simpson(4 * np.pi * r**2 * rho_top, x=r)
    R_eff = simpson(4 * np.pi * r**3 * rho_top, x=r) / Q

    return I2, I4, I0, R_eff

I2_0, I4_0, I0_0, Reff_0 = compute_integrals(r_fine, f_fine, fp_fine)

print("\n=== ЭТАЛОННЫЕ ИНТЕГРАЛЫ ===")
print(f"I2 = {I2_0:.6f}")
print(f"I4 = {I4_0:.6f}")
print(f"I0 = {I0_0:.6f}")
print(f"R_eff = {Reff_0:.6f}")
print(f"Баланс: (I4 - I2) = {I4_0 - I2_0:.6f}, 3α I0 = {3*alpha*I0_0:.6f}")

# ----------------------------------------------------------------------
# 5. Построение графика профиля f(r) и производной
# ----------------------------------------------------------------------
plt.figure(figsize=(8, 5))
plt.plot(r_fine, f_fine, 'b-', linewidth=2, label='$f(r)$')
plt.plot(r_fine, fp_fine, 'r--', linewidth=1.5, label="$f'(r)$")
plt.axhline(0, color='gray', linestyle=':', alpha=0.5)
plt.axhline(np.pi, color='gray', linestyle=':', alpha=0.5)
plt.xlim(0, 10)
plt.ylim(-0.5, 3.5)
plt.xlabel('$r$', fontsize=12)
plt.ylabel('$f, f\'$', fontsize=12)
plt.legend(fontsize=11)
plt.grid(alpha=0.3)
plt.title('Профиль фазы хопфиона', fontsize=14)
plt.tight_layout()
plt.savefig('profile.png', dpi=150)
plt.close()
print("Рисунок 'profile.png' сохранён.")

# ----------------------------------------------------------------------
# 6. Масштабирование и энергетическая яма
# ----------------------------------------------------------------------
def energy_scaled(a):
    """Полная энергия при масштабировании координат r -> a * r."""
    I2 = I2_0 / a
    I4 = I4_0 * a
    I0 = I0_0 * a**3
    E_c = alpha / (Reff_0 * a)
    return I2 + I4 + 3 * alpha * I0 + E_c

a_vals = np.linspace(0.5, 2.0, 200)
E_vals = np.array([energy_scaled(a) for a in a_vals])
a_opt = a_vals[np.argmin(E_vals)]

print(f"\nМинимум энергии достигается при a = {a_opt:.4f}")

plt.figure(figsize=(8, 5))
plt.plot(a_vals, E_vals, 'b-', linewidth=2)
plt.axvline(a_opt, color='red', linestyle='--', label=f'Минимум $a = {a_opt:.3f}$')
plt.xlabel('Масштабный фактор $a$', fontsize=12)
plt.ylabel('Полная энергия $E(a)$', fontsize=12)
plt.legend(fontsize=11)
plt.grid(alpha=0.3)
plt.title('Энергетическая яма фиксированного профиля электрона', fontsize=14)
plt.tight_layout()
plt.savefig('energy_well.png', dpi=150)
plt.close()
print("Рисунок 'energy_well.png' сохранён.")

print("\nГотово. Все файлы созданы.")
