import numpy as np
from scipy.integrate import solve_bvp, quad, simpson
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.optimize import minimize_scalar

# =====================================================================
# 1. ПОЛУЧЕНИЕ ТОЧНОГО ПРОФИЛЯ (BVP) — как в alpha-bvp.py
# =====================================================================
alpha_exact = 0.0072973525693

def ode_system(r, y):
    f, fp = y
    sin_f = np.sin(f)
    cos_f = np.cos(f)
    sin2 = sin_f**2
    denom = r**2 + 2 * sin2
    term1 = 2 * sin_f * cos_f * (1 - fp**2 + sin2/r**2)
    term2 = (alpha_exact / 2) * r**2 * sin_f
    term3 = -2 * r * fp
    fpp = (term1 + term2 + term3) / denom
    return np.vstack((fp, fpp))

def boundary_conditions(ya, yb):
    return np.array([ya[0] - np.pi, yb[0] - 0.0])

# Сетка и начальное приближение
r_min, r_max = 1e-4, 120.0
r_nodes = np.logspace(np.log10(r_min), np.log10(r_max), 1000)

A_guess = 0.8168
B_guess = np.sqrt(alpha_exact)
u_guess = (A_guess / r_nodes) * np.exp(-B_guess * r_nodes)
f_guess = 2 * np.arctan(u_guess)
fp_guess = (2 * u_guess / (1 + u_guess**2)) * (-1/r_nodes - B_guess)
y_guess = np.vstack((f_guess, fp_guess))

print("Решаем BVP для точного профиля...")
sol = solve_bvp(ode_system, boundary_conditions, r_nodes, y_guess, tol=1e-8, max_nodes=50000)

if not sol.success:
    print("Ошибка BVP:", sol.message)
    exit()

print("BVP успешно решён.\n")

# Интерполяция точного профиля
f_interp = interp1d(sol.x, sol.y[0], kind='cubic', fill_value="extrapolate")
fp_interp = interp1d(sol.x, sol.y[1], kind='cubic', fill_value="extrapolate")

# =====================================================================
# 2. ФУНКЦИЯ ВЫЧИСЛЕНИЯ ЭНЕРГИИ ДЛЯ ЗАДАННОГО МАСШТАБА a
# =====================================================================
def energy_for_scale(a, r_min_int=1e-4, r_max_int=100.0, n_points=5000):
    """
    Масштабируем координату: r_phys = a * r (где r - безразмерная координата эталона).
    Профиль f(r_phys) = f_interp(r_phys / a).
    Производная fp_phys = (1/a) * fp_interp(r_phys / a).
    Интегрируем по r_phys.
    """
    r_phys = np.linspace(r_min_int, r_max_int, n_points)
    r_arg = r_phys / a   # аргумент для интерполяции эталонного профиля
    f_vals = f_interp(r_arg)
    fp_vals = fp_interp(r_arg) / a

    sin_f = np.sin(f_vals)
    # избегаем деления на ноль в нуле
    sin_f_r = np.where(r_phys < 1e-8, -fp_vals, sin_f / r_phys)

    # Плотности энергии
    I2_dens = r_phys**2 * fp_vals**2 + 2 * sin_f**2
    I4_dens = sin_f**2 * (2 * fp_vals**2 + sin_f_r**2)
    I0_dens = (1 - np.cos(f_vals)) * r_phys**2

    # Интегралы по объёму (4π r^2 dr)
    I2 = simpson(4 * np.pi * I2_dens, x=r_phys)
    I4 = simpson(4 * np.pi * I4_dens, x=r_phys)
    I0 = simpson(4 * np.pi * I0_dens, x=r_phys)

    # Кулоновский вклад: используем эффективный радиус из топологической плотности заряда
    # ρ_top = (1/4π) n·(∂_x n × ∂_y n). Для сферически-симметричного приближения:
    rho_top = (1/(4*np.pi)) * sin_f**2 * fp_vals  # приближение, точнее было бы с якобианом
    # Эффективный радиус: <r> = ∫ r * ρ_top dV / ∫ ρ_top dV, но заряд Q=1
    Q = simpson(4 * np.pi * r_phys**2 * rho_top, x=r_phys)
    R_eff = simpson(4 * np.pi * r_phys**3 * rho_top, x=r_phys) / Q if abs(Q) > 1e-10 else 1.0
    E_coulomb = alpha_exact / R_eff

    total = I2 + I4 + 3 * alpha_exact * I0 + E_coulomb
    return total, I2, I4, I0, E_coulomb

# =====================================================================
# 3. ПРОВЕРКА ПРИ a=1 (ДОЛЖНО СОВПАДАТЬ С BVP)
# =====================================================================
E1, I2_1, I4_1, I0_1, Ec_1 = energy_for_scale(1.0)
print("=== ЭТАЛОННЫЕ ЗНАЧЕНИЯ ПРИ a=1 ===")
print(f"I2 = {I2_1:.6f}")
print(f"I4 = {I4_1:.6f}")
print(f"I0 = {I0_1:.6f}")
print(f"Кулоновская энергия = {Ec_1:.6f}")
print(f"Полная энергия = {E1:.6f}")
print(f"Баланс: (I4 - I2) = {I4_1 - I2_1:.6f}, 3αI0 = {3*alpha_exact*I0_1:.6f}\n")

# =====================================================================
# 4. ПОИСК МИНИМУМА ПО МАСШТАБУ a
# =====================================================================
def objective(a):
    return energy_for_scale(a)[0]

res = minimize_scalar(objective, bounds=(0.3, 3.0), method='bounded')
a_opt = res.x
E_opt = res.fun

print("=== РЕЗУЛЬТАТЫ ОПТИМИЗАЦИИ ===")
print(f"Оптимальный масштаб a = {a_opt:.6f}")
print(f"Минимальная полная энергия = {E_opt:.6f}")

# Детали при оптимальном масштабе
E_opt_full, I2_opt, I4_opt, I0_opt, Ec_opt = energy_for_scale(a_opt)
print(f"I2(a_opt) = {I2_opt:.6f}")
print(f"I4(a_opt) = {I4_opt:.6f}")
print(f"I0(a_opt) = {I0_opt:.6f}")
print(f"Кулоновская энергия = {Ec_opt:.6f}")
print(f"Баланс: (I4 - I2) = {I4_opt - I2_opt:.6f}, 3αI0 = {3*alpha_exact*I0_opt:.6f}")

# =====================================================================
# 5. ГРАФИК ЭНЕРГЕТИЧЕСКОЙ ЯМЫ
# =====================================================================
a_vals = np.linspace(0.5, 2.0, 50)
E_vals = np.array([objective(a) for a in a_vals])

plt.figure(figsize=(8,5))
plt.plot(a_vals, E_vals, 'b-', lw=2, label='Полная энергия')
plt.axvline(a_opt, color='r', linestyle='--', label=f'Минимум при a = {a_opt:.3f}')
plt.xlabel('Масштабный фактор $a$')
plt.ylabel('Энергия')
plt.title('Энергетическая яма электрона-хопфиона (точный BVP-профиль)')
plt.legend()
plt.grid(alpha=0.3)
plt.savefig('energy_well_exact.png', dpi=150)
plt.show()
