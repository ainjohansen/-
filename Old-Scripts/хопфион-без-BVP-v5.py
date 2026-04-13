import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_bvp, simpson
from scipy.interpolate import interp1d
from scipy.optimize import minimize_scalar
import os

alpha = 0.0072973525693

# ------------------------------------------------------------
# 1. ТОЧНЫЙ BVP-СОЛВЕР (как в alpha-bvp.py)
# ------------------------------------------------------------
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

def solve_bvp_exact():
    r_min, r_max = 1e-4, 120.0
    r_nodes = np.logspace(np.log10(r_min), np.log10(r_max), 1000)

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
    return sol.x, sol.y[0], sol.y[1]

# ------------------------------------------------------------
# 2. ЗАГРУЗКА ИЛИ ВЫЧИСЛЕНИЕ ПРОФИЛЯ
# ------------------------------------------------------------
profile_file = 'bvp_exact_profile.npz'
if os.path.exists(profile_file):
    data = np.load(profile_file)
    r_bvp = data['r']
    f_bvp = data['f']
    fp_bvp = data['fp']
    print("Загружен сохранённый точный профиль BVP.")
else:
    print("Решаем BVP (может занять несколько секунд)...")
    r_bvp, f_bvp, fp_bvp = solve_bvp_exact()
    np.savez(profile_file, r=r_bvp, f=f_bvp, fp=fp_bvp)
    print("BVP решён и сохранён.")

# ------------------------------------------------------------
# 3. ВЫЧИСЛЕНИЕ ЭТАЛОННЫХ ИНТЕГРАЛОВ ПРИ a=1
# ------------------------------------------------------------
def compute_reference_integrals(r, f, fp):
    """Вычисляет I2, I4, I0, R_eff для заданного профиля."""
    sin_f = np.sin(f)
    # избегаем деления на ноль
    sin_f_r = np.where(r < 1e-8, -fp, sin_f / r)

    I2_dens = r**2 * fp**2 + 2 * sin_f**2
    I4_dens = sin_f**2 * (2 * fp**2 + sin_f_r**2)
    I0_dens = (1 - np.cos(f)) * r**2

    I2 = simpson(4 * np.pi * I2_dens, x=r)
    I4 = simpson(4 * np.pi * I4_dens, x=r)
    I0 = simpson(4 * np.pi * I0_dens, x=r)

    # Топологическая плотность заряда
    rho_top = (1/(4*np.pi)) * 2 * fp * sin_f**2 / (r**2 + 1e-12)
    Q = simpson(4 * np.pi * r**2 * rho_top, x=r)
    R_eff = simpson(4 * np.pi * r**3 * rho_top, x=r) / Q

    return I2, I4, I0, R_eff

I2_ref, I4_ref, I0_ref, R_eff_ref = compute_reference_integrals(r_bvp, f_bvp, fp_bvp)

print("\n=== ЭТАЛОННЫЕ ИНТЕГРАЛЫ ПРИ a=1 ===")
print(f"I2 = {I2_ref:.6f}")
print(f"I4 = {I4_ref:.6f}")
print(f"I0 = {I0_ref:.6f}")
print(f"R_eff = {R_eff_ref:.6f}")
print(f"Баланс: (I4 - I2) = {I4_ref - I2_ref:.6f}, 3αI0 = {3*alpha*I0_ref:.6f}")

# ------------------------------------------------------------
# 4. МАСШТАБИРОВАНИЕ И ЭНЕРГЕТИЧЕСКАЯ ЯМА
# ------------------------------------------------------------
def total_energy_scaled(a):
    """Энергия при масштабировании координат r -> a*r."""
    I2 = I2_ref / a
    I4 = I4_ref * a
    I0 = I0_ref * a**3
    E_coulomb = alpha / (R_eff_ref * a)
    return I2 + I4 + 3 * alpha * I0 + E_coulomb

# Поиск минимума
res = minimize_scalar(total_energy_scaled, bounds=(0.5, 2.0), method='bounded')
a_opt = res.x
E_min = res.fun

print("\n=== РЕЗУЛЬТАТЫ МАСШТАБИРОВАНИЯ ===")
print(f"Оптимальный масштаб a = {a_opt:.6f}")
print(f"Минимальная энергия E = {E_min:.6f}")
print(f"I2(a) = {I2_ref/a_opt:.6f}")
print(f"I4(a) = {I4_ref*a_opt:.6f}")
print(f"I0(a) = {I0_ref*a_opt**3:.6f}")
print(f"Кулоновская энергия = {alpha/(R_eff_ref*a_opt):.6f}")
print(f"Баланс при a_opt: (I4 - I2) = {I4_ref*a_opt - I2_ref/a_opt:.6f}, 3αI0 = {3*alpha*I0_ref*a_opt**3:.6f}")

# ------------------------------------------------------------
# 5. ГРАФИК ЯМЫ
# ------------------------------------------------------------
a_vals = np.linspace(0.5, 2.0, 200)
E_vals = [total_energy_scaled(a) for a in a_vals]

plt.figure(figsize=(8,5))
plt.plot(a_vals, E_vals, 'b-', lw=2)
plt.axvline(a_opt, color='r', linestyle='--', label=f'Минимум a = {a_opt:.4f}')
plt.xlabel('Масштабный фактор a')
plt.ylabel('Полная энергия')
plt.title('Энергетическая яма электрона-хопфиона (точное масштабирование BVP)')
plt.legend()
plt.grid(alpha=0.3)
plt.savefig('energy_well_exact_scaling.png', dpi=150)
plt.show()
