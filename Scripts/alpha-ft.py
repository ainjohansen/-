import numpy as np
from scipy.integrate import solve_bvp, quad
import matplotlib.pyplot as plt

# ============================================================
# 1. ПАРАМЕТРЫ ЗАДАЧИ (как в успешном запуске)
# ============================================================
alpha_exact = 0.0072973525693

def ode_system(r, y):
    f, fp = y
    sin_f = np.sin(f)
    cos_f = np.cos(f)
    sin2 = sin_f*sin_f
    denom = r*r + 2*sin2
    term1 = 2 * sin_f * cos_f * (1 - fp*fp + sin2/(r*r))
    term2 = (alpha_exact / 2) * r*r * sin_f
    term3 = -2 * r * fp
    fpp = (term1 + term2 + term3) / denom
    return np.vstack((fp, fpp))

def bc(ya, yb):
    return np.array([ya[0] - np.pi, yb[0]])

r_min = 1e-4
r_max = 120.0
n = 1000
r = np.logspace(np.log10(r_min), np.log10(r_max), n)

A0 = 0.8168
B0 = np.sqrt(alpha_exact)
u0 = (A0 / r) * np.exp(-B0 * r)
f0 = 2 * np.arctan(u0)
fp0 = (2 * u0 / (1 + u0**2)) * (-1/r - B0)
y_init = np.vstack((f0, fp0))

print("Решаем BVP...")
sol = solve_bvp(ode_system, bc, r, y_init, max_nodes=50000, verbose=2)
if not sol.success:
    print("BVP не сошлось:", sol.message)
    exit()
print("BVP успешно решено")

# ============================================================
# 2. ВЫЧИСЛЕНИЕ ИНТЕГРАЛОВ С УЧЁТОМ ХВОСТА
# ============================================================
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

# Интегрирование до r_max
I2, err2 = quad(integrand_I2, r_min, r_max, limit=2000, epsabs=1e-12, epsrel=1e-12)
I4, err4 = quad(integrand_I4, r_min, r_max, limit=2000, epsabs=1e-12, epsrel=1e-12)
I0, err0 = quad(integrand_I0, r_min, r_max, limit=2000, epsabs=1e-12, epsrel=1e-12)

# Асимптотический хвост
alpha_calc = (I4 - I2) / (3 * I0)
m = np.sqrt(alpha_calc)
f_R = sol.sol(r_max)[0]
C = f_R * np.sqrt(r_max) * np.exp(m * r_max)

def tail_f(r):
    return C * np.exp(-m * r) / np.sqrt(r)
def tail_fp(r):
    return -C * np.exp(-m * r) * (m + 0.5/r) / np.sqrt(r)

I2_tail, _ = quad(lambda r: r*r*tail_fp(r)**2 + 2*np.sin(tail_f(r))**2, r_max, np.inf, limit=1000)
I4_tail, _ = quad(lambda r: (np.sin(tail_f(r))**2)*(2*tail_fp(r)**2 + (np.sin(tail_f(r))/r)**2), r_max, np.inf, limit=1000)
I0_tail, _ = quad(lambda r: (1 - np.cos(tail_f(r)))*r*r, r_max, np.inf, limit=1000)

I2 += I2_tail
I4 += I4_tail
I0 += I0_tail
alpha_final = (I4 - I2) / (3 * I0)

# ============================================================
# 3. НАГЛЯДНАЯ ТАБЛИЦА
# ============================================================
print("\n" + "="*70)
print("РЕЗУЛЬТАТЫ РАСЧЁТА (с учётом асимптотического хвоста)")
print("="*70)
print(f"{'Интеграл':<20} {'Значение':<20} {'Погрешность':<20}")
print("-"*70)
print(f"{'I2':<20} {I2:<20.10f} {err2:<20.2e}")
print(f"{'I4':<20} {I4:<20.10f} {err4:<20.2e}")
print(f"{'I0':<20} {I0:<20.10f} {err0:<20.2e}")
print("-"*70)
print(f"{'Δ = I4 - I2':<20} {I4 - I2:<20.10f} {'':<20}")
print(f"{'α = (I4-I2)/(3I0)':<20} {alpha_final:<20.10f} {'':<20}")
print(f"{'Экспериментальное α':<20} {alpha_exact:<20.10f} {'':<20}")
print(f"{'Отклонение':<20} {abs(alpha_final - alpha_exact):<20.2e} {'':<20}")
print("="*70)

# ============================================================
# 4. СРАВНЕНИЕ С ПАРАМЕТРИЧЕСКОЙ АППРОКСИМАЦИЕЙ
# ============================================================
A_par = 0.8168
B_par = 0.08542
def f_par(r):
    u = A_par / r * np.exp(-B_par * r)
    return 2 * np.arctan(u)

r_plot = np.logspace(np.log10(r_min), np.log10(r_max), 1000)
f_num = sol.sol(r_plot)[0]
f_par_vals = f_par(r_plot)

abs_err = np.abs(f_num - f_par_vals)
rel_err = abs_err / (np.abs(f_num) + 1e-12)
print("\nСравнение с параметрической аппроксимацией (A=0.8168, B=0.08542):")
print(f"Максимальная абсолютная ошибка: {np.max(abs_err):.3e}")
print(f"Максимальная относительная ошибка: {np.max(rel_err):.3e}")
print(f"Среднеквадратичная ошибка: {np.sqrt(np.mean(abs_err**2)):.3e}")

# ============================================================
# 5. ПОСТРОЕНИЕ ГРАФИКОВ
# ============================================================
plt.style.use('dark_background')
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Электрон-хопфион: точное решение BVP', fontsize=16)

# Профиль f(ρ)
ax = axes[0,0]
ax.plot(r_plot, f_num, 'c-', lw=2, label='Численное решение')
ax.plot(r_plot, f_par_vals, 'r--', lw=1.5, label='Параметрическая аппроксимация')
ax.axhline(np.pi, color='magenta', ls='--', alpha=0.5, label='π (ядро)')
ax.axhline(0, color='gray', ls='--', alpha=0.5, label='0 (вакуум)')
ax.set_xscale('log')
ax.set_xlabel(r'$\rho$')
ax.set_ylabel(r'$f(\rho)$')
ax.legend()
ax.grid(alpha=0.2)

# Разность профилей
ax = axes[0,1]
ax.plot(r_plot, f_num - f_par_vals, 'w-', lw=1)
ax.set_xscale('log')
ax.set_xlabel(r'$\rho$')
ax.set_ylabel(r'$\Delta f(\rho)$')
ax.set_title('Разность численного и параметрического профилей')
ax.grid(alpha=0.2)

# Плотности энергии (линейный масштаб)
ax = axes[1,0]
r_lin = np.linspace(r_min, 8, 1000)
f_lin = sol.sol(r_lin)[0]
fp_lin = sol.sol(r_lin)[1]
sin_f_lin = np.sin(f_lin)
sin2_lin = sin_f_lin**2
sin_f_over_r = sin_f_lin / r_lin
dens_I2 = r_lin**2 * fp_lin**2 + 2 * sin2_lin
dens_I4 = sin2_lin * (2 * fp_lin**2 + sin_f_over_r**2)
dens_I0 = (1 - np.cos(f_lin)) * r_lin**2

ax.plot(r_lin, dens_I4, 'm-', lw=2, label='$\\mathcal{I}_4$ (жёсткость)')
ax.plot(r_lin, dens_I2, 'c-', lw=2, label='$\\mathcal{I}_2$ (упругость)')
ax.plot(r_lin, 3*alpha_final*dens_I0, 'y--', lw=2, label='$3\\alpha\\mathcal{I}_0$ (масса)')
ax.set_xlabel(r'$\rho$')
ax.set_ylabel('Плотность энергии')
ax.set_xlim(0, 5)
ax.legend()
ax.grid(alpha=0.2)

# Плотности энергии (логарифмический масштаб)
ax = axes[1,1]
r_log = np.logspace(np.log10(r_min), np.log10(50), 1000)
f_log = sol.sol(r_log)[0]
fp_log = sol.sol(r_log)[1]
sin_f_log = np.sin(f_log)
sin2_log = sin_f_log**2
sin_f_over_r_log = sin_f_log / r_log
dens_I2_log = r_log**2 * fp_log**2 + 2 * sin2_log
dens_I4_log = sin2_log * (2 * fp_log**2 + sin_f_over_r_log**2)
dens_I0_log = (1 - np.cos(f_log)) * r_log**2

ax.plot(r_log, dens_I4_log, 'm-', lw=2, label='$\\mathcal{I}_4$')
ax.plot(r_log, dens_I2_log, 'c-', lw=2, label='$\\mathcal{I}_2$')
ax.plot(r_log, 3*alpha_final*dens_I0_log, 'y--', lw=2, label='$3\\alpha\\mathcal{I}_0$')
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlabel(r'$\rho$')
ax.set_ylabel('Плотность энергии (лог)')
ax.legend()
ax.grid(alpha=0.2)

plt.tight_layout()
plt.show()
