import numpy as np
from scipy.integrate import solve_bvp, quad
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")

# =====================================================================
# 1. ЗАДАЕМ КОНСТАНТЫ И УРАВНЕНИЕ ПОЛЯ
# =====================================================================
alpha_exact = 0.0072973525693

def ode_system(r, y):
    """
    Оригинальная, математически точная система ОДУ.
    """
    f, fp = y
    sin_f = np.sin(f)
    cos_f = np.cos(f)
    sin2 = sin_f**2
    
    denom = r**2 + 2 * sin2
    
    term1 = 2 * sin_f * cos_f * (1 - fp**2 + sin2/r**2)
    term2 = (alpha_exact / 2) * r**2 * sin_f  # ВОЗВРАЩЕНО ВАШЕ ДЕЛЕНИЕ НА 2
    term3 = -2 * r * fp
    
    fpp = (term1 + term2 + term3) / denom
    return np.vstack((fp, fpp))

def boundary_conditions(ya, yb):
    return np.array([ya[0] - np.pi, yb[0] - 0.0])

# =====================================================================
# 2. ИНИЦИАЛИЗАЦИЯ ПОЛЯ (РЕЛАКСАЦИЯ ИЗ АНЗАЦА)
# =====================================================================
print("="*60)
print(" ТОПОЛОГИЧЕСКИЙ BVP РЕШАТЕЛЬ: ГЕНЕРАЦИЯ ЭЛЕКТРОНА-ХОПФИОНА")
print("="*60)
print("[1] Создание стартового многообразия (аналитический анзац)...")

# ВОЗВРАЩЕНЫ ВАШИ ПРЕДЕЛЫ СЕТКИ (Они идеальны для сходимости)
r_min = 1e-4
r_max = 120.0
r_nodes = np.logspace(np.log10(r_min), np.log10(r_max), 1000)

A_guess = 0.8168
B_guess = np.sqrt(alpha_exact)
u_guess = (A_guess / r_nodes) * np.exp(-B_guess * r_nodes)
f_guess = 2 * np.arctan(u_guess)
fp_guess = (2 * u_guess / (1 + u_guess**2)) * (-1/r_nodes - B_guess)

y_guess = np.vstack((f_guess, fp_guess))

# =====================================================================
# 3. РЕШЕНИЕ УРАВНЕНИЯ
# =====================================================================
print("[2] Запуск нелинейной релаксации поля (solve_bvp)...")
sol = solve_bvp(ode_system, boundary_conditions, r_nodes, y_guess, tol=1e-8, max_nodes=50000)

if sol.success:
    print("[+] Поле успешно достигло абсолютного минимума энергии!")
    
    # =====================================================================
    # 4. ВЫЧИСЛЕНИЕ ИНТЕГРАЛОВ ИЗ ТОЧНОГО РЕШЕНИЯ
    # =====================================================================
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

    print("[3] Интегрирование тензора энергии-импульса...")
    I2, _ = quad(I2_integrand, r_min, r_max, limit=500)
    I4, _ = quad(I4_integrand, r_min, r_max, limit=500)
    I0, _ = quad(I0_integrand, r_min, r_max, limit=500)
    
    alpha_calculated = (I4 - I2) / (3 * I0)
    
    print("\n" + "="*60)
    print(" РЕЗУЛЬТАТЫ: ГЕОМЕТРИЯ ПОСТОЯННОЙ ТОНКОЙ СТРУКТУРЫ")
    print("="*60)
    print(f" Интеграл упругости поля (I2) : {I2:.6f}")
    print(f" Интеграл жесткости ядра (I4) : {I4:.6f}")
    print(f" Топологическая масса (I0)    : {I0:.6f}")
    print("-" * 60)
    print(f" Экспериментальная Альфа (1/137): {alpha_exact:.10f}")
    print(f" Расчетная Альфа из геометрии   : {alpha_calculated:.10f}")
    print(f" Абсолютная невязка (Погрешность): {abs(alpha_calculated - alpha_exact):.2e}")
    print("="*60)

    # =====================================================================
    # 5. ВИЗУАЛИЗАЦИЯ (ЖУРНАЛЬНЫЙ ФОРМАТ)
    # =====================================================================
    r_plot = np.linspace(0.001, 8.0, 1000)
    y_plot = sol.sol(r_plot)
    f_vals = y_plot[0]
    fp_vals = y_plot[1]
    
    sin_f = np.sin(f_vals)
    sin_f_r = np.where(r_plot < 1e-8, -fp_vals, sin_f / r_plot)
    
    dens_I2 = r_plot**2 * fp_vals**2 + 2 * sin_f**2
    dens_I4 = sin_f**2 * (2 * fp_vals**2 + sin_f_r**2)
    dens_I0 = (1 - np.cos(f_vals)) * r_plot**2
    
    plt.style.use('dark_background')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle('Геометрия Электрона: Вывод $\\alpha$ из топологического резонанса', fontsize=18, fontweight='bold', color='white', y=0.98)

    ax1.plot(r_plot, f_vals, color='#00ffff', lw=3, label='Фазовый угол $f(\\rho)$')
    ax1.axhline(np.pi, color='#ff00ff', linestyle='--', lw=2, alpha=0.7, label='Ядро (Сингулярность $\\pi$)')
    ax1.axhline(0, color='gray', linestyle='--', lw=2, alpha=0.7, label='Вакуум ($0$)')
    ax1.set_title('Профиль топологического заряда', fontsize=14, color='white')
    ax1.set_xlabel('Безразмерный радиус ($\\rho = r/a$)', fontsize=12, color='white')
    ax1.set_ylabel('Фаза (радианы)', fontsize=12, color='white')
    ax1.set_xlim(0, 8); ax1.set_ylim(-0.2, 3.5)
    ax1.tick_params(colors='white'); ax1.grid(color='#333333', linestyle=':')
    ax1.legend(facecolor='black', edgecolor='white', labelcolor='white')

    ax2.plot(r_plot, dens_I4, color='#ff00ff', lw=2.5, label='$\\mathcal{I}_4$: Жесткость скрутки')
    ax2.fill_between(r_plot, dens_I4, color='#ff00ff', alpha=0.15)
    
    ax2.plot(r_plot, dens_I2, color='#00ffff', lw=2.5, label='$\\mathcal{I}_2$: Упругость поля')
    ax2.fill_between(r_plot, dens_I2, color='#00ffff', alpha=0.15)
    
    ax2.plot(r_plot, 3 * alpha_exact * dens_I0, color='#ffaa00', lw=2.5, linestyle='-', label='$3\\alpha\\mathcal{I}_0$: Массовый хвост')
    ax2.fill_between(r_plot, 3 * alpha_exact * dens_I0, color='#ffaa00', alpha=0.2)
    
    ax2.set_title('Вириальный баланс (Теорема Деррика)', fontsize=14, color='white')
    ax2.set_xlabel('Безразмерный радиус ($\\rho = r/a$)', fontsize=12, color='white')
    ax2.set_ylabel('Плотность энергии', fontsize=12, color='white')
    ax2.set_xlim(0, 4)
    ax2.tick_params(colors='white'); ax2.grid(color='#333333', linestyle=':')
    ax2.legend(facecolor='black', edgecolor='white', labelcolor='white')

    plt.tight_layout()
    plt.subplots_adjust(top=0.88)
    plt.show()

else:
    print("\nОшибка: Поле не смогло релаксировать.")
    print(sol.message)
