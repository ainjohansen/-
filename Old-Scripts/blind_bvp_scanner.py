import numpy as np
from scipy.integrate import solve_bvp, quad
from scipy.optimize import root_scalar
import multiprocessing as mp
import matplotlib.pyplot as plt
import time
import warnings

warnings.filterwarnings("ignore")

# =====================================================================
# ДВИЖОК СИМУЛЯЦИИ (Исправлен артефакт сетки)
# =====================================================================
def simulate_abstract_universe(K):
    def ode_system(r, y):
        f, fp = y
        sin_f = np.sin(f)
        cos_f = np.cos(f)
        sin2 = sin_f**2
        denom = r**2 + 2 * sin2
        
        term1 = 2 * sin_f * cos_f * (1 - fp**2 + sin2/r**2)
        term2 = (K / 2) * r**2 * sin_f
        term3 = -2 * r * fp
        return np.vstack((fp, (term1 + term2 + term3) / denom))

    def boundary_conditions(ya, yb):
        return np.array([ya[0] - np.pi, yb[0] - 0.0])

    # 1. РЕЖИМ ГЛУБОКОГО КОСМОСА: Сетка до R = 1000
    r_nodes = np.logspace(-4, 3, 2000) 
    B_guess = np.sqrt(K) if K > 0 else 0.1
    u_guess = (0.8 / r_nodes) * np.exp(-B_guess * r_nodes)
    f_guess = 2 * np.arctan(u_guess)
    fp_guess = (2 * u_guess / (1 + u_guess**2)) * (-1/r_nodes - B_guess)
    y_guess = np.vstack((f_guess, fp_guess))

    # 2. УЛЬТРА-ТОЧНОСТЬ: tol=1e-8
    sol = solve_bvp(ode_system, boundary_conditions, r_nodes, y_guess, tol=1e-8, max_nodes=50000)

    if not sol.success:
        return (K, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan)

    def i2_int(r):
        y = sol.sol(r)
        return r**2 * y[1]**2 + 2 * np.sin(y[0])**2

    def i4_int(r):
        y = sol.sol(r)
        sin_f = np.sin(y[0])
        sin_f_r = sin_f / r if r > 1e-8 else -y[1]
        return sin_f**2 * (2 * y[1]**2 + sin_f_r**2)

    def i0_int(r):
        y = sol.sol(r)
        return (1 - np.cos(y[0])) * r**2

    # 3. ПОЛНЫЙ СБОР МАССЫ: Интегрируем до R = 800
    try:
        I2, _ = quad(i2_int, 1e-4, 800.0, limit=500)
        I4, _ = quad(i4_int, 1e-4, 800.0, limit=500)
        I0, _ = quad(i0_int, 1e-4, 800.0, limit=500)
    except:
        return (K, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan)

    E_total = I4 + I2 + 3 * K * I0
    
    try:
        res = root_scalar(lambda r: sol.sol(r)[0] - np.pi/2, bracket=[0.1, 5.0])
        R_core = res.root if res.converged else np.nan
    except:
        R_core = np.nan
        
    tension_origin = -sol.sol(1e-4)[1]
    K_calc = (I4 - I2) / (3 * I0) if I0 != 0 else np.nan

    return (K, I2, I4, I0, E_total, R_core, tension_origin, K_calc)

# =====================================================================
# ГЛАВНЫЙ БЛОК УПРАВЛЕНИЯ КЛАСТЕРОМ
# =====================================================================
if __name__ == '__main__':
    print("="*75)
    print(" АБСОЛЮТНО СЛЕПОЕ BVP-СКАНИРОВАНИЕ (ВЕРСИЯ С АНАЛИТИКОЙ)")
    print("="*75)
    
    K_vals = np.linspace(0.005, 0.025, 400)
    threads = min(22, mp.cpu_count() - 2)
    
    start_time = time.time()
    with mp.Pool(processes=threads) as pool:
        results = pool.map(simulate_abstract_universe, K_vals)
        
    print(f"Сканирование {len(K_vals)} миров завершено за {time.time() - start_time:.1f} сек.")

    data = np.array([r for r in results if not np.isnan(r[1])])
    K_v       = data[:, 0]
    I2_v      = data[:, 1]
    I4_v      = data[:, 2]
    I0_v      = data[:, 3]
    E_tot_v   = data[:, 4]
    R_core_v  = data[:, 5]
    Tension_v = data[:, 6]
    K_calc_v  = data[:, 7]

    ALPHA_EXP = 0.0072973525

    # =====================================================================
    # СТАТИСТИЧЕСКИЙ ВЫВОД В КОНСОЛЬ
    # =====================================================================
    print("\n" + "="*75)
    print(" РЕЗУЛЬТАТЫ ПОИСКА ТОПОЛОГИЧЕСКОГО РЕЗОНАНСА")
    print("="*75)

    # 1. Поиск Идеального Баланса (Точка Деррика)
    # Это место, где заданная Альфа (K_v) идеально совпадает с вычисленной из интегралов (K_calc_v)
    mismatch = np.abs(K_v - K_calc_v)
    best_idx = np.argmin(mismatch)
    
    K_ideal = K_v[best_idx]
    K_calc_ideal = K_calc_v[best_idx]
    R_ideal = R_core_v[best_idx]
    
    # 2. Наблюдаемые значения в нашей Вселенной
    exp_idx = np.argmin(np.abs(K_v - ALPHA_EXP))
    
    print("1. ТОЧКА АБСОЛЮТНОЙ САМОСОГЛАСОВАННОСТИ (Где математика замыкается сама на себя):")
    print(f"   Идеальная Мера K        = {K_ideal:.6f} (1/{1/K_ideal:.2f})")
    print(f"   Вычисленная Деррик-Мера = {K_calc_ideal:.6f}")
    print(f"   Погрешность согласования= {mismatch[best_idx]:.2e}")
    print(f"   Топологический радиус   = {R_ideal:.6f}")
    print(f"   Отношение I4 / I2       = {(I4_v[best_idx]/I2_v[best_idx]):.6f}")

    print("\n2. НАША ЭКСПЕРИМЕНТАЛЬНАЯ ВСЕЛЕННАЯ (Для сравнения):")
    print(f"   Мера 1/137              = {ALPHA_EXP:.6f}")
    print(f"   Топологический радиус   = {R_core_v[exp_idx]:.6f}")
    print(f"   Отношение I4 / I2       = {(I4_v[exp_idx]/I2_v[exp_idx]):.6f}")
    print(f"   Натяжение центра f'(0)  = {Tension_v[exp_idx]:.6f}")
    
    error_percent = abs(K_ideal - ALPHA_EXP) / ALPHA_EXP * 100
    print(f"\nВывод: Свободная математика отклоняется от физики на {error_percent:.2f}%")
    print("="*75)

    # =====================================================================
    # ОБНОВЛЕННАЯ ВИЗУАЛИЗАЦИЯ
    # =====================================================================
    plt.style.use('dark_background')
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle('Анализ BVP Континуума: В поисках идеальной меры', fontsize=16)

    # График 1: Баланс Деррика (ЭТО САМОЕ ВАЖНОЕ)
    ax1 = axes[0, 0]
    ax1.plot(K_v, K_calc_v, color='cyan', lw=2.5, label='Вычисленная $\\alpha$ (Теорема Деррика)')
    ax1.plot(K_v, K_v, color='magenta', lw=1.5, linestyle=':', label='Идеальное совпадение $K=K$')
    ax1.scatter([K_ideal], [K_calc_ideal], color='white', s=100, zorder=5, label=f'Аттрактор: 1/{1/K_ideal:.1f}')
    ax1.axvline(ALPHA_EXP, color='yellow', linestyle='--', alpha=0.8, label='Физика (1/137)')
    ax1.set_title('Проверка Самосогласованности')
    ax1.set_xlabel('Заданная абстрактная мера K')
    ax1.set_ylabel('Мера Деррика из интегралов')
    ax1.legend()
    ax1.grid(color='white', alpha=0.1)

    # График 2: Отношение I4 / I2
    ax2 = axes[0, 1]
    ax2.plot(K_v, I4_v / I2_v, color='lime', lw=2)
    ax2.axvline(ALPHA_EXP, color='yellow', linestyle='--', alpha=0.8)
    ax2.axvline(K_ideal, color='white', linestyle='--', alpha=0.5)
    ax2.set_title('Структурный резонанс (I4 / I2)')
    ax2.set_xlabel('Абстрактная Мера (K)')
    ax2.set_ylabel('Отношение Интегралов')
    ax2.grid(color='white', alpha=0.1)

    # График 3: Радиус ядра (Теперь должен быть плавным)
    ax3 = axes[1, 0]
    ax3.plot(K_v, R_core_v, color='orange', lw=2)
    ax3.axvline(ALPHA_EXP, color='yellow', linestyle='--', alpha=0.8)
    ax3.axvline(K_ideal, color='white', linestyle='--', alpha=0.5)
    ax3.set_title('Точная деформация Ядра (Исправлен артефакт сетки)')
    ax3.set_xlabel('Абстрактная Мера (K)')
    ax3.set_ylabel('Радиус $R_{core}$')
    ax3.grid(color='white', alpha=0.1)

    # График 4: Натяжение f'(0)
    ax4 = axes[1, 1]
    ax4.plot(K_v, Tension_v, color='violet', lw=2)
    ax4.axvline(ALPHA_EXP, color='yellow', linestyle='--', alpha=0.8)
    ax4.axvline(K_ideal, color='white', linestyle='--', alpha=0.5)
    ax4.set_title('Натяжение центральной сингулярности')
    ax4.set_xlabel('Абстрактная Мера (K)')
    ax4.set_ylabel("Производная фазы f'(0)")
    ax4.grid(color='white', alpha=0.1)

    plt.tight_layout()
    plt.show()
