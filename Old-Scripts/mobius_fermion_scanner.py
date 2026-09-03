import numpy as np
from scipy.integrate import solve_bvp, quad
from scipy.optimize import root_scalar
import multiprocessing as mp
import matplotlib.pyplot as plt
import time
import warnings
import sys

warnings.filterwarnings("ignore")

# =====================================================================
# ТОПОЛОГИЧЕСКИЕ ПАРАМЕТРЫ ФЕРМИОНА (Спин 1/2)
# =====================================================================
# Для Сферического Ежа: A=2, B=2, C=1
# Для Ленты Мёбиуса (Хопфиона) с полуцелым спином геометрия другая:
SPIN = 0.5
A = 2 * SPIN**2  # 0.5
B = 2 * SPIN**2  # 0.5
C = SPIN**2      # 0.25

# =====================================================================
# ДВИЖОК СИМУЛЯЦИИ (Обобщенная геометрия)
# =====================================================================
def simulate_mobius_universe(K):
    def ode_system(r, y):
        f, fp = y
        sin_f = np.sin(f)
        cos_f = np.cos(f)
        sin2 = sin_f**2
        
        # Обобщенный знаменатель и числитель с учетом закрутки A, B, C
        denom = 2 * (r**2 + B * sin2)
        
        term1 = np.sin(2*f) * (A - B * fp**2 + 2 * C * sin2 / r**2)
        term2 = K * r**2 * sin_f
        term3 = -4 * r * fp
        
        return np.vstack((fp, (term1 + term2 + term3) / denom))

    def boundary_conditions(ya, yb):
        return np.array([ya[0] - np.pi, yb[0] - 0.0])

    # Глубокий космос
    r_nodes = np.logspace(-4, 3.0, 2000)
    B_guess = np.sqrt(K) if K > 0 else 0.1
    u_guess = (0.8 / r_nodes) * np.exp(-B_guess * r_nodes)
    f_guess = 2 * np.arctan(u_guess)
    fp_guess = (2 * u_guess / (1 + u_guess**2)) * (-1/r_nodes - B_guess)
    y_guess = np.vstack((f_guess, fp_guess))

    # Сверхточное решение
    sol = solve_bvp(ode_system, boundary_conditions, r_nodes, y_guess, tol=1e-7, max_nodes=30000)

    if not sol.success:
        return (K, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan)

    # Интегралы с новыми геометрическими весами
    def i2_int(r):
        y = sol.sol(r)
        return r**2 * y[1]**2 + A * np.sin(y[0])**2

    def i4_int(r):
        y = sol.sol(r)
        sin_f = np.sin(y[0])
        sin_f_r = sin_f / r if r > 1e-8 else -y[1]
        return sin_f**2 * (B * y[1]**2 + C * sin_f_r**2)

    def i0_int(r):
        y = sol.sol(r)
        return (1 - np.cos(y[0])) * r**2

    try:
        I2, _ = quad(i2_int, 1e-4, 500.0, limit=300)
        I4, _ = quad(i4_int, 1e-4, 500.0, limit=300)
        I0, _ = quad(i0_int, 1e-4, 500.0, limit=300)
    except:
        return (K, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan)

    E_total = I4 + I2 + 3 * K * I0
    
    try:
        res = root_scalar(lambda r: sol.sol(r)[0] - np.pi/2, bracket=[0.01, 20.0])
        R_core = res.root if res.converged else np.nan
    except:
        R_core = np.nan
        
    tension_origin = -sol.sol(1e-4)[1]
    K_calc = (I4 - I2) / (3 * I0) if I0 != 0 else np.nan

    return (K, I2, I4, I0, E_total, R_core, tension_origin, K_calc)

# =====================================================================
# ГЛАВНЫЙ БЛОК: КЛАСТЕР 
# =====================================================================
if __name__ == '__main__':
    print("="*75)
    print(" BVP-СКАНИРОВАНИЕ ТОПОЛОГИИ ЛЕНТЫ МЁБИУСА (ФЕРМИОН СПИН 1/2)")
    print("="*75)
    
    # Сканируем диапазон вокруг 1/137
    K_vals = np.linspace(0.002, 0.015, 400)
    threads = min(22, mp.cpu_count() - 2)
    
    print(f"[*] Задействовано потоков: {threads}")
    print(f"[*] Топологические веса: A={A}, B={B}, C={C}")
    print(f"[*] Ищем истинный баланс Деррика...\n")
    
    results =[]
    start_time = time.time()
    
    with mp.Pool(processes=threads) as pool:
        for i, res in enumerate(pool.imap_unordered(simulate_mobius_universe, K_vals), 1):
            results.append(res)
            
            elapsed = time.time() - start_time
            fps = i / elapsed
            eta = (len(K_vals) - i) / fps if fps > 0 else 0
            
            sys.stdout.write(f"\r[►] Прогресс: {i}/{len(K_vals)} | Успешных миров | Прошло: {elapsed:.1f}с | Осталось: {eta:.1f}с")
            sys.stdout.flush()
            
    print("\n\n[*] Вычисления завершены! Анализируем...")

    data = np.array([r for r in results if not np.isnan(r[1])])
    data = data[data[:, 0].argsort()]
    
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
    # СТАТИСТИЧЕСКИЙ АНАЛИЗ (ПОИСК АТТРАКТОРА)
    # =====================================================================
    mismatch = np.abs(K_v - K_calc_v)
    best_idx = np.argmin(mismatch)
    
    K_ideal = K_v[best_idx]
    
    print("\n" + "="*75)
    print(" ИТОГИ МОДЕЛИРОВАНИЯ ЗАКРУЧЕННОГО ХОПФИОНА")
    print("="*75)
    print(f"ТОЧКА МАТЕМАТИЧЕСКОГО БАЛАНСА (Теорема Деррика):")
    print(f"  Идеальная Мера K        = {K_ideal:.6f}  (1/{1/K_ideal:.2f})")
    print(f"  Экспериментальная Мера  = {ALPHA_EXP:.6f}  (1/{1/ALPHA_EXP:.2f})")
    
    error = abs(K_ideal - ALPHA_EXP) / ALPHA_EXP * 100
    print(f"\nОтклонение математики от физики: {error:.2f}%")
    print("="*75)

    # =====================================================================
    # ВИЗУАЛИЗАЦИЯ
    # =====================================================================
    plt.style.use('dark_background')
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('Лента Мёбиуса: Поиск топологического резонанса', fontsize=16)

    # 1. Баланс Деррика
    ax1 = axes[0]
    ax1.plot(K_v, K_calc_v, color='cyan', lw=3, label='Вычисленная Мера (Из интегралов)')
    ax1.plot(K_v, K_v, color='magenta', lw=2, linestyle=':', label='Идеальное совпадение $y=x$')
    ax1.axvline(ALPHA_EXP, color='yellow', linestyle='--', alpha=0.8, label='Физика (1/137)')
    ax1.set_title('Истинный аттрактор Деррика')
    ax1.set_xlabel('Заданная абстрактная мера K')
    ax1.set_ylabel('Мера из интегралов $K_{calc}$')
    ax1.legend()
    ax1.grid(color='white', alpha=0.1)

    # 2. Отношение I4 / I2
    ax2 = axes[1]
    ax2.plot(K_v, I4_v / I2_v, color='lime', lw=2)
    ax2.axvline(ALPHA_EXP, color='yellow', linestyle='--', alpha=0.8)
    ax2.set_title('Структурный резонанс (I4 / I2)')
    ax2.set_xlabel('Абстрактная Мера K')
    ax2.set_ylabel('Отношение Интегралов')
    ax2.grid(color='white', alpha=0.1)

    plt.tight_layout()
    plt.show()