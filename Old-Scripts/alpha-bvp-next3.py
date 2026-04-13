import numpy as np
from scipy.integrate import solve_bvp, quad
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

# =================================================================
# ТОЧНАЯ МОДЕЛЬ ХОПФИОНА
# =================================================================
def ode_system(r, y, alpha):
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

def solve_for_alpha(alpha):
    """Решает BVP и возвращает (alpha, energy, alpha_calc, success)"""
    if alpha <= 0:
        return (alpha, np.nan, np.nan, False)
    # Динамический предел: хвост exp(-sqrt(alpha)*r) должен затухнуть
    r_max = min(800.0, 50.0 / np.sqrt(alpha))
    r_min = 1e-6
    r_nodes = np.logspace(np.log10(r_min), np.log10(r_max), 2000)
    
    # Начальное приближение (анзац)
    B_guess = np.sqrt(alpha)
    # Эмпирическая подгонка амплитуды, но можно и константу
    A_guess = 0.8168 * np.sqrt(0.0073 / alpha) if alpha > 0 else 0.8168
    u_guess = (A_guess / r_nodes) * np.exp(-B_guess * r_nodes)
    f_guess = 2 * np.arctan(u_guess)
    fp_guess = (2 * u_guess / (1 + u_guess**2)) * (-1/r_nodes - B_guess)
    y_guess = np.vstack((f_guess, fp_guess))
    
    try:
        sol = solve_bvp(lambda r, y: ode_system(r, y, alpha),
                        boundary_conditions,
                        r_nodes, y_guess, tol=1e-7, max_nodes=30000)
        if not sol.success:
            return (alpha, np.nan, np.nan, False)
    except:
        return (alpha, np.nan, np.nan, False)
    
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
    
    try:
        I2, _ = quad(I2_int, r_min, r_max, limit=300, epsabs=1e-8)
        I4, _ = quad(I4_int, r_min, r_max, limit=300, epsabs=1e-8)
        I0, _ = quad(I0_int, r_min, r_max, limit=300, epsabs=1e-8)
    except:
        return (alpha, np.nan, np.nan, False)
    
    if I0 == 0:
        return (alpha, np.nan, np.nan, False)
    
    alpha_calc = (I4 - I2) / (3 * I0)
    energy = I2 + I4 + 3 * alpha * I0
    return (alpha, energy, alpha_calc, True)

# =================================================================
# ПАРАЛЛЕЛЬНОЕ СКАНИРОВАНИЕ
# =================================================================
def main():
    print("="*70)
    print(" ПОИСК МИНИМУМА ЭНЕРГИИ ХОПФИОНА ПО α")
    print("="*70)
    print("Сканируем α в диапазоне, где решения существуют, и ищем минимум E(α).\n")
    
    alpha_min = 0.004
    alpha_max = 0.010
    num_steps = 80
    alphas = np.linspace(alpha_min, alpha_max, num_steps)
    print(f"Диапазон α: [{alpha_min:.4f}, {alpha_max:.4f}], шагов: {num_steps}")
    
    n_workers = min(8, cpu_count())
    print(f"Используется {n_workers} потоков")
    
    with Pool(processes=n_workers) as pool:
        results = list(tqdm(pool.imap(solve_for_alpha, alphas),
                            total=len(alphas), desc="Сканирование"))
    
    # Фильтруем успешные
    valid = [(a, e, ac) for (a, e, ac, ok) in results if ok and not np.isnan(e)]
    if not valid:
        print("Нет успешных решений!")
        return
    
    alpha_vals, energies, alpha_calc_vals = zip(*valid)
    alpha_vals = np.array(alpha_vals)
    energies = np.array(energies)
    alpha_calc_vals = np.array(alpha_calc_vals)
    
    # Находим минимум энергии
    min_idx = np.argmin(energies)
    best_alpha = alpha_vals[min_idx]
    best_energy = energies[min_idx]
    best_alpha_calc = alpha_calc_vals[min_idx]
    
    # Экспериментальное значение (только для справки)
    alpha_exp = 0.0072973525693
    
    print("\n" + "="*70)
    print(" РЕЗУЛЬТАТЫ СКАНИРОВАНИЯ")
    print("="*70)
    print(f"Всего успешных решений: {len(valid)} из {num_steps}")
    print(f"Минимум энергии при α = {best_alpha:.6f} (1/{1/best_alpha:.1f})")
    print(f"Энергия в минимуме: {best_energy:.6f}")
    print(f"Вычисленная α из интегралов в этой точке: {best_alpha_calc:.6f}")
    print(f"Экспериментальная α: {alpha_exp:.6f} (1/137.036)")
    print(f"Отклонение предсказанного минимума от эксперимента: {abs(best_alpha - alpha_exp)/alpha_exp*100:.2f}%")
    print("="*70)
    
    # Построим график
    plt.style.use('dark_background')
    plt.figure(figsize=(10,6))
    plt.plot(alpha_vals, energies, 'o-', color='cyan', markersize=4, linewidth=1)
    plt.axvline(alpha_exp, color='yellow', linestyle='--', label='Эксперимент (1/137)')
    plt.axvline(best_alpha, color='red', linestyle=':', label=f'Минимум α = {best_alpha:.5f}')
    plt.xlabel('α')
    plt.ylabel('Полная энергия (безразм.)')
    plt.title('Энергетический ландшафт хопфиона')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig('energy_vs_alpha.png', dpi=150)
    plt.show()
    
    print("\nГрафик сохранён как energy_vs_alpha.png")
    if abs(best_alpha - alpha_exp) / alpha_exp < 0.05:
        print("*** ПРЕДСКАЗАНИЕ СОВПАДАЕТ С ЭКСПЕРИМЕНТОМ В ПРЕДЕЛАХ 5% ***")
    else:
        print("Минимум не совпадает с 1/137. Возможно, требуется учёт топологической закрутки (ленты Мёбиуса).")

if __name__ == "__main__":
    main()
