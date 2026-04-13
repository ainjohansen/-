import numpy as np
from scipy.integrate import solve_bvp, quad
from scipy.optimize import root_scalar
import matplotlib.pyplot as plt
from multiprocessing import Pool, cpu_count
import time
import warnings
from tqdm import tqdm

warnings.filterwarnings("ignore")

# =====================================================================
# ТОЧНЫЕ УРАВНЕНИЯ (как в alpha-bvp.py)
# =====================================================================
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
    # Адаптивный предел: чтобы экспонента затухла до 1e-10
    B = np.sqrt(alpha)
    r_max = min(500.0, 200.0 / B) if B > 0 else 500.0
    r_min = 1e-5
    r_nodes = np.logspace(np.log10(r_min), np.log10(r_max), 2000)
    
    # Начальное приближение
    A_guess = 0.8168
    u_guess = (A_guess / r_nodes) * np.exp(-B * r_nodes)
    f_guess = 2 * np.arctan(u_guess)
    fp_guess = (2 * u_guess / (1 + u_guess**2)) * (-1/r_nodes - B)
    y_guess = np.vstack((f_guess, fp_guess))
    
    try:
        sol = solve_bvp(lambda r, y: ode_system(r, y, alpha),
                        boundary_conditions,
                        r_nodes, y_guess, tol=1e-9, max_nodes=50000)
        if not sol.success:
            return (alpha, False, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    except Exception as e:
        return (alpha, False, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    
    # Интегрирование до r_max (хвост экспоненциально мал)
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
        I2, _ = quad(I2_int, r_min, r_max, limit=500, epsabs=1e-10, epsrel=1e-8)
        I4, _ = quad(I4_int, r_min, r_max, limit=500, epsabs=1e-10, epsrel=1e-8)
        I0, _ = quad(I0_int, r_min, r_max, limit=500, epsabs=1e-10, epsrel=1e-8)
    except:
        return (alpha, False, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    
    if I0 == 0:
        return (alpha, False, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    
    alpha_calc = (I4 - I2) / (3 * I0)
    energy = I2 + I4 + 3 * alpha * I0
    
    # Радиус, где f=π/2
    try:
        res = root_scalar(lambda r: sol.sol(r)[0] - np.pi/2, bracket=[0.01, 20.0])
        R_core = res.root if res.converged else np.nan
    except:
        R_core = np.nan
    
    tension = -sol.sol(1e-5)[1]
    return (alpha, True, energy, alpha_calc, I2, I4, I0, R_core, tension)

def main():
    print("="*70)
    print(" ВЫСОКОТОЧНЫЙ BVP-СКАНЕР (адаптивный r_max, tol=1e-9)")
    print("="*70)
    
    alpha_min, alpha_max = 0.001, 0.015
    num_steps = 100  # можно увеличить, но будет дольше
    alpha_list = np.linspace(alpha_min, alpha_max, num_steps)
    print(f"Сканирование {len(alpha_list)} значений α от {alpha_min} до {alpha_max}")
    print("Это может занять 1-2 часа...")
    
    n_workers = min(8, cpu_count())  # уменьшим потоки, чтобы не перегружать память
    print(f"Используется {n_workers} потоков")
    
    start = time.time()
    with Pool(processes=n_workers) as pool:
        results = list(tqdm(pool.imap(solve_for_alpha, alpha_list),
                            total=len(alpha_list), desc="Сканирование"))
    elapsed = time.time() - start
    print(f"\nЗавершено за {elapsed/60:.1f} мин")
    
    # Фильтрация по самосогласованности
    valid = []
    for r in results:
        if not r[1]:
            continue
        alpha, _, energy, alpha_calc, I2, I4, I0, R_core, tension = r
        if abs(alpha_calc - alpha) > 1e-6:
            continue  # строгий критерий
        valid.append(r)
    
    if not valid:
        print("Нет успешных решений")
        return
    
    valid = np.array(valid, dtype=object)
    alpha_vals = np.array([v[0] for v in valid])
    energies = np.array([v[2] for v in valid])
    alpha_calc_vals = np.array([v[3] for v in valid])
    
    min_idx = np.argmin(energies)
    best_alpha = alpha_vals[min_idx]
    best_energy = energies[min_idx]
    
    alpha_exp = 0.0072973525693
    exp_idx = np.argmin(np.abs(alpha_vals - alpha_exp))
    exp_energy = energies[exp_idx] if abs(alpha_vals[exp_idx] - alpha_exp) < 0.0001 else None
    
    print("\n" + "="*70)
    print(" РЕЗУЛЬТАТЫ")
    print("="*70)
    print(f"Успешных решений: {len(valid)} из {len(alpha_list)}")
    print(f"Минимум энергии при α = {best_alpha:.6f} (1/{1/best_alpha:.1f})")
    print(f"Энергия в минимуме: {best_energy:.6f}")
    if exp_energy:
        print(f"Энергия при α = {alpha_exp:.6f}: {exp_energy:.6f}")
        print(f"Разница: {exp_energy - best_energy:.6f}")
    print(f"Отклонение от эксперимента: {abs(best_alpha - alpha_exp)/alpha_exp*100:.2f}%")
    print("="*70)
    
    # График
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(10,6))
    ax.plot(alpha_vals, energies, 'o-', color='cyan', markersize=4, linewidth=1)
    ax.axvline(alpha_exp, color='yellow', linestyle='--', label='Эксперимент')
    ax.axvline(best_alpha, color='red', linestyle=':', label='Минимум')
    ax.set_xlabel('α')
    ax.set_ylabel('Энергия')
    ax.legend()
    ax.grid(alpha=0.3)
    plt.title('Энергетический ландшафт хопфиона (высокая точность)')
    plt.tight_layout()
    plt.savefig('bvp_scan_high_precision.png', dpi=150)
    plt.show()
    
    if best_alpha < alpha_exp:
        print("\nПредсказанный минимум лежит ниже экспериментального.")
        print("Возможно, необходимо ввести дополнительный масштаб (например, радиус протона).")
        print("Однако, если решение при α=1/137 существует и имеет более высокую энергию, ")
        print("то природа могла бы выбрать минимум. Это несоответствие требует анализа.")
    else:
        print("\nМинимум близок к экспериментальному значению.")

if __name__ == "__main__":
    main()