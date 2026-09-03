import numpy as np
from scipy.integrate import solve_ivp, cumulative_trapezoid
from scipy.optimize import root_scalar
from scipy.interpolate import UnivariateSpline
import matplotlib.pyplot as plt
from multiprocessing import Pool, cpu_count
import pickle
import os

# =========================
# ГЛОБАЛЬНЫЕ ПАРАМЕТРЫ
# =========================
r_min = 1e-4
r_max = 20.0
N = 1200
r_eval = np.linspace(r_min, r_max, N)
alpha_exp = 0.0072973525693
CACHE_FILE = 'integral_cache.pkl'

def make_ode(alpha):
    def ode(r, y):
        f, df = y
        r2 = r*r + 1e-12
        sinf = np.sin(f)
        cosf = np.cos(f)
        sin2f = np.sin(2*f)
        sinf2 = sinf**2

        A = 1 + 2 * sinf2 / r2
        B = 2/r + 2 * sin2f / r2
        dA_dr = (4*sinf*cosf*df)/r2 - (4*sinf2)/(r*r2)

        rhs = (sin2f / r2
               + sin2f * (df**2) / r2
               - (sinf2 * sin2f) / (r2*r2)
               + alpha * sinf)

        ddf = (rhs - B*df - dA_dr*df) / A
        return [df, ddf]
    return ode

def shoot(alpha, s):
    sol = solve_ivp(
        make_ode(alpha),
        [r_min, r_max],
        [np.pi, s],
        t_eval=[r_max],
        rtol=1e-4, atol=1e-6,
        method='LSODA'
    )
    if not sol.success:
        return 1e3
    return sol.y[0, -1]

def find_s(alpha):
    brackets = [(-50, -1e-3), (-100, -1), (-10, -0.1)]
    for br in brackets:
        try:
            res = root_scalar(
                lambda s: shoot(alpha, s),
                bracket=br,
                method='bisect',
                xtol=1e-3,
                maxiter=50
            )
            if res.converged:
                return res.root
        except:
            continue
    raise RuntimeError(f"Не удалось найти s для α={alpha}")

def solve_profile(alpha):
    s = find_s(alpha)
    sol = solve_ivp(
        make_ode(alpha),
        [r_min, r_max],
        [np.pi, s],
        t_eval=r_eval,
        rtol=1e-4, atol=1e-6,
        method='LSODA'
    )
    if not sol.success:
        raise RuntimeError("Профиль не сошёлся")
    return sol.t, sol.y[0], sol.y[1]

def compute_integrals(r, f, df):
    sinf = np.sin(f)
    r2 = r*r
    I2 = np.trapezoid((df**2 + 2*sinf**2/r2) * r2, r)
    I4 = np.trapezoid(
        (2*sinf**2 * df**2 / r2 + sinf**4 / (r2*r2)) * r2, r
    )
    I0 = np.trapezoid((1 - np.cos(f)) * r2, r)
    return I2, I4, I0

def find_kcrit(I2, I4, I0, alpha):
    def func(k):
        return I2 - k*I4 + 3*alpha*I0
    res = root_scalar(func, bracket=[1.0, 10.0], method='bisect')
    return res.root

def solve_poisson(r, f, df, alpha):
    sinf = np.sin(f)
    r2 = r*r
    j0_top = np.where(r > 1e-4, (sinf**2 * df) / (2 * np.pi**2 * r2), 0)
    e_charge = np.sqrt(4 * np.pi * alpha)
    j0 = e_charge * j0_top
    rhs = -j0 * r2
    I1 = cumulative_trapezoid(rhs, r, initial=0)
    dA_dr = I1 / r2
    dA_dr[0] = 0
    E_em = 2 * np.pi * np.trapezoid(dA_dr**2 * r2, r)
    return E_em

def compute_one(alpha):
    try:
        r, f, df = solve_profile(alpha)
        I2, I4, I0 = compute_integrals(r, f, df)
        k = find_kcrit(I2, I4, I0, alpha)
        E_core = I2 + k*I4 + 3*alpha*I0
        E_em_base = solve_poisson(r, f, df, alpha)
        return alpha, E_core, E_em_base, I2, I4, I0, k
    except Exception as e:
        print(f"Ошибка для α={alpha:.6f}: {e}")
        return None

def compute_all_profiles(alpha_vals):
    cache = {}
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, 'rb') as f:
            cache = pickle.load(f)

    to_compute = [a for a in alpha_vals if a not in cache]
    if to_compute:
        print(f"Вычисление {len(to_compute)} новых α...")
        with Pool(processes=cpu_count()) as pool:
            results = pool.map(compute_one, to_compute)
        for res in results:
            if res is not None:
                a, E_core, E_em_base, *_ = res
                cache[a] = (E_core, E_em_base)
        with open(CACHE_FILE, 'wb') as f:
            pickle.dump(cache, f)
    else:
        print("Все данные загружены из кэша.")

    data = {}
    for a in alpha_vals:
        if a in cache:
            data[a] = cache[a]
    return data

def total_energy(alpha, K, data):
    E_core, E_em_base = data[alpha]
    return E_core + K * E_em_base

def find_min_alpha(K, alpha_vals, data):
    alphas = []
    energies = []
    for a in alpha_vals:
        if a in data:
            alphas.append(a)
            energies.append(total_energy(a, K, data))
    alphas = np.array(alphas)
    energies = np.array(energies)
    # Используем сплайн 3-й степени
    spline = UnivariateSpline(alphas, energies, k=3, s=0)
    alpha_fine = np.linspace(alphas.min(), alphas.max(), 500)
    E_fine = spline(alpha_fine)
    return alpha_fine[np.argmin(E_fine)]

if __name__ == "__main__":
    # Густая сетка α
    alpha_vals = np.linspace(0.005, 0.010, 50)

    print("Шаг 1: Вычисление профилей и интегралов...")
    data = compute_all_profiles(alpha_vals)

    # Уточнённый диапазон K
    K_candidates = np.linspace(585, 595, 41)  # шаг ~0.25
    best_K = None
    best_diff = np.inf
    best_alpha_min = None

    print("Шаг 2: Подбор K в диапазоне [585, 595]...")
    for K in K_candidates:
        a_min = find_min_alpha(K, alpha_vals, data)
        diff = abs(a_min - alpha_exp)
        if diff < best_diff:
            best_diff = diff
            best_K = K
            best_alpha_min = a_min

    print("\n" + "="*60)
    print(f"Оптимальное K = {best_K:.3f}")
    print(f"Минимум α = {best_alpha_min:.9f}")
    print(f"Эксперимент α = {alpha_exp:.9f}")
    print(f"Отклонение: {best_diff/alpha_exp*100:.6f}%")
    print("="*60)

    # График
    alphas = []
    energies = []
    for a in alpha_vals:
        if a in data:
            alphas.append(a)
            energies.append(total_energy(a, best_K, data))
    alphas = np.array(alphas)
    energies = np.array(energies)

    plt.style.use('dark_background')
    plt.figure(figsize=(10,6))
    plt.plot(alphas, energies, 'o-', color='cyan', lw=2)
    plt.axvline(best_alpha_min, color='yellow', linestyle='--', label=f'min α={best_alpha_min:.6f}')
    plt.axvline(alpha_exp, color='red', linestyle=':', label='1/137')
    plt.xlabel('α')
    plt.ylabel('E_total')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.title(f'Энергия с K={best_K:.3f}')
    plt.tight_layout()
    plt.savefig('energy_refined.png', dpi=150)
    plt.show()
