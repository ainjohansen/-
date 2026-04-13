import numpy as np
from scipy.integrate import solve_ivp, cumulative_trapezoid
from scipy.optimize import root_scalar
import matplotlib.pyplot as plt
from multiprocessing import Pool, cpu_count

# =========================
# ГЛОБАЛЬНЫЕ ПАРАМЕТРЫ
# =========================
r_min = 1e-4
r_max = 20.0
N = 1200
r_eval = np.linspace(r_min, r_max, N)

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
    A0 = cumulative_trapezoid(dA_dr, r, initial=0)

    tail = int(0.7 * len(r))
    r_tail = r[tail:]
    A_tail = A0[tail:]
    inv_r = 1.0 / r_tail
    coeffs = np.polyfit(inv_r, A_tail, 1)
    k = coeffs[0]
    e_eff = 4 * np.pi * k
    alpha_em = e_eff**2 / (4 * np.pi)

    E_em = 2 * np.pi * np.trapezoid(dA_dr**2 * r2, r)
    return A0, dA_dr, e_eff, alpha_em, E_em

def compute_for_alpha(alpha, K=1.0):
    try:
        r, f, df = solve_profile(alpha)
        I2, I4, I0 = compute_integrals(r, f, df)
        k = find_kcrit(I2, I4, I0, alpha)
        E_core = I2 + k*I4 + 3*alpha*I0
        _, _, _, _, E_em = solve_poisson(r, f, df, alpha)
        E_total = E_core + K * E_em
        return {'alpha': alpha, 'E_total': E_total, 'E_core': E_core, 'E_em': E_em}
    except:
        return None

def find_best_alpha(K, alpha_vals):
    with Pool(processes=cpu_count()) as pool:
        results = pool.starmap(compute_for_alpha, [(a, K) for a in alpha_vals])
    results = [r for r in results if r is not None]
    if not results:
        return None
    results.sort(key=lambda x: x['alpha'])
    alphas = np.array([r['alpha'] for r in results])
    E_totals = np.array([r['E_total'] for r in results])
    idx = np.argmin(E_totals)
    return alphas[idx]

# =========================
# ПОДБОР K ДЛЯ СОВПАДЕНИЯ С 1/137
# =========================
if __name__ == "__main__":
    alpha_vals = np.linspace(0.005, 0.012, 20)
    alpha_exp = 0.00729735

    # Функция невязки: best_alpha(K) - alpha_exp
    def residual(K):
        best = find_best_alpha(K, alpha_vals)
        if best is None:
            return 1.0
        return best - alpha_exp

    print("Подбор коэффициента K...")
    # Ищем K в диапазоне [100, 10000]
    try:
        sol = root_scalar(residual, bracket=[100, 10000], method='bisect', xtol=1e-3)
        K_opt = sol.root
        print(f"Оптимальное K = {K_opt:.2f}")
    except:
        # Если не получилось, грубая оценка
        K_opt = 1500
        print(f"Не удалось точно найти K, используем K={K_opt}")

    # Финальный расчёт с оптимальным K
    with Pool(processes=cpu_count()) as pool:
        results = pool.starmap(compute_for_alpha, [(a, K_opt) for a in alpha_vals])
    results = [r for r in results if r is not None]
    results.sort(key=lambda x: x['alpha'])
    alphas = np.array([r['alpha'] for r in results])
    E_totals = np.array([r['E_total'] for r in results])
    E_cores = np.array([r['E_core'] for r in results])
    E_ems = np.array([r['E_em'] for r in results])

    min_idx = np.argmin(E_totals)
    best_alpha = alphas[min_idx]

    print("\n" + "="*60)
    print(f"При K = {K_opt:.2f}:")
    print(f"Минимум E_total при α = {best_alpha:.6f}")
    print(f"Экспериментальная α = {alpha_exp:.6f}")
    print(f"Отклонение: {abs(best_alpha - alpha_exp)/alpha_exp*100:.2f}%")
    print("="*60)

    # График
    plt.style.use('dark_background')
    plt.figure(figsize=(10,6))
    plt.plot(alphas, E_totals, 'o-', color='lime', lw=2, label=f'E_total (K={K_opt:.1f})')
    plt.axvline(best_alpha, color='yellow', linestyle='--', label=f'Минимум α={best_alpha:.5f}')
    plt.axvline(alpha_exp, color='red', linestyle=':', label=f'1/137 = {alpha_exp:.5f}')
    plt.xlabel('α')
    plt.ylabel('Полная энергия')
    plt.title('Энергия хопфиона с масштабированной электромагнитной добавкой')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig('energy_with_K_opt.png', dpi=150)
    plt.show()
