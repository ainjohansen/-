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
    # Топологическая плотность (безразмерная)
    j0_top = np.where(r > 1e-4, (sinf**2 * df) / (2 * np.pi**2 * r2), 0)
    # Заряд электрона e = sqrt(4π*α)
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

def compute_for_alpha(alpha):
    try:
        r, f, df = solve_profile(alpha)
        I2, I4, I0 = compute_integrals(r, f, df)
        k = find_kcrit(I2, I4, I0, alpha)
        E_core = I2 + k*I4 + 3*alpha*I0

        _, _, e_eff, alpha_em, E_em = solve_poisson(r, f, df, alpha)

        E_total = E_core + E_em

        return {
            'alpha': alpha,
            'E_core': E_core,
            'E_em': E_em,
            'E_total': E_total,
            'k': k,
            'e_eff': e_eff,
            'alpha_em': alpha_em,
            'I2': I2, 'I4': I4, 'I0': I0
        }
    except Exception as e:
        print(f"Ошибка для α={alpha:.6f}: {e}")
        return None

if __name__ == "__main__":
    alpha_vals = np.linspace(0.005, 0.012, 20)
    alpha_exp = 0.00729735

    print(f"Запуск на {cpu_count()} ядрах для {len(alpha_vals)} α...")
    with Pool(processes=cpu_count()) as pool:
        results = pool.map(compute_for_alpha, alpha_vals)

    results = [res for res in results if res is not None]
    if not results:
        print("Нет успешных решений!")
        exit()

    results.sort(key=lambda x: x['alpha'])
    alphas = np.array([r['alpha'] for r in results])
    E_cores = np.array([r['E_core'] for r in results])
    E_ems = np.array([r['E_em'] for r in results])
    E_totals = np.array([r['E_total'] for r in results])

    min_idx = np.argmin(E_totals)
    best_alpha = alphas[min_idx]

    print("\n" + "="*60)
    print("РЕЗУЛЬТАТЫ С МАСШТАБИРОВАННЫМ ЗАРЯДОМ")
    print("="*60)
    for i in range(len(alphas)):
        print(f"α={alphas[i]:.6f}: E_core={E_cores[i]:.4f}, E_em={E_ems[i]:.4f}, E_total={E_totals[i]:.4f}")
    print("="*60)
    print(f"Минимум E_total при α = {best_alpha:.6f}")
    print(f"Экспериментальная α = {alpha_exp:.6f}")
    print(f"Отклонение: {abs(best_alpha - alpha_exp)/alpha_exp*100:.2f}%")
    print("="*60)

    plt.style.use('dark_background')
    plt.figure(figsize=(12,5))
    plt.subplot(1,2,1)
    plt.plot(alphas, E_cores, 'o-', label='E_core')
    plt.plot(alphas, E_totals, 's-', label='E_total')
    plt.axvline(best_alpha, color='yellow', linestyle='--', label=f'min {best_alpha:.5f}')
    plt.axvline(alpha_exp, color='red', linestyle=':', label='1/137')
    plt.xlabel('α')
    plt.ylabel('Энергия')
    plt.legend()
    plt.grid(alpha=0.3)

    plt.subplot(1,2,2)
    plt.plot(alphas, E_ems, 'o-', color='magenta', label='E_em')
    plt.xlabel('α')
    plt.ylabel('E_em')
    plt.legend()
    plt.grid(alpha=0.3)

    plt.suptitle('Энергия с правильной зависимостью E_em ~ α')
    plt.tight_layout()
    plt.savefig('energy_correct_em.png', dpi=150)
    plt.show()
