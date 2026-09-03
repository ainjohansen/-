import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import root_scalar
import time

# =========================
# НАСТРОЙКИ (быстрые, но надёжные)
# =========================
r_min = 1e-4
r_max = 20.0
N = 800               # поменьше точек для скорости
r_eval = np.linspace(r_min, r_max, N)

# =========================
# ОДУ (как в рабочем alpha-bvp.py)
# =========================
def make_ode(alpha):
    def ode(r, y):
        f, df = y
        r2 = r*r + 1e-12
        sinf = np.sin(f)
        cosf = np.cos(f)
        sin2f = np.sin(2*f)
        sinf2 = sinf**2

        # Регуляризованные коэффициенты
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

# =========================
# Стрельба с таймаутом
# =========================
def shoot(alpha, s, timeout=5):
    start = time.time()
    try:
        sol = solve_ivp(
            make_ode(alpha),
            [r_min, r_max],
            [np.pi, s],
            t_eval=[r_max],
            rtol=1e-3, atol=1e-5,
            method='LSODA'   # более устойчивый
        )
        if not sol.success:
            return 1e3
        # Если решение расходится на хвосте
        if abs(sol.y[0, -1]) > 100:
            return np.sign(sol.y[0, -1]) * 100
        return sol.y[0, -1]
    except Exception as e:
        return 1e3

def find_s(alpha):
    """Находит s = f'(r_min) с проверкой нескольких диапазонов"""
    # Попробуем несколько начальных приближений
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

# =========================
# Профиль
# =========================
def solve_profile(alpha):
    s = find_s(alpha)
    sol = solve_ivp(
        make_ode(alpha),
        [r_min, r_max],
        [np.pi, s],
        t_eval=r_eval,
        rtol=1e-3, atol=1e-5,
        method='LSODA'
    )
    if not sol.success:
        raise RuntimeError("Не сошёлся профиль")
    return sol.t, sol.y[0], sol.y[1]

# =========================
# Интегралы
# =========================
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

# =========================
# ГЛАВНЫЙ ЦИКЛ (ПОСЛЕДОВАТЕЛЬНЫЙ, С ВЫВОДОМ)
# =========================
alpha_vals = np.linspace(0.005, 0.012, 15)  # 15 точек для теста
alpha_exp = 0.00729735

results = []
print(f"{'α':<10} {'I2':<8} {'I4':<8} {'I0':<8} {'k_crit':<8} {'E_core':<10}")
print("-" * 60)

for alpha in alpha_vals:
    print(f"{alpha:.6f}   ", end='', flush=True)
    try:
        r, f, df = solve_profile(alpha)
        I2, I4, I0 = compute_integrals(r, f, df)
        k = find_kcrit(I2, I4, I0, alpha)
        E_core = I2 + k*I4 + 3*alpha*I0
        results.append((alpha, E_core, k, I2, I4, I0))
        print(f"{I2:.3f}   {I4:.3f}   {I0:.3f}   {k:.3f}     {E_core:.4f}")
    except Exception as e:
        print(f"ОШИБКА: {e}")

if not results:
    print("Нет успешных решений. Проверьте параметры сетки или метод.")
    exit()

# Преобразуем в массивы
alphas = np.array([r[0] for r in results])
E_cores = np.array([r[1] for r in results])

# Оценка C (если нужно)
dalpha = alphas[1] - alphas[0]
dE = np.gradient(E_cores, dalpha)
idx_exp = np.argmin(np.abs(alphas - alpha_exp))
C = -dE[idx_exp] * alpha_exp**2

E_total = E_cores - C / alphas
min_idx = np.argmin(E_total)
best_alpha = alphas[min_idx]

print("\n" + "="*60)
print(f"Минимум E_core при α = {alphas[np.argmin(E_cores)]:.6f}")
print(f"Минимум E_total при α = {best_alpha:.6f}")
print(f"Экспериментальная α = {alpha_exp:.6f}")
print(f"Отклонение: {abs(best_alpha - alpha_exp)/alpha_exp*100:.2f}%")
print("="*60)

# Быстрый график (опционально)
import matplotlib.pyplot as plt
plt.figure(figsize=(8,5))
plt.plot(alphas, E_cores, 'o-', label='E_core')
plt.plot(alphas, E_total, 's-', label='E_total = E_core - C/α')
plt.axvline(alpha_exp, color='red', linestyle=':', label='1/137')
plt.axvline(best_alpha, color='yellow', linestyle='--', label=f'min при {best_alpha:.5f}')
plt.xlabel('α')
plt.ylabel('Энергия')
plt.legend()
plt.grid(alpha=0.3)
plt.title('Энергия хопфиона с эл.маг. добавкой')
plt.show()
