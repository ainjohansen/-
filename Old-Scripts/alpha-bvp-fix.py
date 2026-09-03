import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import root_scalar

# =========================
# 0. ПАРАМЕТРЫ
# =========================

r_min = 1e-4
r_max = 20.0
N = 2000
r_eval = np.linspace(r_min, r_max, N)

# =========================
# 1. УРАВНЕНИЕ ИЗ ФУНКЦИОНАЛА
# I2 + I4 + alpha I0
# =========================

def make_ode(alpha):

    def ode(r, y):
        f, df = y

        # избегаем деления на 0
        r2 = r*r + 1e-12
        sinf = np.sin(f)
        cosf = np.cos(f)
        sin2f = np.sin(2*f)

        # ---- коэффициенты из вариации ----
        A = 1 + 2*(sinf**2)/r2
        B = 2/r + 2*sin2f/r2

        # производная "жёсткости"
        dA_dr = (4*sinf*cosf*df)/r2 - (4*sinf**2)/(r*r2)

        # правая часть
        rhs = (
            sin2f / r2                 # sigma-модель
            + sin2f * (df**2) / r2     # Skyrme
            - (sinf**2 * sin2f) / (r2*r2)
            + alpha * sinf             # массовый член
        )

        # итоговое уравнение
        ddf = (rhs - B*df - dA_dr*df) / A

        return [df, ddf]

    return ode

# =========================
# 2. SHOOTING
# =========================

def shoot(alpha, s):
    sol = solve_ivp(
        make_ode(alpha),
        [r_min, r_max],
        [np.pi, s],
        t_eval=[r_max],
        rtol=1e-5,
        atol=1e-7,
        method='RK45'
    )
    if not sol.success:
        return 1e3
    return sol.y[0, -1]   # f(r_max)

def find_s(alpha):
    try:
        res = root_scalar(
            lambda s: shoot(alpha, s),
            bracket=[-50, -1e-3],
            method='bisect',
            xtol=1e-5
        )
        return res.root
    except:
        return None

# =========================
# 3. РЕШЕНИЕ ПРОФИЛЯ
# =========================

def solve_profile(alpha):

    s = find_s(alpha)
    if s is None:
        print(f"[alpha={alpha:.6f}] ❌ shooting failed")
        return None

    print(f"[alpha={alpha:.6f}] s = {s:.6f}")

    sol = solve_ivp(
        make_ode(alpha),
        [r_min, r_max],
        [np.pi, s],
        t_eval=r_eval,
        rtol=1e-5,
        atol=1e-7,
        method='RK45'
    )

    if not sol.success:
        print("❌ integration failed")
        return None

    r = sol.t
    f = sol.y[0]
    df = sol.y[1]

    # сохранить профиль
    np.savez(f"profile_{alpha:.6f}.npz", r=r, f=f, df=df)

    return r, f, df

# =========================
# 4. ИНТЕГРАЛЫ
# =========================

def compute_integrals(r, f, df):

    sinf = np.sin(f)
    r2 = r*r

    I2 = np.trapezoid((df**2 + 2*sinf**2/r2) * r2, r)

    I4 = 4.0 *  np.trapezoid(
        (2*sinf**2 * df**2 / r2 + sinf**4 / (r2*r2)) * r2,
        r
    )

    I0 = np.trapezoid((1 - np.cos(f)) * r2, r)

    return I2, I4, I0

# =========================
# 5. ДЕРРИК
# =========================

def derrick_residual(alpha):

    res = solve_profile(alpha)
    if res is None:
        return None

    r, f, df = res

    I2, I4, I0 = compute_integrals(r, f, df)

    R = I2 - I4 + 3*alpha*I0

    print(f"I2={I2:.6f}  I4={I4:.6f}  I0={I0:.6f}  R={R:.6e}")

    return R

# =========================
# 6. СКАН ПО alpha
# =========================

def scan_alpha():

    alphas = np.linspace(0.001, 0.02, 10)

    results = []

    print("\n=== SCAN START ===\n")

    for a in alphas:
        R = derrick_residual(a)
        if R is not None:
            results.append((a, R))
            print(f"alpha={a:.6f} | R={R:.6e}")
        else:
            print(f"alpha={a:.6f} ❌ fail")

    return results

# =========================
# MAIN
# =========================

if __name__ == "__main__":

    results = scan_alpha()

    print("\n=== DONE ===")
