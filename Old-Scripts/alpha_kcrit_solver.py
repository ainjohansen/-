import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import root_scalar

# =========================
# параметры
# =========================

r_min = 1e-4
r_max = 20.0
N = 2000
r_eval = np.linspace(r_min, r_max, N)

# =========================
# ODE
# =========================

def make_ode(alpha):

    def ode(r, y):
        f, df = y

        r2 = r*r + 1e-12
        sinf = np.sin(f)
        cosf = np.cos(f)
        sin2f = np.sin(2*f)

        A = 1 + 2*(sinf**2)/r2
        B = 2/r + 2*sin2f/r2

        dA_dr = (4*sinf*cosf*df)/r2 - (4*sinf**2)/(r*r2)

        rhs = (
            sin2f / r2
            + sin2f * (df**2) / r2
            - (sinf**2 * sin2f) / (r2*r2)
            + alpha * sinf
        )

        ddf = (rhs - B*df - dA_dr*df) / A

        return [df, ddf]

    return ode

# =========================
# shooting
# =========================

def shoot(alpha, s):
    sol = solve_ivp(
        make_ode(alpha),
        [r_min, r_max],
        [np.pi, s],
        t_eval=[r_max],
        rtol=1e-5,
        atol=1e-7
    )
    if not sol.success:
        return 1e3
    return sol.y[0, -1]

def find_s(alpha):
    res = root_scalar(
        lambda s: shoot(alpha, s),
        bracket=[-50, -1e-3],
        method='bisect'
    )
    return res.root

# =========================
# профиль
# =========================

def solve_profile(alpha):
    s = find_s(alpha)

    sol = solve_ivp(
        make_ode(alpha),
        [r_min, r_max],
        [np.pi, s],
        t_eval=r_eval,
        rtol=1e-5,
        atol=1e-7
    )

    r = sol.t
    f = sol.y[0]
    df = sol.y[1]

    return r, f, df

# =========================
# интегралы
# =========================

def compute_integrals(r, f, df):

    sinf = np.sin(f)
    r2 = r*r

    I2 = np.trapezoid((df**2 + 2*sinf**2/r2) * r2, r)

    I4 = np.trapezoid(
        (2*sinf**2 * df**2 / r2 + sinf**4 / (r2*r2)) * r2,
        r
    )

    I0 = np.trapezoid((1 - np.cos(f)) * r2, r)

    return I2, I4, I0

# =========================
# Деррик по k
# =========================

def derrick_k(k, I2, I4, I0, alpha):
    return I2 - k*I4 + 3*alpha*I0

# =========================
# поиск kcrit
# =========================

def find_kcrit(I2, I4, I0, alpha):

    res = root_scalar(
        lambda k: derrick_k(k, I2, I4, I0, alpha),
        bracket=[1.0, 10.0],
        method='bisect'
    )

    return res.root

# =========================
# физические величины
# =========================

def compute_observables(r, f, df, I2, I4, I0, k, alpha):

    # энергия
    E = I2 + k*I4 + alpha*I0

    # эффективный радиус (взвешенный)
    density = (df**2 + np.sin(f)**2 / r**2)
    norm = np.trapezoid(density * r**2, r)

    R_eff = np.sqrt(
        np.trapezoid(density * r**4, r) / norm
    )

    # альфа из баланса
    alpha_eff = (k*I4 - I2) / (3*I0)

    return E, R_eff, alpha_eff

# =========================
# MAIN
# =========================

if __name__ == "__main__":

    alpha_input = 0.0073  # начальное (не критично)

    print("=== SOLVING PROFILE ===")
    r, f, df = solve_profile(alpha_input)

    print("=== INTEGRALS ===")
    I2, I4, I0 = compute_integrals(r, f, df)
    print(f"I2={I2:.6f}, I4={I4:.6f}, I0={I0:.6f}")

    print("=== FIND kcrit ===")
    kcrit = find_kcrit(I2, I4, I0, alpha_input)
    print(f"kcrit = {kcrit:.6f}")

    print("=== OBSERVABLES ===")
    E, R_eff, alpha_eff = compute_observables(
        r, f, df, I2, I4, I0, kcrit, alpha_input
    )

    print(f"Energy     E = {E:.6f}")
    print(f"Radius  R_eff = {R_eff:.6f}")
    print(f"alpha_eff  = {alpha_eff:.6f}")
