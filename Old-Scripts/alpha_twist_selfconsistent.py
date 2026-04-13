import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import root_scalar

# =========================
# параметры
# =========================

r_min = 1e-4
r_max = 20.0
N = 3000
r_eval = np.linspace(r_min, r_max, N)

alpha_input = 0.0073

# =========================
# ODE для f(r)
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

    return sol.t, sol.y[0], sol.y[1]

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
# twist
# =========================

def compute_I_twist(r, f, C):

    sinf = np.sin(f)
    r2 = r*r + 1e-12

    # избегаем деления на 0
    denom = (r2 * sinf**2 + 1e-12)

    psi_prime = C / denom

    integrand = sinf**2 * psi_prime**2 * r2

    I_twist = np.trapezoid(integrand, r)

    return I_twist

# =========================
# Деррик
# =========================

def derrick(C, r, f, df, I2, I4, I0, alpha):

    I_twist = compute_I_twist(r, f, C)

    k_eff = 1 + I_twist / I4

    R = I2 - k_eff * I4 + 3 * alpha * I0

    return R, k_eff, I_twist

# =========================
# поиск C*
# =========================

def find_C_star(r, f, df, I2, I4, I0, alpha):

    def func(C):
        R, _, _ = derrick(C, r, f, df, I2, I4, I0, alpha)
        return R

    res = root_scalar(func, bracket=[1e-6, 50], method='bisect')

    return res.root

# =========================
# MAIN
# =========================

if __name__ == "__main__":

    print("=== SOLVE PROFILE ===")
    r, f, df = solve_profile(alpha_input)

    print("=== INTEGRALS ===")
    I2, I4, I0 = compute_integrals(r, f, df)
    print(f"I2={I2:.6f}, I4={I4:.6f}, I0={I0:.6f}")

    print("=== FIND C* ===")
    C_star = find_C_star(r, f, df, I2, I4, I0, alpha_input)
    print(f"C* = {C_star:.6f}")

    R, k_eff, I_twist = derrick(C_star, r, f, df, I2, I4, I0, alpha_input)

    print("=== RESULTS ===")
    print(f"I_twist = {I_twist:.6f}")
    print(f"k_eff   = {k_eff:.6f}")
    print(f"R       = {R:.6e}")

    alpha_eff = (k_eff * I4 - I2) / (3 * I0)

    print(f"alpha_eff = {alpha_eff:.6f}")
