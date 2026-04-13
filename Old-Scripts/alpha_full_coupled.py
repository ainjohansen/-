import numpy as np
import time
from scipy.integrate import solve_ivp
from scipy.optimize import root_scalar

# =========================
# параметры
# =========================

r_min = 1e-4
r_max = 20.0
N = 1500
r_eval = np.linspace(r_min, r_max, N)

alpha_input = 0.0073

# =========================
# ODE (с учётом twist)
# =========================

def make_ode(alpha, C):

    def ode(r, y):
        f, df = y

        r2 = r*r + 1e-12
        sinf = np.sin(f)
        cosf = np.cos(f)
        sin2f = np.sin(2*f)

        # регуляризация
        sinf2 = sinf*sinf + (r*r)

        A = 1 + 2*(sinf2)/r2
        B = 2/r + 2*sin2f/r2

        dA_dr = (4*sinf*cosf*df)/r2 - (4*sinf2)/(r*r2)

        # стандартная часть
        rhs = (
            sin2f / r2
            + sin2f * (df**2) / r2
            - (sinf2 * sin2f) / (r2*r2)
            + alpha * sinf
        )

        # ===== twist вклад =====
        psi_sq = (C*C) / (r2*r2 * sinf2*sinf2 + 1e-8)   # psi'^2
        rhs += sin2f * psi_sq

        ddf = (rhs - B*df - dA_dr*df) / A

        return [df, ddf]

    return ode

# =========================
# shooting
# =========================

def shoot(alpha, C, s):
    sol = solve_ivp(
        make_ode(alpha, C),
        [r_min, r_max],
        [np.pi, s],
        t_eval=[r_max],
        rtol=1e-5,
        atol=1e-7
    )

    if not sol.success:
        return np.nan

    val = sol.y[0, -1]

    if np.isnan(val) or abs(val) > 100:
        return np.sign(val) * 100

    return val

def find_s(alpha, C):

    print(f"  [shoot] scanning bracket...")

    s_vals = np.linspace(-50, -0.001, 40)
    vals = []

    for s in s_vals:
        v = shoot(alpha, C, s)
        vals.append(v)
        print(f"    s={s:.3f} -> {v:.3e}")

    for i in range(len(s_vals)-1):
        if np.sign(vals[i]) != np.sign(vals[i+1]):
            print(f"  [shoot] bracket found!")
            return root_scalar(
                lambda s: shoot(alpha, C, s),
                bracket=[s_vals[i], s_vals[i+1]],
                method='bisect'
            ).root

    raise RuntimeError("❌ no valid bracket for shooting")

# =========================
# профиль
# =========================

def solve_profile(alpha, C):

    t0 = time.time()

    print(f"  [profile] solving ODE (C={C:.3e})")

    s = find_s(alpha, C)

    sol = solve_ivp(
        make_ode(alpha, C),
        [r_min, r_max],
        [np.pi, s],
        t_eval=r_eval,
        rtol=1e-4,   # быстрее
        atol=1e-6
    )

    dt = time.time() - t0
    print(f"  [profile] done in {dt:.2f}s")

    return sol.t, sol.y[0], sol.y[1]

# =========================
# интегралы
# =========================

def compute_integrals(r, f, df, C):

    sinf = np.sin(f)
    r2 = r*r + 1e-12

    I2 = np.trapezoid((df**2 + 2*sinf**2/r2) * r2, r)

    I4 = np.trapezoid(
        (2*sinf**2 * df**2 / r2 + sinf**4 / (r2*r2)) * r2,
        r
    )

    I0 = np.trapezoid((1 - np.cos(f)) * r2, r)

    # twist
    sinf2 = sinf*sinf + 1e-12
    psi_sq = (C*C) / (r2*r2 * sinf2*sinf2)
    I_twist = np.trapezoid(sinf2 * psi_sq * r2, r)

    return I2, I4, I0, I_twist

# =========================
# Деррик
# =========================

def derrick(C):

    print(f"\n[derrick] C={C:.6e}")

    t0 = time.time()

    r, f, df = solve_profile(alpha_input, C)

    I2, I4, I0, I_twist = compute_integrals(r, f, df, C)

    k_eff = 1 + I_twist / I4
    R = I2 - k_eff * I4 + 3 * alpha_input * I0

    dt = time.time() - t0

    print(f"  I2={I2:.4f} I4={I4:.4f} I_twist={I_twist:.4f}")
    print(f"  k_eff={k_eff:.4f}  R={R:.4e}")
    print(f"  time={dt:.2f}s")

    return R, k_eff, I2, I4, I0, I_twist

# =========================
# поиск C*
# =========================

def find_C_star():

    print("\n=== ROOT SEARCH START ===\n")

    def func(C):
        R, *_ = derrick(C)
        return R

    res = root_scalar(
        func,
        bracket=[1e-6, 1e-2],
        method='bisect',
        xtol=1e-5
    )

    print("\n=== ROOT FOUND ===")
    return res.root

# =========================
# MAIN
# =========================

if __name__ == "__main__":

    print("=== FIND C* (FULL COUPLED) ===")
    C_star = find_C_star()

    print(f"\nC* = {C_star:.8e}")

    R, k_eff, I2, I4, I0, I_twist = derrick(C_star)

    alpha_eff = (k_eff * I4 - I2) / (3 * I0)

    print("\n=== RESULTS ===")
    print(f"I2={I2:.6f}, I4={I4:.6f}, I0={I0:.6f}")
    print(f"I_twist={I_twist:.6f}")
    print(f"k_eff={k_eff:.6f}")
    print(f"R={R:.6e}")
    print(f"alpha_eff={alpha_eff:.6f}")
