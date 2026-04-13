import numpy as np
from scipy.integrate import solve_bvp

# =========================
# параметры
# =========================

r_min = 1e-4
r_max = 20.0
N = 800

r = np.linspace(r_min, r_max, N)

alpha = 0.0073

# =========================
# система ODE
# y = [f, df]
# p = [C]
# =========================

def ode(r, y, p):

    C = p[0]

    f = y[0]
    df = y[1]

    r2 = r*r + 1e-12

    sinf = np.sin(f)
    cosf = np.cos(f)
    sin2f = np.sin(2*f)

    # правильная регуляризация
    sinf2 = sinf*sinf + r*r

    A = 1 + 2*sinf2/r2
    B = 2/r + 2*sin2f/r2

    dA = (4*sinf*cosf*df)/r2 - (4*sinf2)/(r*r2)

    # базовая часть
    rhs = (
        sin2f/r2
        + sin2f*(df**2)/r2
        - (sinf2*sin2f)/(r2*r2)
        + alpha*sinf
    )

    # twist
    psi_sq = (C*C)/(r2*r2*sinf2*sinf2 + 0.1**4)

    rhs += sin2f * psi_sq

    ddf = (rhs - B*df - dA*df) / A

    return np.vstack([df, ddf])

# =========================
# граничные условия
# =========================

def bc(ya, yb, p):

    f0, df0 = ya
    fR, dfR = yb

    C = p[0]

    # 3 условия:
    # f(0)=pi
    # f(R)=0
    # df(R)=0 (гладкий хвост)

    return np.array([
        f0 - np.pi,
        fR,
        dfR
    ])

# =========================
# начальное приближение
# =========================

f_guess = np.pi * np.exp(-r)
df_guess = -np.pi * np.exp(-r)

y_guess = np.vstack([f_guess, df_guess])

# начальный guess для C
p_guess = np.array([1e-3])

# =========================
# решение
# =========================

print("=== SOLVING BVP ===")

sol = solve_bvp(
    ode,
    bc,
    r,
    y_guess,
    p_guess,
    max_nodes=20000,
    tol=1e-3,
    verbose=2
)

if not sol.success:
    print("❌ BVP не сошёлся")
    exit()

print("\n=== BVP SOLVED ===")

r = sol.x
f = sol.y[0]
df = sol.y[1]
C = sol.p[0]

print(f"C = {C:.6e}")

# =========================
# интегралы
# =========================

def compute_integrals(r, f, df, C):

    sinf = np.sin(f)
    r2 = r*r + 1e-12

    I2 = np.trapezoid((df**2 + 2*sinf**2/r2) * r2, r)

    I4 = np.trapezoid(
        (2*sinf**2 * df**2 / r2 + sinf**4/(r2*r2)) * r2,
        r
    )

    I0 = np.trapezoid((1 - np.cos(f)) * r2, r)

    sinf2 = sinf*sinf + r*r
    psi_sq = (C*C)/(r2*r2*sinf2*sinf2 + 1e-8)

    I_twist = np.trapezoid(sinf2 * psi_sq * r2, r)

    return I2, I4, I0, I_twist

I2, I4, I0, I_twist = compute_integrals(r, f, df, C)

k_eff = 1 + I_twist / I4

R = I2 - k_eff * I4 + 3 * alpha * I0

alpha_eff = (k_eff * I4 - I2) / (3 * I0)

print("\n=== RESULTS ===")
print(f"I2={I2:.6f}")
print(f"I4={I4:.6f}")
print(f"I0={I0:.6f}")
print(f"I_twist={I_twist:.6f}")
print(f"k_eff={k_eff:.6f}")
print(f"R={R:.6e}")
print(f"alpha_eff={alpha_eff:.6f}")
