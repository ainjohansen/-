import numpy as np
from scipy.integrate import solve_bvp, quad
from scipy.optimize import brentq
import matplotlib.pyplot as plt

# ============================================================
# ПАРАМЕТРЫ
# ============================================================

r_min = 1e-4
r_max = 50

# логарифмическая сетка
r = np.logspace(np.log10(r_min), np.log10(r_max), 800)

TAU = 0.2
W = 3
L = 0.5   # <-- КРИТИЧНО: регуляризация ядра


# ============================================================
# ОДУ
# ============================================================

def make_ode(alpha, tau=TAU, W=W, L=L):

    def ode(r, y):
        f, fp = y

        sin_f = np.sin(f)
        cos_f = np.cos(f)
        sin2 = sin_f**2

        denom = r**2 + 2 * sin2 + 1e-12

        term1 = 2 * sin_f * cos_f * (1 - fp**2 + sin2 / (r**2 + 1e-12))
        term2 = (alpha / 2) * r**2 * sin_f
        term3 = -2 * r * fp

        # --- СТАБИЛЬНЫЙ TWIST ---
        twist = tau * W**2 * sin_f**2 / (r**2 + L**2)

        fpp = (term1 + term2 + term3 + twist) / denom

        return np.vstack((fp, fpp))

    return ode


# ============================================================
# ГРАНИЧНЫЕ УСЛОВИЯ
# ============================================================

def bc(ya, yb):
    return np.array([
        ya[0] - np.pi,
        yb[0]
    ])


# ============================================================
# НАЧАЛЬНОЕ ПРИБЛИЖЕНИЕ
# ============================================================

def initial_guess(alpha):
    A = 0.8
    B = np.sqrt(alpha + 1e-6)

    u = (A / r) * np.exp(-B * r)
    f = 2 * np.arctan(u)

    fp = (2 * u / (1 + u**2)) * (-1/r - B)

    return np.vstack((f, fp))


# ============================================================
# ИНТЕГРАЛЫ
# ============================================================

def compute_integrals(sol, tau=TAU, W=W, L=L):

    def I2(r):
        y = sol.sol(r)
        sin_f = np.sin(y[0])

        return (
            r**2 * y[1]**2 +
            2 * sin_f**2 +
            tau * W**2 * sin_f**2 / (r**2 + L**2)
        )

    def I4(r):
        y = sol.sol(r)
        sin_f = np.sin(y[0])

        if r > 1e-8:
            sin_f_r = sin_f / r
        else:
            sin_f_r = -y[1]

        return sin_f**2 * (2 * y[1]**2 + sin_f_r**2)

    def I0(r):
        y = sol.sol(r)
        return (1 - np.cos(y[0])) * r**2

    I2_val, _ = quad(I2, r_min, r_max, limit=300)
    I4_val, _ = quad(I4, r_min, r_max, limit=300)
    I0_val, _ = quad(I0, r_min, r_max, limit=300)

    return I2_val, I4_val, I0_val


# ============================================================
# ДЕРРИК НЕВЯЗКА
# ============================================================

def derrick_residual(alpha, verbose=True):

    ode = make_ode(alpha)
    y_guess = initial_guess(alpha)

    sol = solve_bvp(
        ode,
        bc,
        r,
        y_guess,
        tol=2e-4,
        max_nodes=30000
    )

    if not sol.success:
        if verbose:
            print(f"α={alpha:.6f} ❌ no convergence")
        return np.nan

    I2, I4, I0 = compute_integrals(sol)

    R = I4 - I2 - 3 * alpha * I0

    if verbose:
        print(f"α={alpha:.6f} | R={R:.6e}")

    return R


# ============================================================
# СКАН
# ============================================================

print("\n=== SCAN START ===\n")

alphas = np.linspace(0.003, 0.015, 25)
R_vals = []

for a in alphas:
    R = derrick_residual(a, verbose=True)
    R_vals.append(R)


# ============================================================
# ОЧИСТКА
# ============================================================

alphas_clean = []
R_clean = []

for a, R in zip(alphas, R_vals):
    if not np.isnan(R):
        alphas_clean.append(a)
        R_clean.append(R)

alphas_clean = np.array(alphas_clean)
R_clean = np.array(R_clean)


# ============================================================
# ПОИСК КОРНЯ
# ============================================================

print("\n=== ROOT SEARCH ===\n")

root_alpha = None

for i in range(len(R_clean) - 1):
    if R_clean[i] * R_clean[i + 1] < 0:
        root_alpha = brentq(
            lambda a: derrick_residual(a, verbose=False),
            alphas_clean[i],
            alphas_clean[i + 1]
        )
        break

if root_alpha:
    print(f"\n🎯 НАЙДЕНО α = {root_alpha:.10f}")
else:
    print("\n❌ Корень не найден")


# ============================================================
# ГРАФИК
# ============================================================

plt.figure(figsize=(8, 5))
plt.plot(alphas_clean, R_clean, 'o-', label="R(alpha)")
plt.axhline(0, linestyle='--')

if root_alpha:
    plt.axvline(root_alpha, linestyle='--', label=f"α={root_alpha:.6f}")

plt.xlabel("alpha")
plt.ylabel("Derrick residual")
plt.title(f"R(alpha), tau={TAU}, W={W}, L={L}")
plt.legend()
plt.grid()
plt.show()
