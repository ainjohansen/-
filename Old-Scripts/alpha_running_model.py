import numpy as np
from scipy.integrate import cumulative_trapezoid
import matplotlib.pyplot as plt

# =========================
# 1. ПАРАМЕТРЫ
# =========================

r_max = 20.0
N = 2000
r = np.linspace(1e-4, r_max, N)
dr = r[1] - r[0]

# =========================
# 2. ПРОФИЛЬ f(r)
# =========================
# !!! ЗАМЕНИ НА СВОЙ ЧИСЛЕННЫЙ ПРОФИЛЬ !!!

def f_profile(r):
    # тестовый гладкий профиль (убывающий)
    return np.pi * np.exp(-r)

def df_dr(r):
    f = f_profile(r)
    return -np.pi * np.exp(-r)

f = f_profile(r)
df = df_dr(r)

# =========================
# 3. ТОПОЛОГИЧЕСКАЯ ПЛОТНОСТЬ j0
# =========================

def j0_density(r, f, df):
    return (np.sin(f)**2 * df) / (2 * np.pi**2 * r**2)

j0 = j0_density(r, f, df)

# нормировка заряда
Q = 4 * np.pi * np.trapz(j0 * r**2, r)
print(f"Topological charge Q ≈ {Q:.6f}")

# =========================
# 4. РЕШЕНИЕ ПУАССОНА (РАДИАЛЬНО)
# =========================
# d/dr (r^2 dA/dr) = - e_eff * r^2 j0
# сначала решаем без e_eff (берём =1), потом нормируем

# интеграл RHS
rhs = - j0 * r**2
I1 = cumulative_trapezoid(rhs, r, initial=0)

# dA/dr
dA_dr = I1 / (r**2)

# A(r)
A = cumulative_trapezoid(dA_dr, r, initial=0)

# =========================
# 5. ИЗВЛЕЧЕНИЕ e_eff
# =========================
# A(r) ~ e_eff / (4π r)

# берём хвост
tail_start = int(0.8 * N)
r_tail = r[tail_start:]
A_tail = A[tail_start:]

# линейная регрессия A ~ k / r
inv_r = 1.0 / r_tail
coef = np.polyfit(inv_r, A_tail, 1)

k = coef[0]  # A ≈ k*(1/r)
e_eff = 4 * np.pi * k

alpha = e_eff**2 / (4 * np.pi)

print(f"e_eff ≈ {e_eff:.6f}")
print(f"alpha ≈ {alpha:.6f}  (compare 1/137 ≈ {1/137:.6f})")

# =========================
# 6. RUNNING α(μ)
# =========================

def alpha_running(mu, alpha0, beta0, Lambda):
    return alpha0 / (1 + alpha0 * 4*np.pi * beta0 * np.log(Lambda**2 / mu**2))

# масштаб ядра
R_core = 1.0
Lambda = 1.0 / R_core

# оценка beta0 (грубая, можно заменить на численную)
K2 = np.trapz((df**2 + 2*np.sin(f)**2 / r**2) * r**2, r)
beta0 = K2 / (24 * np.pi**2)

print(f"beta0 ≈ {beta0:.6f}")

# диапазон масштабов
mus = np.logspace(-2, 1, 100)
alphas = alpha_running(mus, alpha, beta0, Lambda)

# =========================
# 7. ГРАФИК
# =========================

plt.figure(figsize=(6,4))
plt.semilogx(mus, alphas)
plt.axhline(1/137, linestyle='--')
plt.xlabel("μ (scale)")
plt.ylabel("α(μ)")
plt.title("Running alpha from nonlinear vacuum model")
plt.grid(True)
plt.show()
