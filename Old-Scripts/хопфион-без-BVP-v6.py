import numpy as np
from scipy.integrate import simpson
from scipy.optimize import minimize
import matplotlib.pyplot as plt

alpha = 0.0072973525693

# ------------------------------------------------------------
# 1. АНАЛИТИЧЕСКИЙ АНЗАЦ (ДВА ПАРАМЕТРА)
# ------------------------------------------------------------
def f_ansatz(r, p, q):
    """f(r) = pi * tanh(p*r)/r * exp(-q*r)"""
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.where(r < 1e-8, np.pi * p,
                        np.pi * np.tanh(p * r) / r * np.exp(-q * r))

# ------------------------------------------------------------
# 2. ВЫЧИСЛЕНИЕ ВСЕХ ЭНЕРГЕТИЧЕСКИХ ВКЛАДОВ
# ------------------------------------------------------------
def compute_energy_and_balance(p, q, alpha_val, n_int=5000):
    r = np.linspace(1e-6, 80.0, n_int)
    f = f_ansatz(r, p, q)
    fp = np.gradient(f, r)
    sin_f = np.sin(f)
    sin_f_r = np.where(r < 1e-8, -fp, sin_f / r)

    # Плотности I2, I4, I0
    I2_dens = r**2 * fp**2 + 2 * sin_f**2
    I4_dens = sin_f**2 * (2 * fp**2 + sin_f_r**2)
    I0_dens = (1 - np.cos(f)) * r**2

    I2 = simpson(4 * np.pi * I2_dens, x=r)
    I4 = simpson(4 * np.pi * I4_dens, x=r)
    I0 = simpson(4 * np.pi * I0_dens, x=r)

    # Топологическая плотность заряда
    rho_top = (1/(4*np.pi)) * 2 * fp * sin_f**2 / (r**2 + 1e-12)

    # Вычисление кулоновской энергии через интегралы
    # Q(r) = ∫₀ʳ 4π x² ρ(x) dx
    integrand_Q = 4 * np.pi * r**2 * rho_top
    Q = np.zeros_like(r)
    for i in range(1, len(r)):
        Q[i] = simpson(integrand_Q[:i+1], x=r[:i+1])

    # Потенциал Φ(r) = α ∫_r^∞ Q(x)/x² dx
    integrand_Phi = Q / r**2
    Phi = np.zeros_like(r)
    for i in range(len(r)-1, -1, -1):
        Phi[i] = alpha_val * simpson(integrand_Phi[i:], x=r[i:])

    # Кулоновская энергия: E_c = ½ ∫ ρ Φ dV
    E_coulomb = 0.5 * simpson(4 * np.pi * r**2 * rho_top * Phi, x=r)

    total = I2 + I4 + 3 * alpha_val * I0 + E_coulomb
    return total, I2, I4, I0, E_coulomb

# ------------------------------------------------------------
# 3. ПОИСК ОПТИМАЛЬНЫХ ПАРАМЕТРОВ ДЛЯ ЗАДАННОГО α
# ------------------------------------------------------------
def optimize_parameters(alpha_val):
    def objective(params):
        p, q = params
        E, _, _, _, _ = compute_energy_and_balance(p, q, alpha_val)
        return E
    res = minimize(objective, [1.0, 0.5], method='Nelder-Mead',
                   bounds=[(0.5, 3.0), (0.1, 1.5)])
    return res.x

# ------------------------------------------------------------
# 4. ПОСТРОЕНИЕ ЗАВИСИМОСТИ ЭНЕРГИИ ОТ α
# ------------------------------------------------------------
alphas = np.linspace(0.005, 0.015, 20)
energies = []
for a in alphas:
    p_opt, q_opt = optimize_parameters(a)
    E, I2, I4, I0, Ec = compute_energy_and_balance(p_opt, q_opt, a)
    energies.append(E)
    print(f"α = {a:.6f} | p={p_opt:.4f}, q={q_opt:.4f} | E={E:.4f} | I4-I2={I4-I2:.4f}, 3αI0={3*a*I0:.4f}")

# График
plt.figure(figsize=(8,5))
plt.plot(alphas, energies, 'b-o')
plt.axvline(alpha, color='r', linestyle='--', label=f'Наша α = {alpha:.6f}')
plt.xlabel('α')
plt.ylabel('Минимальная энергия')
plt.title('Адаптация электрона-хопфиона к разным вселенным')
plt.legend()
plt.grid(alpha=0.3)
plt.savefig('adaptation_energy_vs_alpha.png', dpi=150)
plt.show()
