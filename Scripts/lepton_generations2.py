import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
from scipy.optimize import minimize_scalar

# Константы (из предыдущих расчётов)
alpha = 0.00729735
m_e_MeV = 0.51099895
R_e = 386.159  # фм
a = np.sqrt(alpha) * R_e  # масштаб ядра, фм

# Безразмерный функционал энергии (из B.2.1)
def energy_functional(f, rho, df):
    # f(rho), df = f'(rho)
    sinf = np.sin(f)
    term2 = (df**2 + 2*sinf**2/rho**2) * rho**2
    term4 = sinf**2 * (2*df**2 + (sinf/rho)**2) * rho**2
    term0 = (1 - np.cos(f)) * rho**2
    return term4 + term2 + 3*alpha*term0

def integrate_energy(f_profile, rho_grid):
    # Численное интегрирование по сетке
    E_dimless = 0
    for i in range(len(rho_grid)-1):
        rho_mid = (rho_grid[i] + rho_grid[i+1])/2
        drho = rho_grid[i+1] - rho_grid[i]
        f_mid = (f_profile[i] + f_profile[i+1])/2
        df_mid = (f_profile[i+1] - f_profile[i]) / drho
        E_dimless += energy_functional(f_mid, rho_mid, df_mid) * drho
    return E_dimless

def trial_f(rho, W, r0, b):
    # Пробная функция с W намотками: f(0)=π, f(∞)=0, делает W-1 пересечений π/2
    # Используем форму: f = π * (1 - tanh(b*(rho - r0))^W) или лучше:
    # f = π * (1 / (1 + (rho/r0)^(2W))) * exp(-b*rho)
    # Но для W>1 нужны дополнительные узлы, проще взять:
    # f = π * (1 - (rho/(rho+r0))^W) * exp(-b*rho)
    # При малых rho: f ~ π - (π W / r0)*rho + ..., что даёт правильный наклон
    # При больших: экспоненциальный спад.
    return np.pi * (1 - (rho/(rho + r0))**W) * np.exp(-b*rho)

def compute_mass(W, r0_guess=1.0, b_guess=0.5):
    rho_grid = np.logspace(-5, 2, 1000)  # от 1e-5 до 100
    # Минимизация по параметрам r0, b
    def objective(params):
        r0, b = params
        f_vals = trial_f(rho_grid, W, r0, b)
        # Проверка граничных условий
        if f_vals[0] < 2.5 or f_vals[-1] > 0.1:
            return 1e10
        E = integrate_energy(f_vals, rho_grid)
        return E
    # Грубая оптимизация (можно использовать minimize)
    from scipy.optimize import minimize
    res = minimize(objective, [r0_guess, b_guess], method='Nelder-Mead', bounds=[(0.2,5),(0.1,2)])
    r0_opt, b_opt = res.x
    f_opt = trial_f(rho_grid, W, r0_opt, b_opt)
    E_opt = integrate_energy(f_opt, rho_grid)
    # Масса в МэВ: m = (E_dimless * hbar c / a) ??? 
    # Из B.2.1: масштаб a = sqrt(alpha) * R_e, энергия в МэВ = E_dimless * (hbar c / a)
    hbar_c_MeV_fm = 197.32698
    mass_MeV = E_opt * hbar_c_MeV_fm / a
    return mass_MeV, E_opt, r0_opt, b_opt

# Расчёт для W=1 (электрон), W=6 (мюон), W=15 (тау)
print("Вариационная оценка масс по закону M ∝ W^3")
print("="*60)

m_e_calc, E1, r1, b1 = compute_mass(1)
print(f"W=1 (электрон): расч. масса = {m_e_calc:.3f} МэВ (эксп. 0.511 МэВ)")

# Для W=6 и W=15 используем масштабирование, но для точности лучше отдельно оптимизировать
m_mu_calc, E6, r6, b6 = compute_mass(6, r0_guess=3.0, b_guess=0.3)
print(f"W=6 (мюон)   : расч. масса = {m_mu_calc:.1f} МэВ (эксп. 105.66 МэВ)")

m_tau_calc, E15, r15, b15 = compute_mass(15, r0_guess=5.0, b_guess=0.2)
print(f"W=15 (тау)   : расч. масса = {m_tau_calc:.0f} МэВ (эксп. 1776.9 МэВ)")

print("\nОтношения:")
print(f"W=6 / W=1: расч. {m_mu_calc/m_e_calc:.1f}, эксп. 206.8")
print(f"W=15 / W=1: расч. {m_tau_calc/m_e_calc:.1f}, эксп. 3477")
print(f"W^3 отношения: 1 : 216 : 3375")
