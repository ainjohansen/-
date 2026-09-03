import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import simpson

alpha = 0.0072973525693

def f_ansatz(r, r0):
    """Фаза от pi до 0 с экспоненциальным хвостом."""
    x = r / r0
    return np.pi * np.exp(-x) * np.tanh(x) / (x + 1e-12)

def energy_from_ansatz(r0, r_max=20.0, n_points=500):
    r = np.linspace(1e-6, r_max, n_points)
    f = f_ansatz(r, r0)
    fp = np.gradient(f, r)
    sin_f = np.sin(f)
    sin_f_r = np.where(r < 1e-8, -fp, sin_f / r)

    # Плотности
    I2_dens = r**2 * fp**2 + 2 * sin_f**2
    I4_dens = sin_f**2 * (2 * fp**2 + sin_f_r**2)
    I0_dens = (1 - np.cos(f)) * r**2

    # Интегралы по r (сферическая симметрия, 4π r^2 dr)
    I2 = simpson(4 * np.pi * I2_dens, x=r)
    I4 = simpson(4 * np.pi * I4_dens, x=r)
    I0 = simpson(4 * np.pi * I0_dens, x=r)

    # Кулоновская энергия (приближение через эффективный радиус)
    # Эффективный радиус можно оценить как интеграл от r * плотность заряда
    rho_top = sin_f**2 * fp / (2 * np.pi**2)  # нормировка для Q_H=1
    R_eff = simpson(4 * np.pi * r**3 * rho_top, x=r)
    E_coulomb = alpha / R_eff if R_eff > 1e-12 else 0.0

    return I2 + I4 + 3 * alpha * I0 + E_coulomb

# Сканирование
r0_vals = np.linspace(0.5, 3.0, 30)
energies = [energy_from_ansatz(r0) for r0 in r0_vals]

plt.plot(r0_vals, energies, 'b-o')
plt.xlabel('Масштаб $r_0$')
plt.ylabel('Полная энергия')
plt.title('Энергетическая яма для анзаца с экспоненциальным хвостом')
plt.grid(alpha=0.3)
plt.show()
