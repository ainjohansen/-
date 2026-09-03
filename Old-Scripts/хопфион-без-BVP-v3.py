import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import simpson
from scipy.optimize import minimize_scalar

alpha = 0.0072973525693

# ------------------------------------------------------------
# Аналитический анзац для профиля фазы f(r)
# ------------------------------------------------------------
def f_ansatz(r, r0):
    x = r / r0
    # Избегаем деления на ноль
    safe_x = np.where(x < 1e-8, 1e-8, x)
    return np.pi * np.exp(-x) * np.tanh(x) / safe_x

def compute_integrals(r0, r_max=30.0, n=1000):
    r = np.linspace(1e-6, r_max, n)
    f = f_ansatz(r, r0)
    fp = np.gradient(f, r)
    sin_f = np.sin(f)
    sin_f_r = np.where(r < 1e-8, -fp, sin_f / r)

    I2_dens = r**2 * fp**2 + 2 * sin_f**2
    I4_dens = sin_f**2 * (2 * fp**2 + sin_f_r**2)
    I0_dens = (1 - np.cos(f)) * r**2

    # Интегралы по r (сферическая симметрия условная, т.к. хопфион аксиален)
    # Для корректного сравнения с BVP используем те же веса: 4π r^2 dr
    I2 = simpson(4 * np.pi * I2_dens, x=r)
    I4 = simpson(4 * np.pi * I4_dens, x=r)
    I0 = simpson(4 * np.pi * I0_dens, x=r)
    return I2, I4, I0

# ------------------------------------------------------------
# Находим константы A2, A4, A0 (предполагая скейлинг I2 ~ 1/r0, I4 ~ r0, I0 ~ r0^3)
# Для этого вычислим интегралы при r0=1 и восстановим коэффициенты.
# ------------------------------------------------------------
r0_ref = 1.0
I2_ref, I4_ref, I0_ref = compute_integrals(r0_ref)

# Скейлинговые коэффициенты
A2 = I2_ref * r0_ref      # I2 = A2 / r0
A4 = I4_ref / r0_ref      # I4 = A4 * r0
A0 = I0_ref / r0_ref**3   # I0 = A0 * r0^3

print("=== КОЭФФИЦИЕНТЫ АНЗАЦА ===")
print(f"A2 = {A2:.6f}")
print(f"A4 = {A4:.6f}")
print(f"A0 = {A0:.6f}")

# ------------------------------------------------------------
# Полная энергия как функция r0 (включая кулоновский член)
# Кулоновскую энергию аппроксимируем α / r0 (коэффициент ~1)
# ------------------------------------------------------------
def total_energy(r0):
    I2 = A2 / r0
    I4 = A4 * r0
    I0 = A0 * r0**3
    E_coulomb = alpha / r0   # приближение
    return I2 + I4 + 3 * alpha * I0 + E_coulomb

# ------------------------------------------------------------
# Поиск минимума
# ------------------------------------------------------------
res = minimize_scalar(total_energy, bounds=(0.1, 5.0), method='bounded')
r0_opt = res.x
E_min = res.fun

print("\n=== РЕЗУЛЬТАТЫ ОПТИМИЗАЦИИ ===")
print(f"Оптимальный масштаб r0 = {r0_opt:.6f}")
print(f"Минимальная энергия E  = {E_min:.6f}")
print(f"I2(r0_opt) = {A2/r0_opt:.6f}")
print(f"I4(r0_opt) = {A4*r0_opt:.6f}")
print(f"I0(r0_opt) = {A0*r0_opt**3:.6f}")
print(f"Кулоновский вклад = {alpha/r0_opt:.6f}")
print(f"Баланс: (I4 - I2) = {A4*r0_opt - A2/r0_opt:.6f}, 3αI0 = {3*alpha*A0*r0_opt**3:.6f}")

# ------------------------------------------------------------
# График ямы
# ------------------------------------------------------------
r0_vals = np.linspace(0.3, 3.0, 100)
E_vals = [total_energy(r) for r in r0_vals]

plt.figure(figsize=(8,5))
plt.plot(r0_vals, E_vals, 'b-', lw=2)
plt.axvline(r0_opt, color='r', linestyle='--', label=f'Минимум при r0 = {r0_opt:.3f}')
plt.xlabel('Масштаб $r_0$')
plt.ylabel('Полная энергия')
plt.title('Энергетическая яма электрона-хопфиона (аналитический анзац)')
plt.legend()
plt.grid(alpha=0.3)
plt.savefig('hopfion_well_analytic.png', dpi=150)
plt.show()
