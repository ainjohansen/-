import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import simpson
import os

alpha_native = 0.0072973525693

# Загружаем сохранённый BVP-профиль (предполагаем, что он есть)
profile_file = 'bvp_exact_profile.npz'
if not os.path.exists(profile_file):
    print("Файл с профилем не найден. Сначала запустите BVP-солвер.")
    exit()

data = np.load(profile_file)
r_bvp = data['r']
f_bvp = data['f']
fp_bvp = data['fp']

# Вычисляем эталонные интегралы
def compute_reference(r, f, fp):
    sin_f = np.sin(f)
    sin_f_r = np.where(r < 1e-8, -fp, sin_f / r)
    I2_dens = r**2 * fp**2 + 2 * sin_f**2
    I4_dens = sin_f**2 * (2 * fp**2 + sin_f_r**2)
    I0_dens = (1 - np.cos(f)) * r**2
    I2 = simpson(4 * np.pi * I2_dens, x=r)
    I4 = simpson(4 * np.pi * I4_dens, x=r)
    I0 = simpson(4 * np.pi * I0_dens, x=r)
    rho_top = (1/(4*np.pi)) * 2 * fp * sin_f**2 / (r**2 + 1e-12)
    Q = simpson(4 * np.pi * r**2 * rho_top, x=r)
    R_eff = simpson(4 * np.pi * r**3 * rho_top, x=r) / Q
    return I2, I4, I0, R_eff

I2_0, I4_0, I0_0, Reff_0 = compute_reference(r_bvp, f_bvp, fp_bvp)

print("=== ЭТАЛОННЫЕ КОНСТАНТЫ ===")
print(f"I2_0 = {I2_0:.6f}")
print(f"I4_0 = {I4_0:.6f}")
print(f"I0_0 = {I0_0:.6f}")
print(f"Reff_0 = {Reff_0:.6f}")

# Энергия как функция масштаба a и параметра среды alpha
def energy(a, alpha):
    I2 = I2_0 / a
    I4 = I4_0 * a
    I0 = I0_0 * a**3
    Ec = alpha / (Reff_0 * a)
    return I2 + I4 + 3 * alpha * I0 + Ec

# Строим яму для нашей alpha
a_vals = np.linspace(0.7, 1.3, 200)
E_native = [energy(a, alpha_native) for a in a_vals]

# Находим минимум
a_opt = a_vals[np.argmin(E_native)]
print(f"\nДля α = {alpha_native:.6f} оптимум a = {a_opt:.4f}")

# График: энергия vs масштаб
plt.figure(figsize=(8,5))
plt.plot(a_vals, E_native, 'b-', lw=2)
plt.axvline(a_opt, color='r', linestyle='--', label=f'Минимум a={a_opt:.3f}')
plt.xlabel('Масштаб a')
plt.ylabel('Энергия')
plt.title('Энергетическая яма электрона-хопфиона')
plt.legend()
plt.grid(alpha=0.3)
plt.savefig('energy_well_fast.png', dpi=150)
plt.show()
