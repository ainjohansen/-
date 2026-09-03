import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.integrate import simpson
import time

# ------------------------------------------------------------
# Параметры вселенной
# ------------------------------------------------------------
alpha = 0.0072973525693   # постоянная тонкой структуры

# ------------------------------------------------------------
# Аналитическое поле хопфиона (Q_H = 1)
# ------------------------------------------------------------
def hopfion_field(rho, z, phi, r0=1.0):
    """Векторное поле n = (nx, ny, nz) в цилиндрических координатах."""
    denom = rho**2 + z**2 + r0**2
    nx = (2 * rho * z * np.cos(phi) - 2 * rho * r0 * np.sin(phi)) / denom
    ny = (2 * rho * z * np.sin(phi) + 2 * rho * r0 * np.cos(phi)) / denom
    nz = (rho**2 - z**2 - r0**2) / denom
    return nx, ny, nz

# ------------------------------------------------------------
# Аналитические плотности энергии (векторизованные)
# ------------------------------------------------------------
def energy_density_analytic(rho, z, r0=1.0):
    """
    Возвращает полную плотность энергии: e2 + e4 + e0.
    Все производные вычислены аналитически для отображения Хопфа.
    """
    denom = rho**2 + z**2 + r0**2
    denom2 = denom**2
    denom3 = denom**3

    # Для избежания деления на ноль в rho=0 используем малое смещение
    rho_safe = np.where(rho < 1e-12, 1e-12, rho)

    # --- I2: (∂_i n)^2 ---
    # Аналитически: (∂_i n)^2 = 8 (r0^2 + z^2 + rho^2)^{-2}
    # Это точное выражение для данного отображения Хопфа.
    grad2 = 8.0 / denom2

    # --- I4: (n · [∂_x n × ∂_y n])^2 + ... (топологическая плотность) ---
    # Для хопфиона Q_H=1: n·(∂_ρ n × ∂_φ n) = (4 r0)/(rho * denom^2)
    # Плотность I4 есть квадрат этой величины.
    top_density = (4.0 * r0) / (rho_safe * denom2)
    e4 = top_density**2

    # --- I0: массовый член 3α (1 - n_z) ---
    nz = (rho**2 - z**2 - r0**2) / denom
    e0 = 3.0 * alpha * (1.0 - nz)

    return grad2 + e4 + e0

# ------------------------------------------------------------
# Быстрое интегрирование по цилиндрической сетке
# ------------------------------------------------------------
def total_energy_fast(r0, rho_max=15.0, z_max=15.0, n_rho=300, n_z=300):
    """
    Вычисляет полную энергию интегрированием по цилиндрическим координатам.
    Возвращает энергию и сетки для визуализации.
    """
    rho = np.linspace(0, rho_max, n_rho)
    z = np.linspace(-z_max, z_max, n_z)
    RHO, Z = np.meshgrid(rho, z, indexing='ij')

    # Вычисляем плотность энергии (уже умноженную на 2π rho)
    # Формула: E = ∫ 2π rho d rho dz * energy_density
    dens = energy_density_analytic(RHO, Z, r0)
    integrand = 2.0 * np.pi * RHO * dens

    # Интегрируем методом Симпсона
    integral_rho = simpson(integrand, x=rho, axis=0)
    total_E = simpson(integral_rho, x=z)
    E_coulomb = 0.6 * alpha / r0
    return total_E + E_coulomb, RHO, Z, dens
    
# ------------------------------------------------------------
# Сканирование по масштабу r0 и построение ямы
# ------------------------------------------------------------
print("Сканирование масштаба ядра r0...")
r0_vals = np.linspace(0.3, 2.5, 20)
energies = []
start_time = time.time()
for r0 in r0_vals:
    E, _, _, _ = total_energy_fast(r0, rho_max=12.0, z_max=12.0, n_rho=200, n_z=200)
    energies.append(E)
    print(f"r0 = {r0:.2f} → E = {E:.6f}")
print(f"Сканирование завершено за {time.time()-start_time:.2f} сек")

# ------------------------------------------------------------
# Визуализация энергетической ямы
# ------------------------------------------------------------
plt.figure(figsize=(10,5))
plt.subplot(121)
plt.plot(r0_vals, energies, 'b-o', markersize=4)
plt.xlabel('Масштаб ядра $r_0$')
plt.ylabel('Полная энергия')
plt.title('Энергетическая яма хопфиона')
plt.grid(alpha=0.3)

# Находим минимум
min_idx = np.argmin(energies)
r0_min = r0_vals[min_idx]
plt.axvline(r0_min, color='red', linestyle='--', label=f'Минимум при $r_0$ = {r0_min:.3f}')
plt.legend()

# ------------------------------------------------------------
# Энергетический профиль (2D-сечение)
# ------------------------------------------------------------
print("\nПостроение 2D-профиля плотности энергии...")
E_opt, RHO, Z, DENS = total_energy_fast(r0_min, rho_max=5.0, z_max=5.0, n_rho=400, n_z=400)

plt.subplot(122)
# Логарифмический масштаб для контраста
plt.contourf(RHO, Z, DENS, levels=50, cmap='inferno')
plt.colorbar(label='Плотность энергии')
plt.xlabel(r'$\rho$')
plt.ylabel('z')
plt.title(f'Сечение энергетического профиля ($r_0$ = {r0_min:.3f})')
plt.axis('equal')

plt.tight_layout()
plt.savefig('hopfion_energy_well.png', dpi=150)
plt.show()

# ------------------------------------------------------------
# 3D-визуализация силовых линий (опционально)
# ------------------------------------------------------------
print("\nГенерация 3D-визуализации линий поля...")
fig = plt.figure(figsize=(8,8))
ax = fig.add_subplot(111, projection='3d')

# Параметры тора, по которому будем рисовать линии
R_torus = 1.5
a_torus = 0.8
n_lines = 16
theta_vals = np.linspace(0, 2*np.pi, n_lines, endpoint=False)

# Цветовая карта для линий
colors = plt.cm.hsv(np.linspace(0, 1, n_lines))

for i, theta in enumerate(theta_vals):
    s = np.linspace(0, 2*np.pi, 200)
    x = (R_torus + a_torus * np.cos(s)) * np.cos(theta)
    y = (R_torus + a_torus * np.cos(s)) * np.sin(theta)
    z = a_torus * np.sin(s)

    # Вычисляем поле в этих точках
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    nx, ny, nz = hopfion_field(rho, z, phi, r0=r0_min)

    # Рисуем линию с цветом, зависящим от направления поля (можно просто одноцветные)
    ax.plot(x, y, z, color=colors[i], lw=1.5, alpha=0.8)

# Настройки вида
ax.set_xlim([-2.8, 2.8])
ax.set_ylim([-2.8, 2.8])
ax.set_zlim([-2.8, 2.8])
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('Силовые линии хопфиона')
plt.savefig('hopfion_3d_lines.png', dpi=150)
plt.show()

print("\nГотово! Проверьте сохранённые изображения.")
