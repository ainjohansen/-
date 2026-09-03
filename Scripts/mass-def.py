import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
from scipy.stats import binned_statistic
import time
import os
import pandas as pd

# =====================================================================
# 0. ПАРАМЕТРЫ И ФУНКЦИИ
# =====================================================================

def compute_energy_field(curve_points, grid_pts, dV, A_core, B_tail, alpha):
    """
    Вычисляет энергию поля для заданной кривой (узла) и сетки.
    Возвращает массив плотности энергии и полную массу.
    """
    tree = cKDTree(curve_points)
    d, _ = tree.query(grid_pts, workers=-1)
    d = np.maximum(d, 1e-12)

    u = (A_core / d) * np.exp(-B_tail * d)
    f = 2 * np.arctan(u)
    df_dd = (2 * u / (1 + u**2)) * (-1/d - B_tail)

    sin_f = np.sin(f)
    sin_f_d = np.where(d < 1e-8, -df_dd, sin_f / d)

    dens_I2 = d**2 * df_dd**2 + 2 * sin_f**2
    dens_I4 = sin_f**2 * (2 * df_dd**2 + sin_f_d**2)
    dens_I0 = (1 - np.cos(f)) * d**2

    E_total = dens_I4 + dens_I2 + 3 * alpha * dens_I0
    Mass = np.sum(E_total) * dV
    return E_total, Mass

def run_calculation(grid_size, bound=5.0, A_core=0.8168, B_tail=0.08542, alpha=0.00729735):
    """
    Запускает расчёт для протона и нейтрона на сетке grid_size.
    Возвращает: безразмерную массу протона, нейтрона, дефект масс (безразм.), дефект в МэВ.
    """
    print(f"\n--- Расчёт на сетке {grid_size}^3 ---")
    # Генерация сетки
    x_g = np.linspace(-bound, bound, grid_size)
    y_g = np.linspace(-bound, bound, grid_size)
    z_g = np.linspace(-bound, bound, grid_size)
    X, Y, Z = np.meshgrid(x_g, y_g, z_g, indexing='ij')
    grid_pts = np.vstack((X.ravel(), Y.ravel(), Z.ravel())).T
    dV = (2 * bound / grid_size)**3

    # Построение кривых протона и нейтрона
    t = np.linspace(0, 2 * np.pi, 5000)
    R_torus, a_torus = 2.0, 0.8

    # Протон (гладкий трилистник)
    xp = (R_torus + a_torus * np.cos(3 * t)) * np.cos(2 * t)
    yp = (R_torus + a_torus * np.cos(3 * t)) * np.sin(2 * t)
    zp = a_torus * np.sin(3 * t)
    proton_curve = np.vstack([xp, yp, zp]).T

    # Нейтрон (трилистник + дефект Хопфа)
    defect_center = 0.0
    dt_ang = np.pi - np.abs(np.pi - np.abs(t - defect_center))
    defect = np.exp(-(dt_ang / 0.25)**2)
    twist_freq = 15
    twist_amp = 0.0867
    xn = xp + twist_amp * defect * np.cos(twist_freq * t)
    yn = yp + twist_amp * defect * np.sin(twist_freq * t)
    zn = zp + twist_amp * defect * np.cos(twist_freq * t + np.pi/2)
    neutron_curve = np.vstack([xn, yn, zn]).T

    # Расчёт энергий
    E_proton, Mass_p = compute_energy_field(proton_curve, grid_pts, dV, A_core, B_tail, alpha)
    E_neutron, Mass_n = compute_energy_field(neutron_curve, grid_pts, dV, A_core, B_tail, alpha)

    Delta_Mass = Mass_n - Mass_p
    Ratio = Delta_Mass / Mass_p
    Mass_defect_MeV = 938.272 * Ratio

    # Сохраняем также профили для последнего расчёта (будет использовано при построении)
    if grid_size == max_grid:
        # Для дальнейшей визуализации радиального профиля
        R_cm = np.sqrt(X**2 + Y**2 + Z**2).ravel()
        bins = np.linspace(0, bound, 120)
        hist_p, bin_edges, _ = binned_statistic(R_cm, E_proton, statistic='sum', bins=bins)
        hist_n, bin_edges, _ = binned_statistic(R_cm, E_neutron, statistic='sum', bins=bins)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        shell_volume = 4 * np.pi * bin_centers**2 * (bins[1] - bins[0])
        dens_p = (hist_p * dV) / shell_volume
        dens_n = (hist_n * dV) / shell_volume
        # Сохраняем глобально для отображения
        global last_dens_p, last_dens_n, last_bin_centers
        last_dens_p, last_dens_n, last_bin_centers = dens_p, dens_n, bin_centers

    print(f"  Безразмерная масса протона: {Mass_p:.4f}")
    print(f"  Безразмерная масса нейтрона: {Mass_n:.4f}")
    print(f"  Дефект масс (безразм.): {Delta_Mass:.4f}")
    print(f"  Дефект масс (МэВ): {Mass_defect_MeV:.3f}")

    return Mass_p, Mass_n, Delta_Mass, Mass_defect_MeV

# =====================================================================
# 1. ЗАПУСК РАСЧЁТОВ ДЛЯ РАЗНЫХ СЕТОК
# =====================================================================

print("="*60)
print(" ЧЕСТНЫЙ 3D-ИНТЕГРАЛ: СХОДИМОСТЬ ПО СЕТКЕ")
print("="*60)

# Список разрешений сетки (можете добавить свои)
grid_sizes = [100, 150, 200, 250, 300]
max_grid = max(grid_sizes)

results = []
for gs in grid_sizes:
    start_time = time.time()
    M_p, M_n, dM, dM_MeV = run_calculation(gs)
    elapsed = time.time() - start_time
    results.append([gs, M_p, M_n, dM, dM_MeV, elapsed])
    print(f"  Время расчёта: {elapsed:.1f} сек")

# Сохраняем в CSV
df = pd.DataFrame(results, columns=['grid_size', 'Mass_p', 'Mass_n', 'Delta_Mass', 'Delta_Mass_MeV', 'time_sec'])
df.to_csv('mass_convergence.csv', index=False)
print("\nРезультаты сохранены в mass_convergence.csv")

# =====================================================================
# 2. ГРАФИК СХОДИМОСТИ
# =====================================================================
plt.style.use('dark_background')
fig, ax = plt.subplots(figsize=(10, 6))

ax.plot(df['grid_size'], df['Delta_Mass_MeV'], 'o-', color='cyan', linewidth=2, markersize=8, label='Расчётное значение')
ax.axhline(y=1.293, color='red', linestyle='--', linewidth=1.5, label='Эксперимент (1.293 МэВ)')
ax.set_xlabel('Размер сетки (N³)', fontsize=12)
ax.set_ylabel('Дефект масс Δm (МэВ)', fontsize=12)
ax.set_title('Сходимость дефекта масс нейтрона с увеличением разрешения сетки', fontsize=14)
ax.grid(color='white', alpha=0.2)
ax.legend()

# Простая экстраполяция (обратная зависимость от N)
from scipy.optimize import curve_fit
def fit_func(N, a, b):
    return a + b / N
popt, _ = curve_fit(fit_func, df['grid_size'], df['Delta_Mass_MeV'], p0=[1.293, 10])
N_extrap = np.linspace(100, 500, 100)
ax.plot(N_extrap, fit_func(N_extrap, *popt), '--', color='gray', alpha=0.6, label=f'Экстраполяция: {popt[0]:.3f} МэВ')
ax.legend()

plt.tight_layout()
plt.savefig('mass_convergence.png', dpi=150)
plt.show()

# =====================================================================
# 3. РАДИАЛЬНЫЙ ПРОФИЛЬ ДЛЯ САМОГО ДЕТАЛЬНОГО РАСЧЁТА
# =====================================================================
if 'last_dens_p' in globals():
    fig2, ax2 = plt.subplots(figsize=(12, 7))
    ax2.plot(last_bin_centers, last_dens_p, color='cyan', lw=2.5, label='ПРОТОН ($p^+$)', alpha=0.9)
    ax2.plot(last_bin_centers, last_dens_n, color='coral', lw=2.5, linestyle='--', label='НЕЙТРОН ($n^0$)', alpha=0.9)
    ax2.fill_between(last_bin_centers, last_dens_p, last_dens_n, where=(last_dens_n > last_dens_p),
                     color='yellow', alpha=0.5, label='Дефект масс (вплетённый хопфион)')
    ax2.set_title(f'Радиальный профиль плотности энергии (сетка {max_grid}³)')
    ax2.set_xlabel('Радиальное расстояние от центра (безразм.)')
    ax2.set_ylabel('Усреднённая радиальная плотность ρ_M(r)')
    ax2.set_xlim(0, 4)
    ax2.grid(color='white', alpha=0.1)
    ax2.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig('radial_profile.png', dpi=150)
    plt.show()
else:
    print("Радиальные профили не сохранены (запустите с max_grid из списка).")

print("="*60)
print(" РАБОТА ЗАВЕРШЕНА")
print("="*60)
