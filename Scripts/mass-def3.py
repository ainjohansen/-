import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
from scipy.optimize import curve_fit
import time
import pandas as pd
import os

# =====================================================================
# 0. ПАРАМЕТРЫ И ФУНКЦИИ
# =====================================================================

def compute_energy_field(curve_points, grid_pts, dV, A_core, B_tail, alpha):
    """Вычисляет энергию поля для заданной кривой и сетки."""
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

def generate_curves(t, R_torus=2.0, a_torus=0.8, twist_amp=0.0867, twist_freq=15):
    """Генерирует кривые протона и нейтрона."""
    xp = (R_torus + a_torus * np.cos(3 * t)) * np.cos(2 * t)
    yp = (R_torus + a_torus * np.cos(3 * t)) * np.sin(2 * t)
    zp = a_torus * np.sin(3 * t)
    proton_curve = np.vstack([xp, yp, zp]).T

    defect_center = 0.0
    dt_ang = np.pi - np.abs(np.pi - np.abs(t - defect_center))
    defect = np.exp(-(dt_ang / 0.25)**2)
    xn = xp + twist_amp * defect * np.cos(twist_freq * t)
    yn = yp + twist_amp * defect * np.sin(twist_freq * t)
    zn = zp + twist_amp * defect * np.cos(twist_freq * t + np.pi/2)
    neutron_curve = np.vstack([xn, yn, zn]).T

    return proton_curve, neutron_curve

def adaptive_refinement(curve, grid_pts, dV, A_core, B_tail, alpha,
                        refine_factor=3, radius_refine=0.3, grid_size=None):
    """
    Уточнение интеграла вблизи кривой.
    Возвращает поправку к массе (разница между интегралом на мелкой и грубой сетке).
    """
    print("  Адаптивное сгущение вблизи кривой...")
    tree_curve = cKDTree(curve)
    dist, _ = tree_curve.query(grid_pts, workers=-1)
    refine_mask = dist < radius_refine
    if not np.any(refine_mask):
        print("  Нет точек в области сгущения!")
        return 0.0, 0.0, 0.0

    coarse_pts = grid_pts[refine_mask]

    # Bounding box
    min_c = coarse_pts.min(axis=0) - radius_refine
    max_c = coarse_pts.max(axis=0) + radius_refine

    # Шаг грубой сетки (берём по x)
    unique_x = np.unique(grid_pts[:,0])
    dx_coarse = unique_x[1] - unique_x[0] if len(unique_x) > 1 else 1.0

    # Число точек мелкой сетки по каждому измерению
    n_refine = int(refine_factor * ( (max_c[0] - min_c[0]) / dx_coarse )) + 1
    if n_refine > 200:
        n_refine = 200
        print(f"  Ограничиваем количество точек по оси до {n_refine}")

    x_fine = np.linspace(min_c[0], max_c[0], n_refine)
    y_fine = np.linspace(min_c[1], max_c[1], n_refine)
    z_fine = np.linspace(min_c[2], max_c[2], n_refine)
    Xf, Yf, Zf = np.meshgrid(x_fine, y_fine, z_fine, indexing='ij')
    fine_pts = np.vstack((Xf.ravel(), Yf.ravel(), Zf.ravel())).T
    dV_fine = (x_fine[1] - x_fine[0])**3

    # Оставляем только точки в радиусе radius_refine от кривой
    dist_fine, _ = tree_curve.query(fine_pts, workers=-1)
    fine_mask = dist_fine < radius_refine
    fine_pts = fine_pts[fine_mask]
    if len(fine_pts) == 0:
        print("  Нет мелких точек в области сгущения!")
        return 0.0, 0.0, 0.0

    def get_density(points):
        d, _ = cKDTree(curve).query(points, workers=-1)
        d = np.maximum(d, 1e-12)
        u = (A_core / d) * np.exp(-B_tail * d)
        f = 2 * np.arctan(u)
        df_dd = (2 * u / (1 + u**2)) * (-1/d - B_tail)
        sin_f = np.sin(f)
        sin_f_d = np.where(d < 1e-8, -df_dd, sin_f / d)
        dens_I2 = d**2 * df_dd**2 + 2 * sin_f**2
        dens_I4 = sin_f**2 * (2 * df_dd**2 + sin_f_d**2)
        dens_I0 = (1 - np.cos(f)) * d**2
        return dens_I4 + dens_I2 + 3 * alpha * dens_I0

    fine_density = get_density(fine_pts)
    fine_energy = np.sum(fine_density) * dV_fine

    coarse_density = get_density(coarse_pts)
    coarse_energy = np.sum(coarse_density) * dV

    correction = fine_energy - coarse_energy
    print(f"  Поправка от сгущения: {correction:.6f} (безразм.)")
    return correction, fine_energy, coarse_energy

def run_calculation(grid_size, bound=5.0, A_core=0.8168, B_tail=0.08542, alpha=0.00729735,
                    do_refine=False, refine_factor=3, radius_refine=0.3):
    """
    Запускает расчёт для протона и нейтрона на сетке grid_size.
    Возвращает: безразмерную массу протона, нейтрона, дефект масс (безразм.), дефект в МэВ.
    Если do_refine=True, применяет адаптивное сгущение для нейтрона.
    """
    print(f"\n--- Расчёт на сетке {grid_size}^3 ---")
    x_g = np.linspace(-bound, bound, grid_size)
    y_g = np.linspace(-bound, bound, grid_size)
    z_g = np.linspace(-bound, bound, grid_size)
    X, Y, Z = np.meshgrid(x_g, y_g, z_g, indexing='ij')
    grid_pts = np.vstack((X.ravel(), Y.ravel(), Z.ravel())).T
    dV = (2 * bound / grid_size)**3

    t = np.linspace(0, 2 * np.pi, 5000)
    proton_curve, neutron_curve = generate_curves(t)

    # Протон
    E_proton, Mass_p = compute_energy_field(proton_curve, grid_pts, dV, A_core, B_tail, alpha)

    # Нейтрон (базовый расчёт)
    E_neutron_coarse, Mass_n_coarse = compute_energy_field(neutron_curve, grid_pts, dV, A_core, B_tail, alpha)

    if do_refine:
        correction, fine_energy, coarse_energy = adaptive_refinement(
            neutron_curve, grid_pts, dV, A_core, B_tail, alpha,
            refine_factor, radius_refine, grid_size)
        Mass_n = Mass_n_coarse + correction
        print(f"  Грубая масса нейтрона: {Mass_n_coarse:.4f}")
        print(f"  Уточнённая масса нейтрона: {Mass_n:.4f}")
    else:
        Mass_n = Mass_n_coarse

    Delta_Mass = Mass_n - Mass_p
    Ratio = Delta_Mass / Mass_p
    Mass_defect_MeV = 938.272 * Ratio

    print(f"  Безразмерная масса протона: {Mass_p:.4f}")
    print(f"  Безразмерная масса нейтрона: {Mass_n:.4f}")
    print(f"  Дефект масс (безразм.): {Delta_Mass:.4f}")
    print(f"  Дефект масс (МэВ): {Mass_defect_MeV:.3f}")

    return Mass_p, Mass_n, Delta_Mass, Mass_defect_MeV

# =====================================================================
# 1. ЗАПУСК РАСЧЁТОВ ДЛЯ РАЗНЫХ СЕТОК (если CSV нет)
# =====================================================================

print("="*60)
print(" ЧЕСТНЫЙ 3D-ИНТЕГРАЛ: СХОДИМОСТЬ ПО СЕТКЕ")
print("="*60)

csv_file = 'mass_convergence.csv'
if not os.path.exists(csv_file):
    grid_sizes = [100, 150, 200, 250, 300]
    results = []
    for gs in grid_sizes:
        start_time = time.time()
        M_p, M_n, dM, dM_MeV = run_calculation(gs, do_refine=False)
        elapsed = time.time() - start_time
        results.append([gs, M_p, M_n, dM, dM_MeV, elapsed])
        print(f"  Время расчёта: {elapsed:.1f} сек")

    df = pd.DataFrame(results, columns=['grid_size', 'Mass_p', 'Mass_n', 'Delta_Mass', 'Delta_Mass_MeV', 'time_sec'])
    df.to_csv(csv_file, index=False)
    print("\nРезультаты сохранены в", csv_file)
else:
    df = pd.read_csv(csv_file)
    print("Загружены существующие данные из", csv_file)
    print(df.to_string(index=False))

# =====================================================================
# 2. АДАПТИВНОЕ СГУЩЕНИЕ ДЛЯ САМОГО ВЫСОКОГО РАЗРЕШЕНИЯ
# =====================================================================
max_grid = df['grid_size'].max()
print("\n" + "="*60)
print(f" ПРОВОДИМ АДАПТИВНОЕ СГУЩЕНИЕ ДЛЯ СЕТКИ {max_grid}^3")
print("="*60)

start_ref = time.time()
M_p_ref, M_n_ref, dM_ref, dM_MeV_ref = run_calculation(max_grid, do_refine=True,
                                                        refine_factor=3, radius_refine=0.3)
elapsed_ref = time.time() - start_ref
print(f"  Время расчёта с уточнением: {elapsed_ref:.1f} сек")

# =====================================================================
# 3. ЭКСТРАПОЛЯЦИЯ И ГРАФИК СХОДИМОСТИ
# =====================================================================
def fit_func(N, a, b):
    return a + b / N

N_vals = df['grid_size'].values
dM_vals = df['Delta_Mass_MeV'].values

# Аппроксимация для N>=150 (лучшая сходимость)
mask = N_vals >= 150
popt, pcov = curve_fit(fit_func, N_vals[mask], dM_vals[mask], p0=[1.293, 10])
a_fit, b_fit = popt
print("\nАппроксимация Δm(N) = a + b/N")
print(f"  a = {a_fit:.5f} МэВ (экстраполированное значение)")
print(f"  b = {b_fit:.5f} МэВ")

# Построение графика
plt.style.use('dark_background')
fig, ax = plt.subplots(figsize=(10, 6))

ax.plot(N_vals, dM_vals, 'o-', color='cyan', linewidth=2, markersize=8, label='Расчётное значение (без сгущения)')
ax.plot(max_grid, dM_MeV_ref, 's', color='yellow', markersize=10, label=f'С адаптивным сгущением ({max_grid}³)')

ax.axhline(y=1.293, color='red', linestyle='--', linewidth=1.5, label='Эксперимент (1.293 МэВ)')

N_fine = np.linspace(100, 500, 200)
ax.plot(N_fine, fit_func(N_fine, a_fit, b_fit), '--', color='gray', alpha=0.8,
        label=f'Экстраполяция: {a_fit:.3f} МэВ')

ax.set_xlabel('Размер сетки (N³)', fontsize=12)
ax.set_ylabel('Дефект масс Δm (МэВ)', fontsize=12)
ax.set_title('Сходимость дефекта масс нейтрона с увеличением разрешения сетки', fontsize=14)
ax.grid(color='white', alpha=0.2)
ax.legend()

plt.tight_layout()
plt.savefig('mass_convergence.png', dpi=150)
plt.show()

# =====================================================================
# 4. ИТОГОВЫЙ ВЫВОД
# =====================================================================
print("\n" + "="*60)
print(" ИТОГОВЫЕ РЕЗУЛЬТАТЫ")
print("="*60)
print(f"Экстраполированное значение дефекта масс (без сгущения): {a_fit:.3f} МэВ")
print(f"Экспериментальное значение: 1.293 МэВ")
print(f"Расхождение: {abs(a_fit - 1.293):.3f} МэВ ({abs(a_fit/1.293 - 1)*100:.2f}%)")
print(f"\nПосле адаптивного сгущения на сетке {max_grid}³:")
print(f"  Дефект масс: {dM_MeV_ref:.3f} МэВ")
print(f"  Расхождение с экспериментом: {abs(dM_MeV_ref - 1.293):.3f} МэВ ({abs(dM_MeV_ref/1.293 - 1)*100:.2f}%)")
print("\nГрафик сохранён как mass_convergence.png")
