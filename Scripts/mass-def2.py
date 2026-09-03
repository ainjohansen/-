import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
from scipy.stats import binned_statistic
from scipy.optimize import curve_fit
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

def refine_near_curve(curve_points, grid_pts, coarse_E, coarse_dV, 
                      refine_radius, refine_factor, A_core, B_tail, alpha):
    """
    Простое адаптивное сгущение: для точек, лежащих ближе refine_radius к кривой,
    пересчитывает плотность энергии на более мелкой сетке (refine_factor раз мельче)
    и заменяет значение в грубой сетке.
    Возвращает уточнённый массив плотности энергии.
    """
    print(f"  Адаптивное сгущение: радиус={refine_radius}, фактор={refine_factor}")
    tree = cKDTree(curve_points)
    # Индексы точек, близких к кривой
    dist, idx = tree.query(grid_pts)
    close_mask = dist < refine_radius
    close_indices = np.where(close_mask)[0]
    n_close = len(close_indices)
    if n_close == 0:
        print("  Нет точек для сгущения.")
        return coarse_E

    print(f"  Найдено {n_close} точек для уточнения ({100*n_close/len(grid_pts):.2f}%)")
    
    # Создаём мелкую сетку вокруг каждой близкой точки (упрощённо: локальный куб)
    # Для простоты: будем для каждой близкой точки брать мелкую сетку в её окрестности
    # и усреднять вклад. Это не идеально, но даст улучшение.
    refined_E = coarse_E.copy()
    # Определяем шаг мелкой сетки
    coarse_step = (grid_pts.max(axis=0) - grid_pts.min(axis=0)) / (int(round(len(grid_pts)**(1/3))) - 1)
    fine_step = coarse_step / refine_factor
    
    # Для каждой близкой точки (можно обрабатывать группы, но для простоты цикл)
    # Чтобы не замедлять сильно, ограничим количество обрабатываемых точек
    max_points = min(n_close, 10000)  # ограничение на время
    for i in range(max_points):
        pt = grid_pts[close_indices[i]]
        # Создаём мелкую сетку вокруг pt в кубе со стороной 2*refine_radius
        half_size = refine_radius
        n_steps = int(2 * half_size / fine_step) + 1
        if n_steps > 20:  # ограничим размер мелкой сетки
            n_steps = 20
            fine_step = 2 * half_size / (n_steps - 1)
        xs = np.linspace(pt[0] - half_size, pt[0] + half_size, n_steps)
        ys = np.linspace(pt[1] - half_size, pt[1] + half_size, n_steps)
        zs = np.linspace(pt[2] - half_size, pt[2] + half_size, n_steps)
        Xf, Yf, Zf = np.meshgrid(xs, ys, zs, indexing='ij')
        fine_pts = np.vstack((Xf.ravel(), Yf.ravel(), Zf.ravel())).T
        dV_fine = (2 * half_size / (n_steps - 1))**3
        # Вычисляем энергию на мелкой сетке
        E_fine, _ = compute_energy_field(curve_points, fine_pts, dV_fine, A_core, B_tail, alpha)
        # Усредняем плотность энергии в грубой ячейке
        refined_E[close_indices[i]] = np.mean(E_fine)
    
    return refined_E

def run_calculation(grid_size, bound=5.0, A_core=0.8168, B_tail=0.08542, alpha=0.00729735,
                    do_refine=False, refine_radius=0.2, refine_factor=3):
    """
    Запускает расчёт для протона и нейтрона на сетке grid_size.
    Если do_refine=True, применяет адаптивное сгущение к нейтрону.
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

    # Расчёт протона
    E_proton, Mass_p = compute_energy_field(proton_curve, grid_pts, dV, A_core, B_tail, alpha)

    # Расчёт нейтрона (возможно с адаптивным сгущением)
    if do_refine:
        E_neutron_coarse, Mass_n_coarse = compute_energy_field(neutron_curve, grid_pts, dV, A_core, B_tail, alpha)
        E_neutron = refine_near_curve(neutron_curve, grid_pts, E_neutron_coarse, dV,
                                      refine_radius, refine_factor, A_core, B_tail, alpha)
        Mass_n = np.sum(E_neutron) * dV
        print(f"  Эффект сгущения: масса нейтрона изменилась с {Mass_n_coarse:.4f} на {Mass_n:.4f}")
    else:
        E_neutron, Mass_n = compute_energy_field(neutron_curve, grid_pts, dV, A_core, B_tail, alpha)

    Delta_Mass = Mass_n - Mass_p
    Ratio = Delta_Mass / Mass_p
    Mass_defect_MeV = 938.272 * Ratio

    # Сохраняем профили для последнего расчёта (если нужно)
    # (здесь можно добавить сохранение, но для простоты опустим)

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

# Список разрешений сетки
grid_sizes = [100, 150, 200, 250, 300]
# Включить адаптивное сгущение для нейтрона? (для последних размеров может быть долго)
do_refine = False  # сначала без сгущения, потом можно включить для одного размера

results = []
for gs in grid_sizes:
    start_time = time.time()
    M_p, M_n, dM, dM_MeV = run_calculation(gs, do_refine=do_refine)
    elapsed = time.time() - start_time
    results.append([gs, M_p, M_n, dM, dM_MeV, elapsed])
    print(f"  Время расчёта: {elapsed:.1f} сек")

# Сохраняем в CSV
df = pd.DataFrame(results, columns=['grid_size', 'Mass_p', 'Mass_n', 'Delta_Mass', 'Delta_Mass_MeV', 'time_sec'])
df.to_csv('mass_convergence.csv', index=False)
print("\nРезультаты сохранены в mass_convergence.csv")

# =====================================================================
# 2. ГРАФИК СХОДИМОСТИ С АППРОКСИМАЦИЕЙ
# =====================================================================
plt.style.use('dark_background')
fig, ax = plt.subplots(figsize=(10, 6))

ax.plot(df['grid_size'], df['Delta_Mass_MeV'], 'o-', color='cyan', linewidth=2, markersize=8, label='Расчётное значение')
ax.axhline(y=1.293, color='red', linestyle='--', linewidth=1.5, label='Эксперимент (1.293 МэВ)')

# Аппроксимация функцией a + b/N
def fit_func(N, a, b):
    return a + b / N

popt, pcov = curve_fit(fit_func, df['grid_size'], df['Delta_Mass_MeV'], p0=[1.293, 10])
a_fit, b_fit = popt
a_err = np.sqrt(pcov[0,0])

N_extrap = np.linspace(min(grid_sizes), 500, 100)
ax.plot(N_extrap, fit_func(N_extrap, *popt), '--', color='gray', alpha=0.6,
        label=f'Экстраполяция: {a_fit:.3f} ± {a_err:.3f} МэВ')

ax.set_xlabel('Размер сетки (N³)', fontsize=12)
ax.set_ylabel('Дефект масс Δm (МэВ)', fontsize=12)
ax.set_title('Сходимость дефекта масс нейтрона с увеличением разрешения сетки', fontsize=14)
ax.grid(color='white', alpha=0.2)
ax.legend()

plt.tight_layout()
plt.savefig('mass_convergence.png', dpi=150)
plt.show()

print("\nЭкстраполированное значение дефекта масс (N→∞): {:.3f} ± {:.3f} МэВ".format(a_fit, a_err))
print("Расхождение с экспериментом: {:.3f} МэВ".format(abs(a_fit - 1.293)))

# =====================================================================
# 3. (ОПЦИОНАЛЬНО) ЗАПУСК С АДАПТИВНЫМ СГУЩЕНИЕМ ДЛЯ ОДНОГО РАЗМЕРА
# =====================================================================
# Раскомментировать для проверки эффекта сгущения
"""
print("\n" + "="*60)
print(" РАСЧЁТ С АДАПТИВНЫМ СГУЩЕНИЕМ (СЕТКА 300³)")
print("="*60)
M_p_ref, M_n_ref, dM_ref, dM_MeV_ref = run_calculation(300, do_refine=True, refine_radius=0.2, refine_factor=3)
print(f"Дефект масс с адаптивным сгущением: {dM_MeV_ref:.3f} МэВ")
"""

print("="*60)
print(" РАБОТА ЗАВЕРШЕНА")
print("="*60)
