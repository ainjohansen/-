import numpy as np
from scipy.spatial import cKDTree
import time
import sys
import os
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

print("="*70)
print(" ТОПОЛОГИЧЕСКАЯ МОДЕЛЬ: ИНТЕГРАЛ НЕПРЕРЫВНОСТИ С ВИЗУАЛИЗАЦИЕЙ")
print("="*70)

# =====================================================================
# 1. ВВОД ПАРАМЕТРОВ СЕТКИ И ПРОВЕРКА КЭША
# =====================================================================
grid_input = input("Введите размер сетки N (N^3 квантов, по умолчанию 200): ")
grid_size = int(grid_input) if grid_input.strip().isdigit() else 200

bound = 5.0
dV = (2 * bound / grid_size)**3
cache_filename = f"cache_knot_grid_{grid_size}.npz"

print(f"\nРазрешение сетки : {grid_size}x{grid_size}x{grid_size}")
print(f"Квантов объема   : {grid_size**3:,}")
print(f"Шаг сетки        : {(2*bound/grid_size):.5f} у.е.")

# Фундаментальные константы
A_core = 0.8168      # Константа топологической жесткости
alpha = 0.00729735   # Постоянная тонкой структуры
B_tail = np.sqrt(alpha) # Константа упругости вакуума

# =====================================================================
# 2. ФОРМИРОВАНИЕ ТОПОЛОГИЧЕСКИХ КАРКАСОВ (Быстро, делаем всегда)
# =====================================================================
t = np.linspace(0, 2 * np.pi, 15000) 
R_torus, a_torus = 2.0, 0.8

# ПРОТОН
xp = (R_torus + a_torus * np.cos(3 * t)) * np.cos(2 * t)
yp = (R_torus + a_torus * np.cos(3 * t)) * np.sin(2 * t)
zp = a_torus * np.sin(3 * t)
proton_curve = np.vstack([xp, yp, zp]).T

# НЕЙТРОН
defect_center = 0.0 
dt_ang = np.pi - np.abs(np.pi - np.abs(t - defect_center))
defect = np.exp(-(dt_ang / 0.25)**2)

twist_freq = 15
twist_amp = 0.08636 

xn = xp + twist_amp * defect * np.cos(twist_freq * t)
yn = yp + twist_amp * defect * np.sin(twist_freq * t)
zn = zp + twist_amp * defect * np.cos(twist_freq * t + np.pi/2)
neutron_curve = np.vstack([xn, yn, zn]).T

# =====================================================================
# 3. ТОМОГРАФИЧЕСКОЕ ИНТЕГРИРОВАНИЕ ИЛИ ЗАГРУЗКА ИЗ КЭША
# =====================================================================
start_time = time.time()

if os.path.exists(cache_filename):
    print(f"\n[!] Найден файл кэша '{cache_filename}'. Загрузка результатов...")
    data = np.load(cache_filename)
    Mass_p_total = float(data['Mass_p_total'])
    Mass_n_total = float(data['Mass_n_total'])
    E_p_slice = data['E_p_slice']
    E_n_slice = data['E_n_slice']
    exec_time = time.time() - start_time
else:
    print("\nПостроение пространственных KD-деревьев...")
    tree_p = cKDTree(proton_curve)
    tree_n = cKDTree(neutron_curve)

    Mass_p_total = 0.0
    Mass_n_total = 0.0

    x_g = np.linspace(-bound, bound, grid_size)
    y_g = np.linspace(-bound, bound, grid_size)
    z_g = np.linspace(-bound, bound, grid_size)
    X, Y = np.meshgrid(x_g, y_g, indexing='ij')
    X_flat, Y_flat = X.ravel(), Y.ravel()

    mid_z_index = grid_size // 2 # Индекс центрального среза для сохранения
    E_p_slice = None
    E_n_slice = None

    print(f"Запуск томографического сканирования ({grid_size} срезов)...")

    for i, z in enumerate(z_g):
        Z_flat = np.full_like(X_flat, z)
        slice_pts = np.vstack((X_flat, Y_flat, Z_flat)).T
        
        # --- ПРОТОН ---
        dp, _ = tree_p.query(slice_pts, workers=-1)
        dp = np.maximum(dp, 1e-12)
        u_p = (A_core / dp) * np.exp(-B_tail * dp)
        f_p = 2 * np.arctan(u_p)
        df_dp = (2 * u_p / (1 + u_p**2)) * (-1/dp - B_tail)
        sin_f_p = np.sin(f_p)
        sin_f_dp = np.where(dp < 1e-8, -df_dp, sin_f_p / dp)
        
        E_p = (sin_f_p**2 * (2 * df_dp**2 + sin_f_dp**2)) + \
              (dp**2 * df_dp**2 + 2 * sin_f_p**2) + \
              (3 * alpha * (1 - np.cos(f_p)) * dp**2)
        Mass_p_total += np.sum(E_p) * dV
        
        # --- НЕЙТРОН ---
        dn, _ = tree_n.query(slice_pts, workers=-1)
        dn = np.maximum(dn, 1e-12)
        u_n = (A_core / dn) * np.exp(-B_tail * dn)
        f_n = 2 * np.arctan(u_n)
        df_dn = (2 * u_n / (1 + u_n**2)) * (-1/dn - B_tail)
        sin_f_n = np.sin(f_n)
        sin_f_dn = np.where(dn < 1e-8, -df_dn, sin_f_n / dn)
        
        E_n = (sin_f_n**2 * (2 * df_dn**2 + sin_f_dn**2)) + \
              (dn**2 * df_dn**2 + 2 * sin_f_n**2) + \
              (3 * alpha * (1 - np.cos(f_n)) * dn**2)
        Mass_n_total += np.sum(E_n) * dV
        
        # Сохраняем центральный срез для визуализации
        if i == mid_z_index:
            E_p_slice = E_p.reshape((grid_size, grid_size))
            E_n_slice = E_n.reshape((grid_size, grid_size))

        # Индикатор прогресса
        if (i + 1) % max(1, grid_size//10) == 0 or i == grid_size - 1:
            sys.stdout.write(f"\rОбработано срезов: {i+1}/{grid_size} | Прогресс: {((i+1)/grid_size)*100:.0f}%")
            sys.stdout.flush()
            
    exec_time = time.time() - start_time
    print("\nСохранение результатов в кэш...")
    np.savez_compressed(cache_filename, 
                        Mass_p_total=Mass_p_total, 
                        Mass_n_total=Mass_n_total,
                        E_p_slice=E_p_slice, 
                        E_n_slice=E_n_slice)

# =====================================================================
# 4. АНАЛИТИКА
# =====================================================================
Delta_Mass = Mass_n_total - Mass_p_total
Ratio = Delta_Mass / Mass_p_total

Mass_p_MeV = 938.272088
Mass_defect_MeV = Mass_p_MeV * Ratio
Exp_defect = 1.293332 # Экспериментальный дефект в МэВ

print("\n" + "="*70)
print(f" РЕЗУЛЬТАТЫ (Сетка: {grid_size}^3)")
print("="*70)
print(f"Безразмерная Масса Протона : {Mass_p_total:.6f}")
print(f"Безразмерная Масса Нейтрона: {Mass_n_total:.6f}")
print(f"Безразмерная Дельта (Пружина): {Delta_Mass:.6f}")
print("-" * 70)
print(f"Расчетный Дефект Масс      : {Mass_defect_MeV:.5f} МэВ")
print(f"Экспериментальный Дефект   : {Exp_defect:.5f} МэВ")
print(f"Погрешность модели         : {abs(Mass_defect_MeV - Exp_defect)/Exp_defect * 100:.2f} %")
print("="*70)
print(f"Время расчетов/загрузки    : {exec_time:.2f} сек")
print("=====================================================================")

# =====================================================================
# 5. ВИЗУАЛИЗАЦИЯ РЕЗУЛЬТАТОВ
# =====================================================================
print("\nГенерация визуализаций... Закройте окна графиков для завершения.")

fig = plt.figure(figsize=(16, 10))
plt.style.use('dark_background')
fig.suptitle(f'Топологическая модель Нуклонов (Сетка: {grid_size}^3)', fontsize=18, fontweight='bold')

# --- ГРАФИК 1: 3D Геометрия Узлов ---
ax1 = fig.add_subplot(2, 2, 1, projection='3d')
ax1.plot(proton_curve[:,0], proton_curve[:,1], proton_curve[:,2], 
         label='Протон (Гладкий узел)', color='cyan', alpha=0.6, lw=2)
ax1.plot(neutron_curve[:,0], neutron_curve[:,1], neutron_curve[:,2], 
         label='Нейтрон (Скрутка)', color='magenta', alpha=0.8, lw=1)
ax1.set_title("3D Каркасы (Торические Узлы)")
ax1.set_axis_off()
ax1.legend()

# --- ГРАФИК 2: Тепловая карта плотности энергии Протона (срез Z=0) ---
ax2 = fig.add_subplot(2, 2, 2)
img2 = ax2.imshow(E_p_slice.T, extent=[-bound, bound, -bound, bound], 
                  origin='lower', cmap='inferno', norm=LogNorm(vmin=1e-4, vmax=E_p_slice.max()))
ax2.set_title("Томограмма энергии Протона (Срез Z=0)")
ax2.set_xlabel("X")
ax2.set_ylabel("Y")
fig.colorbar(img2, ax=ax2, label="Плотность энергии (лог. масштаб)")

# --- ГРАФИК 3: Тепловая карта плотности энергии Нейтрона (срез Z=0) ---
ax3 = fig.add_subplot(2, 2, 3)
img3 = ax3.imshow(E_n_slice.T, extent=[-bound, bound, -bound, bound], 
                  origin='lower', cmap='magma', norm=LogNorm(vmin=1e-4, vmax=E_n_slice.max()))
ax3.set_title("Томограмма энергии Нейтрона (Срез Z=0)")
ax3.set_xlabel("X")
ax3.set_ylabel("Y")
fig.colorbar(img3, ax=ax3, label="Плотность энергии (лог. масштаб)")

# --- ГРАФИК 4: Сравнение дефекта масс ---
ax4 = fig.add_subplot(2, 2, 4)
bars = ax4.bar(["Эксперимент", "Расчет модели"], [Exp_defect, Mass_defect_MeV], color=['#44aa99', '#ddcc77'])
ax4.set_title("Сравнение Дефекта Масс (МэВ)")
ax4.set_ylabel("Энергия, МэВ")
ax4.set_ylim(0, max(Exp_defect, Mass_defect_MeV) * 1.3)

# Добавляем значения над столбцами
for bar in bars:
    yval = bar.get_height()
    ax4.text(bar.get_x() + bar.get_width()/2, yval + 0.05, f"{yval:.4f}", 
             ha='center', va='bottom', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.subplots_adjust(top=0.92)
plt.show()
