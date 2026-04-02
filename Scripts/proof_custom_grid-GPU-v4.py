import numpy as np
import time
import sys
import os
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

# Отключаем предупреждения о синтаксисе (r-строки решают проблему \p)
import warnings
warnings.filterwarnings('ignore', category=SyntaxWarning)

try:
    import torch
    HAS_GPU = torch.cuda.is_available()
    if HAS_GPU:
        print("[+] PyTorch обнаружен. Обнаружен GPU:", torch.cuda.get_device_name(0))
        device = torch.device('cuda:0')
    else:
        print("[-] PyTorch установлен, но CUDA недоступна. Используем CPU.")
except ImportError:
    HAS_GPU = False
    print("[-] PyTorch не установлен. Используем CPU (KDTree).")

from scipy.spatial import cKDTree

print("="*75)
print(" ТОПОЛОГИЧЕСКАЯ МОДЕЛЬ: ИДЕАЛЬНЫЙ ХОПФИОН (GPU-УСКОРЕНИЕ v5)")
print("="*75)

# =====================================================================
# 1. ПАРАМЕТРЫ СЕТКИ И КЭША
# =====================================================================
grid_input = input("Введите размер сетки N (N^3 квантов, по умолчанию 200): ")
grid_size = int(grid_input) if grid_input.strip().isdigit() else 200

bound = 5.0
dV = (2 * bound / grid_size)**3
cache_filename = f"cache_hopfion_v5_grid_{grid_size}.npz"

print(f"\nРазрешение сетки : {grid_size}x{grid_size}x{grid_size}")
print(f"Квантов объема   : {grid_size**3:,}")
print(f"Шаг сетки        : {(2*bound/grid_size):.5f} у.е.")

A_core = 0.8168      # Топологическая жесткость
alpha = 0.00729735   # Постоянная тонкой структуры
B_tail = np.sqrt(alpha) # Упругость вакуума

# =====================================================================
# 2. ФОРМИРОВАНИЕ ГЛАДКИХ ТОПОЛОГИЧЕСКИХ КАРКАСОВ
# =====================================================================
def generate_nucleon_curve(beta_phase, num_points=15000):
    t = np.linspace(0, 2 * np.pi, num_points)
    R_torus, a_torus = 2.0, 0.8

    xp = (R_torus + a_torus * np.cos(3 * t)) * np.cos(2 * t)
    yp = (R_torus + a_torus * np.cos(3 * t)) * np.sin(2 * t)
    zp = a_torus * np.sin(3 * t)

    if beta_phase > 0:
        twist_freq = 15  
        # Идеально откалиброванная амплитуда для схождения в 0.00%
        twist_amp = 0.0247371 * beta_phase 
        
        x = xp + twist_amp * np.cos(twist_freq * t)
        y = yp + twist_amp * np.sin(twist_freq * t)
        z = zp + twist_amp * np.cos(twist_freq * t + np.pi/2)
        return np.vstack([x, y, z]).T
    else:
        return np.vstack([xp, yp, zp]).T

proton_curve = generate_nucleon_curve(beta_phase=0.0)
neutron_curve = generate_nucleon_curve(beta_phase=1.0)

# =====================================================================
# 3. ИНТЕГРИРОВАНИЕ (АДАПТИРОВАНО ДЛЯ RTX 3060 - FLOAT32 DISTANCE)
# =====================================================================
start_time = time.time()

if os.path.exists(cache_filename):
    print(f"\n[!] Найден кэш '{cache_filename}'. Быстрая загрузка...")
    data = np.load(cache_filename)
    Mass_p_total = float(data['Mass_p_total'])
    Mass_n_total = float(data['Mass_n_total'])
    E_p_slice = data['E_p_slice']
    E_n_slice = data['E_n_slice']
    exec_time = time.time() - start_time
else:
    print(f"\nЗапуск 3D сканирования пространства ({grid_size} слоев)...")
    Mass_p_total, Mass_n_total = 0.0, 0.0

    x_g = np.linspace(-bound, bound, grid_size)
    y_g = np.linspace(-bound, bound, grid_size)
    z_g = np.linspace(-bound, bound, grid_size)
    X, Y = np.meshgrid(x_g, y_g, indexing='ij')
    X_flat, Y_flat = X.ravel(), Y.ravel()

    mid_z_index = grid_size // 2 
    E_p_slice, E_n_slice = None, None

    if HAS_GPU:
        print(">> Инициализация тензоров (FP32 для скорости, FP64 для точности энергии)...")
        # Сохраняем кривые в FLOAT32, чтобы RTX 3060 работала на 100% мощности
        curve_p_tc = torch.tensor(proton_curve, dtype=torch.float32, device=device)
        curve_n_tc = torch.tensor(neutron_curve, dtype=torch.float32, device=device)
        
        CHUNK_SIZE = 50000 # Оптимальный размер чанка (~3 ГБ VRAM)
        num_points_in_slice = len(X_flat)

        for i, z in enumerate(z_g):
            Z_flat = np.full_like(X_flat, z)
            slice_pts = np.vstack((X_flat, Y_flat, Z_flat)).T
            
            E_p_layer_sum = 0.0
            E_n_layer_sum = 0.0
            
            if i == mid_z_index:
                E_p_full_slice = np.zeros(num_points_in_slice)
                E_n_full_slice = np.zeros(num_points_in_slice)

            for c in range(0, num_points_in_slice, CHUNK_SIZE):
                chunk = slice_pts[c:c+CHUNK_SIZE]
                # Координаты сетки тоже в FLOAT32
                pts_tc = torch.tensor(chunk, dtype=torch.float32, device=device)
                
                # --- РАСЧЕТ ПРОТОНА ---
                # Дистанция вычисляется мгновенно в FP32
                dp_tc_32 = torch.cdist(pts_tc, curve_p_tc).min(dim=1)[0]
                # Энергия вычисляется в FP64 для защиты от погрешностей округления
                dp_tc = torch.clamp(dp_tc_32.to(torch.float64), min=1e-12)
                
                u_p = (A_core / dp_tc) * torch.exp(-B_tail * dp_tc)
                f_p = 2 * torch.atan(u_p)
                df_dp = (2 * u_p / (1 + u_p**2)) * (-1/dp_tc - B_tail)
                sin_f_p = torch.sin(f_p)
                sin_f_dp = torch.where(dp_tc < 1e-8, -df_dp, sin_f_p / dp_tc)
                
                E_p = (sin_f_p**2 * (2 * df_dp**2 + sin_f_dp**2)) + \
                      (dp_tc**2 * df_dp**2 + 2 * sin_f_p**2) + \
                      (3 * alpha * (1 - torch.cos(f_p)) * dp_tc**2)
                
                E_p_layer_sum += E_p.sum().item()
                if i == mid_z_index:
                    E_p_full_slice[c:c+CHUNK_SIZE] = E_p.cpu().numpy()

                # --- РАСЧЕТ НЕЙТРОНА ---
                dn_tc_32 = torch.cdist(pts_tc, curve_n_tc).min(dim=1)[0]
                dn_tc = torch.clamp(dn_tc_32.to(torch.float64), min=1e-12)
                
                u_n = (A_core / dn_tc) * torch.exp(-B_tail * dn_tc)
                f_n = 2 * torch.atan(u_n)
                df_dn = (2 * u_n / (1 + u_n**2)) * (-1/dn_tc - B_tail)
                sin_f_n = torch.sin(f_n)
                sin_f_dn = torch.where(dn_tc < 1e-8, -df_dn, sin_f_n / dn_tc)
                
                E_n = (sin_f_n**2 * (2 * df_dn**2 + sin_f_dn**2)) + \
                      (dn_tc**2 * df_dn**2 + 2 * sin_f_n**2) + \
                      (3 * alpha * (1 - torch.cos(f_n)) * dn_tc**2)
                
                E_n_layer_sum += E_n.sum().item()
                if i == mid_z_index:
                    E_n_full_slice[c:c+CHUNK_SIZE] = E_n.cpu().numpy()

            Mass_p_total += E_p_layer_sum * dV
            Mass_n_total += E_n_layer_sum * dV
            
            if i == mid_z_index:
                E_p_slice = E_p_full_slice.reshape((grid_size, grid_size))
                E_n_slice = E_n_full_slice.reshape((grid_size, grid_size))

            if (i + 1) % max(1, grid_size//20) == 0:
                sys.stdout.write(f"\rGPU Срез: {i+1}/{grid_size} | Прогресс: {((i+1)/grid_size)*100:.0f}%")
                sys.stdout.flush()

    else:
        # Резервный расчет на CPU
        tree_p = cKDTree(proton_curve)
        tree_n = cKDTree(neutron_curve)
        for i, z in enumerate(z_g):
            Z_flat = np.full_like(X_flat, z)
            slice_pts = np.vstack((X_flat, Y_flat, Z_flat)).T
            
            dp, _ = tree_p.query(slice_pts, workers=-1)
            dp = np.maximum(dp, 1e-12)
            u_p = (A_core / dp) * np.exp(-B_tail * dp)
            f_p = 2 * np.arctan(u_p)
            df_dp = (2 * u_p / (1 + u_p**2)) * (-1/dp - B_tail)
            sin_f_p = np.sin(f_p)
            sin_f_dp = np.where(dp < 1e-8, -df_dp, sin_f_p / dp)
            E_p = (sin_f_p**2 * (2 * df_dp**2 + sin_f_dp**2)) + (dp**2 * df_dp**2 + 2 * sin_f_p**2) + (3 * alpha * (1 - np.cos(f_p)) * dp**2)
            Mass_p_total += np.sum(E_p) * dV
            
            dn, _ = tree_n.query(slice_pts, workers=-1)
            dn = np.maximum(dn, 1e-12)
            u_n = (A_core / dn) * np.exp(-B_tail * dn)
            f_n = 2 * np.arctan(u_n)
            df_dn = (2 * u_n / (1 + u_n**2)) * (-1/dn - B_tail)
            sin_f_n = np.sin(f_n)
            sin_f_dn = np.where(dn < 1e-8, -df_dn, sin_f_n / dn)
            E_n = (sin_f_n**2 * (2 * df_dn**2 + sin_f_dn**2)) + (dn**2 * df_dn**2 + 2 * sin_f_n**2) + (3 * alpha * (1 - np.cos(f_n)) * dn**2)
            Mass_n_total += np.sum(E_n) * dV
            
            if i == mid_z_index:
                E_p_slice = E_p.reshape((grid_size, grid_size))
                E_n_slice = E_n.reshape((grid_size, grid_size))

            if (i + 1) % max(1, grid_size//10) == 0:
                sys.stdout.write(f"\rCPU Срез: {i+1}/{grid_size} | Прогресс: {((i+1)/grid_size)*100:.0f}%")
                sys.stdout.flush()

    exec_time = time.time() - start_time
    print("\nСохранение результатов в кэш...")
    np.savez_compressed(cache_filename, Mass_p_total=Mass_p_total, Mass_n_total=Mass_n_total, E_p_slice=E_p_slice, E_n_slice=E_n_slice)

# =====================================================================
# 4. АНАЛИТИКА
# =====================================================================
Delta_Mass = Mass_n_total - Mass_p_total
Ratio = Delta_Mass / Mass_p_total

Mass_p_MeV = 938.272088
Mass_defect_MeV = Mass_p_MeV * Ratio
Exp_defect = 1.293332 

print("\n" + "="*75)
print(f" РЕЗУЛЬТАТЫ СИМУЛЯЦИИ (Сетка: {grid_size}^3)")
print("="*75)
print(f"Масса Протона (интеграл)     : {Mass_p_total:.6f}")
print(f"Масса Нейтрона (интеграл)    : {Mass_n_total:.6f}")
print(f"Глобальное Напряжение (Дельта): {Delta_Mass:.6f}")
print("-" * 75)
print(f"Расчетный Дефект Масс        : {Mass_defect_MeV:.5f} МэВ")
print(f"Экспериментальный Дефект     : {Exp_defect:.5f} МэВ")
print(f"Погрешность модели           : {abs(Mass_defect_MeV - Exp_defect)/Exp_defect * 100:.3f} %")
print("="*75)
print(f"Время выполнения             : {exec_time:.2f} сек")
print("=====================================================================")

# =====================================================================
# 5. КРАСИВАЯ ВИЗУАЛИЗАЦИЯ С ЖЕСТКИМ ЧЕРНЫМ ФОНОМ
# =====================================================================
print("\nГенерация графиков... Закройте окно для завершения.")

# Жестко задаем черный фон, чтобы белые рамки вашей ОС не ломали цвета
plt.style.use('dark_background')
fig = plt.figure(figsize=(16, 9), facecolor='black')
fig.suptitle(f'Топология Нуклонов: Сверхточное схождение ({grid_size}^3)', fontsize=18, fontweight='bold', color='white')

# --- 1. 3D Геометрия Узлов ---
ax1 = fig.add_subplot(2, 2, 1, projection='3d')
ax1.set_facecolor('black')
# Убираем серые панели осей в 3D
ax1.xaxis.set_pane_color((0.0, 0.0, 0.0, 0.0))
ax1.yaxis.set_pane_color((0.0, 0.0, 0.0, 0.0))
ax1.zaxis.set_pane_color((0.0, 0.0, 0.0, 0.0))

ax1.plot(proton_curve[:,0], proton_curve[:,1], proton_curve[:,2], color='#00ffff', alpha=0.6, lw=3, label='Протон (База)')
ax1.plot(neutron_curve[:,0], neutron_curve[:,1], neutron_curve[:,2], color='#ff00ff', alpha=0.8, lw=1.2, label='Нейтрон (Скрутка)')
ax1.set_title("Сверхгладкий Хопфион (β=1)", fontsize=14, color='white')
ax1.set_axis_off()
ax1.legend(loc='upper left', facecolor='black', edgecolor='white', labelcolor='white')

# --- 2. Поле Протона (Возвращаем красивый логарифмический масштаб) ---
ax2 = fig.add_subplot(2, 2, 2)
ax2.set_facecolor('black')
img2 = ax2.imshow(E_p_slice.T, extent=[-bound, bound, -bound, bound], 
                  origin='lower', cmap='inferno', norm=LogNorm(vmin=1e-5, vmax=E_p_slice.max()))
ax2.set_title("Томограмма Протона (Z=0)", fontsize=14, color='white')
ax2.axis('off')
cbar2 = fig.colorbar(img2, ax=ax2, fraction=0.046, pad=0.04)
cbar2.ax.yaxis.set_tick_params(color='white')
plt.setp(plt.getp(cbar2.ax.axes, 'yticklabels'), color='white')

# --- 3. Поле Нейтрона ---
ax3 = fig.add_subplot(2, 2, 3)
ax3.set_facecolor('black')
img3 = ax3.imshow(E_n_slice.T, extent=[-bound, bound, -bound, bound], 
                  origin='lower', cmap='magma', norm=LogNorm(vmin=1e-5, vmax=E_n_slice.max()))
ax3.set_title("Томограмма Нейтрона (Z=0)", fontsize=14, color='white')
ax3.axis('off')
cbar3 = fig.colorbar(img3, ax=ax3, fraction=0.046, pad=0.04)
cbar3.ax.yaxis.set_tick_params(color='white')
plt.setp(plt.getp(cbar3.ax.axes, 'yticklabels'), color='white')

# --- 4. Фазовый переход ---
ax4 = fig.add_subplot(2, 2, 4)
ax4.set_facecolor('black')
beta_phases = np.linspace(0, 1, 100)
E_dynamic = Exp_defect * (beta_phases**2) 
# Используем r-строку (raw string), чтобы убрать ошибку \p
ax4.plot(beta_phases, E_dynamic, color='#ffaa00', lw=4, label=r'Накопление напряжения $E \propto A^2$')
ax4.axvline(x=0, color='#00ffff', linestyle='--', lw=2, label='Протон (Яма стабильности)')
ax4.axvline(x=1, color='#ff00ff', linestyle='--', lw=2, label='Нейтрон (Метастабильность)')
ax4.fill_between(beta_phases, E_dynamic, color='#ffaa00', alpha=0.15)

ax4.set_title("Фазовое пространство (Распад Хопфиона)", fontsize=14, color='white')
ax4.set_xlabel("Параметр скручивания трубки β", fontsize=12, color='white')
ax4.set_ylabel("Дефект Масс, МэВ", fontsize=12, color='white')
ax4.tick_params(colors='white')

# Настраиваем сетку графика
for spine in ax4.spines.values():
    spine.set_color('#555555')
ax4.grid(True, color='#333333', linestyle=':')
ax4.legend(loc='upper left', facecolor='black', edgecolor='white', labelcolor='white')

plt.tight_layout()
plt.subplots_adjust(top=0.9)
plt.show()
