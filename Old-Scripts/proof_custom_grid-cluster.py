import numpy as np
import time
import sys
import os
import threading
import queue
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import multiprocessing
import warnings

warnings.filterwarnings('ignore', category=SyntaxWarning)

# =====================================================================
# ИНИЦИАЛИЗАЦИЯ И ПОИСК ЖЕЛЕЗА
# =====================================================================
print("="*75)
print(" ТОПОЛОГИЧЕСКАЯ МОДЕЛЬ: ГИБРИДНЫЙ КЛАСТЕР (CPU + MULTI-GPU v6)")
print("="*75)

try:
    import torch
    HAS_GPU = torch.cuda.is_available()
    NUM_GPUS = torch.cuda.device_count() if HAS_GPU else 0
except ImportError:
    HAS_GPU = False
    NUM_GPUS = 0

from scipy.spatial import cKDTree

print(f"[Железо] Процессор: Обнаружено {multiprocessing.cpu_count()} логических ядер (Scipy cKDTree)")
if HAS_GPU:
    print(f"[Железо] Видеокарты: Обнаружено {NUM_GPUS} GPU (PyTorch CUDA)")
    for i in range(NUM_GPUS):
        print(f"         - GPU {i}: {torch.cuda.get_device_name(i)}")
else:
    print("[Железо] GPU не обнаружены или не установлен PyTorch.")

# =====================================================================
# ПАРАМЕТРЫ СЕТКИ
# =====================================================================
grid_input = input("\nВведите размер сетки N (N^3 квантов, по умолчанию 200): ")
grid_size = int(grid_input) if grid_input.strip().isdigit() else 200

bound = 5.0
dV = (2 * bound / grid_size)**3
cache_filename = f"cache_hybrid_v6_grid_{grid_size}.npz"

print(f"Разрешение сетки : {grid_size}x{grid_size}x{grid_size}")
print(f"Квантов объема   : {grid_size**3:,}")

A_core = 0.8168
alpha = 0.00729735
B_tail = np.sqrt(alpha)

# =====================================================================
# ГЕНЕРАЦИЯ КРИВЫХ
# =====================================================================
def generate_nucleon_curve(beta_phase, num_points=15000):
    t = np.linspace(0, 2 * np.pi, num_points)
    R_torus, a_torus = 2.0, 0.8
    xp = (R_torus + a_torus * np.cos(3 * t)) * np.cos(2 * t)
    yp = (R_torus + a_torus * np.cos(3 * t)) * np.sin(2 * t)
    zp = a_torus * np.sin(3 * t)

    if beta_phase > 0:
        twist_freq = 15  
        twist_amp = 0.0247371 * beta_phase 
        x = xp + twist_amp * np.cos(twist_freq * t)
        y = yp + twist_amp * np.sin(twist_freq * t)
        z = zp + twist_amp * np.cos(twist_freq * t + np.pi/2)
        return np.vstack([x, y, z]).T
    else:
        return np.vstack([xp, yp, zp]).T

proton_curve = generate_nucleon_curve(0.0)
neutron_curve = generate_nucleon_curve(1.0)

# =====================================================================
# ФУНКЦИИ-РАБОЧИЕ ДЛЯ ПОТОКОВ (WORKERS)
# =====================================================================
# Рабочий для процессора (Ryzen)
def worker_cpu(task_q, res_q, X_flat, Y_flat, z_g, mid_z_index):
    tree_p = cKDTree(proton_curve)
    tree_n = cKDTree(neutron_curve)
    # Оставляем пару потоков для обслуживания GPU, остальные под KDTree
    cpu_workers = max(1, multiprocessing.cpu_count() - 2) 
    
    while True:
        try: i = task_q.get_nowait()
        except queue.Empty: break
        
        Z_flat = np.full_like(X_flat, z_g[i])
        slice_pts = np.vstack((X_flat, Y_flat, Z_flat)).T
        
        dp, _ = tree_p.query(slice_pts, workers=cpu_workers)
        dp = np.maximum(dp, 1e-12)
        u_p = (A_core / dp) * np.exp(-B_tail * dp)
        f_p = 2 * np.arctan(u_p)
        df_dp = (2 * u_p / (1 + u_p**2)) * (-1/dp - B_tail)
        sin_f_p = np.sin(f_p)
        sin_f_dp = np.where(dp < 1e-8, -df_dp, sin_f_p / dp)
        E_p = (sin_f_p**2 * (2 * df_dp**2 + sin_f_dp**2)) + (dp**2 * df_dp**2 + 2 * sin_f_p**2) + (3 * alpha * (1 - np.cos(f_p)) * dp**2)
        
        dn, _ = tree_n.query(slice_pts, workers=cpu_workers)
        dn = np.maximum(dn, 1e-12)
        u_n = (A_core / dn) * np.exp(-B_tail * dn)
        f_n = 2 * np.arctan(u_n)
        df_dn = (2 * u_n / (1 + u_n**2)) * (-1/dn - B_tail)
        sin_f_n = np.sin(f_n)
        sin_f_dn = np.where(dn < 1e-8, -df_dn, sin_f_n / dn)
        E_n = (sin_f_n**2 * (2 * df_dn**2 + sin_f_dn**2)) + (dn**2 * df_dn**2 + 2 * sin_f_n**2) + (3 * alpha * (1 - np.cos(f_n)) * dn**2)
        
        slice_p = E_p.reshape((grid_size, grid_size)) if i == mid_z_index else None
        slice_n = E_n.reshape((grid_size, grid_size)) if i == mid_z_index else None
        
        res_q.put((i, 'CPU', np.sum(E_p), np.sum(E_n), slice_p, slice_n))
        task_q.task_done()

# Рабочий для видеокарт (RTX 3060)
def worker_gpu(device_id, task_q, res_q, X_flat, Y_flat, z_g, mid_z_index):
    device = torch.device(f'cuda:{device_id}')
    curve_p_tc = torch.tensor(proton_curve, dtype=torch.float32, device=device)
    curve_n_tc = torch.tensor(neutron_curve, dtype=torch.float32, device=device)
    CHUNK_SIZE = 50000 
    
    while True:
        try: i = task_q.get_nowait()
        except queue.Empty: break
        
        Z_flat = np.full_like(X_flat, z_g[i])
        slice_pts = np.vstack((X_flat, Y_flat, Z_flat)).T
        
        E_p_sum, E_n_sum = 0.0, 0.0
        slice_p_full = np.zeros(len(X_flat)) if i == mid_z_index else None
        slice_n_full = np.zeros(len(X_flat)) if i == mid_z_index else None
        
        for c in range(0, len(X_flat), CHUNK_SIZE):
            chunk = slice_pts[c:c+CHUNK_SIZE]
            pts_tc = torch.tensor(chunk, dtype=torch.float32, device=device)
            
            # Протон
            dp_tc = torch.clamp(torch.cdist(pts_tc, curve_p_tc).min(dim=1)[0].to(torch.float64), min=1e-12)
            u_p = (A_core / dp_tc) * torch.exp(-B_tail * dp_tc)
            f_p = 2 * torch.atan(u_p)
            df_dp = (2 * u_p / (1 + u_p**2)) * (-1/dp_tc - B_tail)
            sin_f_p = torch.sin(f_p)
            sin_f_dp = torch.where(dp_tc < 1e-8, -df_dp, sin_f_p / dp_tc)
            E_p = (sin_f_p**2 * (2 * df_dp**2 + sin_f_dp**2)) + (dp_tc**2 * df_dp**2 + 2 * sin_f_p**2) + (3 * alpha * (1 - torch.cos(f_p)) * dp_tc**2)
            
            E_p_sum += E_p.sum().item()
            if i == mid_z_index: slice_p_full[c:c+CHUNK_SIZE] = E_p.cpu().numpy()
            
            # Нейтрон
            dn_tc = torch.clamp(torch.cdist(pts_tc, curve_n_tc).min(dim=1)[0].to(torch.float64), min=1e-12)
            u_n = (A_core / dn_tc) * torch.exp(-B_tail * dn_tc)
            f_n = 2 * torch.atan(u_n)
            df_dn = (2 * u_n / (1 + u_n**2)) * (-1/dn_tc - B_tail)
            sin_f_n = torch.sin(f_n)
            sin_f_dn = torch.where(dn_tc < 1e-8, -df_dn, sin_f_n / dn_tc)
            E_n = (sin_f_n**2 * (2 * df_dn**2 + sin_f_dn**2)) + (dn_tc**2 * df_dn**2 + 2 * sin_f_n**2) + (3 * alpha * (1 - torch.cos(f_n)) * dn_tc**2)
            
            E_n_sum += E_n.sum().item()
            if i == mid_z_index: slice_n_full[c:c+CHUNK_SIZE] = E_n.cpu().numpy()

        slice_p = slice_p_full.reshape((grid_size, grid_size)) if i == mid_z_index else None
        slice_n = slice_n_full.reshape((grid_size, grid_size)) if i == mid_z_index else None
        
        res_q.put((i, f'GPU:{device_id}', E_p_sum, E_n_sum, slice_p, slice_n))
        task_q.task_done()

# =====================================================================
# ГЛАВНЫЙ БЛОК УПРАВЛЕНИЯ
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
    print(f"\nРаздача задач на распределенный кластер (Слоев: {grid_size})...")
    
    x_g = np.linspace(-bound, bound, grid_size)
    y_g = np.linspace(-bound, bound, grid_size)
    z_g = np.linspace(-bound, bound, grid_size)
    X, Y = np.meshgrid(x_g, y_g, indexing='ij')
    X_flat, Y_flat = X.ravel(), Y.ravel()
    mid_z_index = grid_size // 2
    
    task_queue = queue.Queue()
    res_queue = queue.Queue()
    
    # Загружаем очередь задач
    for i in range(grid_size):
        task_queue.put(i)
        
    threads =[]
    
    # Запуск CPU потока
    t_cpu = threading.Thread(target=worker_cpu, args=(task_queue, res_queue, X_flat, Y_flat, z_g, mid_z_index))
    t_cpu.start()
    threads.append(t_cpu)
    
    # Запуск GPU потоков (по одному на каждую видеокарту)
    if HAS_GPU:
        for dev_id in range(NUM_GPUS):
            t_gpu = threading.Thread(target=worker_gpu, args=(dev_id, task_queue, res_queue, X_flat, Y_flat, z_g, mid_z_index))
            t_gpu.start()
            threads.append(t_gpu)

    # Мониторинг и сбор результатов
    processed = 0
    Mass_p_sum_raw = 0.0
    Mass_n_sum_raw = 0.0
    E_p_slice, E_n_slice = None, None
    
    stats = {'CPU': 0}
    if HAS_GPU:
        for dev_id in range(NUM_GPUS):
            stats[f'GPU:{dev_id}'] = 0

    while processed < grid_size:
        i, worker_name, ep_sum, en_sum, sp, sn = res_queue.get()
        Mass_p_sum_raw += ep_sum
        Mass_n_sum_raw += en_sum
        if sp is not None: E_p_slice = sp
        if sn is not None: E_n_slice = sn
        
        stats[worker_name] += 1
        processed += 1
        
        # Красивый вывод статистики кто сколько слоев посчитал
        stats_str = " | ".join([f"{k}: {v}" for k, v in stats.items()])
        sys.stdout.write(f"\r[{processed}/{grid_size}] {int((processed)/grid_size*100)}% -> [{stats_str}]    ")
        sys.stdout.flush()

    # Ждем корректного завершения потоков
    task_queue.join()
    for t in threads: t.join()

    # Финальное умножение на дифференциал объема
    Mass_p_total = Mass_p_sum_raw * dV
    Mass_n_total = Mass_n_sum_raw * dV

    exec_time = time.time() - start_time
    print("\n\nСохранение результатов в кэш...")
    np.savez_compressed(cache_filename, Mass_p_total=Mass_p_total, Mass_n_total=Mass_n_total, E_p_slice=E_p_slice, E_n_slice=E_n_slice)

# =====================================================================
# АНАЛИТИКА
# =====================================================================
Delta_Mass = Mass_n_total - Mass_p_total
Ratio = Delta_Mass / Mass_p_total

Mass_p_MeV = 938.272088
Mass_defect_MeV = Mass_p_MeV * Ratio
Exp_defect = 1.293332 

print("\n" + "="*75)
print(f" РЕЗУЛЬТАТЫ СИМУЛЯЦИИ (Сетка: {grid_size}^3)")
print("="*75)
print(f"Расчетный Дефект Масс        : {Mass_defect_MeV:.5f} МэВ")
print(f"Экспериментальный Дефект     : {Exp_defect:.5f} МэВ")
print(f"Погрешность модели           : {abs(Mass_defect_MeV - Exp_defect)/Exp_defect * 100:.3f} %")
print("="*75)
print(f"Время выполнения кластера    : {exec_time:.2f} сек")
print("=====================================================================")

# Графики (как в v5, черный фон)
plt.style.use('dark_background')
fig = plt.figure(figsize=(16, 9), facecolor='black')
fig.suptitle(f'Топология Нуклонов (Сетка: {grid_size}^3)', fontsize=18, fontweight='bold', color='white')

ax1 = fig.add_subplot(2, 2, 1, projection='3d')
ax1.set_facecolor('black')
ax1.xaxis.set_pane_color((0,0,0,0)); ax1.yaxis.set_pane_color((0,0,0,0)); ax1.zaxis.set_pane_color((0,0,0,0))
ax1.plot(proton_curve[:,0], proton_curve[:,1], proton_curve[:,2], color='#00ffff', alpha=0.6, lw=3, label='Протон')
ax1.plot(neutron_curve[:,0], neutron_curve[:,1], neutron_curve[:,2], color='#ff00ff', alpha=0.8, lw=1.2, label='Нейтрон')
ax1.set_axis_off()
ax1.legend(loc='upper left', facecolor='black', edgecolor='white', labelcolor='white')

ax2 = fig.add_subplot(2, 2, 2)
ax2.set_facecolor('black')
img2 = ax2.imshow(E_p_slice.T, extent=[-bound, bound, -bound, bound], origin='lower', cmap='inferno', norm=LogNorm(vmin=1e-5, vmax=E_p_slice.max()))
ax2.axis('off'); fig.colorbar(img2, ax=ax2, fraction=0.046, pad=0.04).ax.yaxis.set_tick_params(color='white')

ax3 = fig.add_subplot(2, 2, 3)
ax3.set_facecolor('black')
img3 = ax3.imshow(E_n_slice.T, extent=[-bound, bound, -bound, bound], origin='lower', cmap='magma', norm=LogNorm(vmin=1e-5, vmax=E_n_slice.max()))
ax3.axis('off'); fig.colorbar(img3, ax=ax3, fraction=0.046, pad=0.04).ax.yaxis.set_tick_params(color='white')

ax4 = fig.add_subplot(2, 2, 4)
ax4.set_facecolor('black')
beta_phases = np.linspace(0, 1, 100)
ax4.plot(beta_phases, Exp_defect * (beta_phases**2), color='#ffaa00', lw=4, label=r'Накопление энергии')
ax4.axvline(x=0, color='#00ffff', linestyle='--', lw=2, label='Протон (Яма стабильности)')
ax4.axvline(x=1, color='#ff00ff', linestyle='--', lw=2, label='Нейтрон (Метастабильность)')
ax4.set_xlabel("Параметр скручивания трубки β", color='white')
ax4.set_ylabel("Дефект Масс, МэВ", color='white')
ax4.tick_params(colors='white'); ax4.grid(True, color='#333333', linestyle=':')
ax4.legend(loc='upper left', facecolor='black', edgecolor='white', labelcolor='white')

plt.tight_layout()
plt.subplots_adjust(top=0.9)
plt.show()
