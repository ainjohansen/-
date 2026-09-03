import numpy as np
import time
import sys
import os
import threading
import queue
import matplotlib.pyplot as plt
import multiprocessing
import warnings

warnings.filterwarnings('ignore', category=SyntaxWarning)

# =====================================================================
# ИНИЦИАЛИЗАЦИЯ И ПОИСК ЖЕЛЕЗА
# =====================================================================
print("="*75)
print(" ЭКСПЕРИМЕНТ: ЗАВИСИМОСТЬ ДЕФЕКТА МАСС ОТ СКРУЧИВАНИЯ УЗЛА")
print("="*75)

try:
    import torch
    HAS_GPU = torch.cuda.is_available()
    NUM_GPUS = torch.cuda.device_count() if HAS_GPU else 0
except ImportError:
    HAS_GPU = False
    NUM_GPUS = 0

from scipy.spatial import cKDTree

print(f"[Железо] Процессор: Обнаружено {multiprocessing.cpu_count()} логических ядер")
if HAS_GPU:
    print(f"[Железо] Видеокарты: Обнаружено {NUM_GPUS} GPU")
    for i in range(NUM_GPUS):
        print(f"         - GPU {i}: {torch.cuda.get_device_name(i)}")
else:
    print("[Железо] GPU не обнаружены или не установлен PyTorch.")

# =====================================================================
# ПАРАМЕТРЫ СЕТКИ И ЭКСПЕРИМЕНТА
# =====================================================================
grid_input = input("\nВведите размер сетки N (рекомендуется 150-250 для точности): ")
grid_size = int(grid_input) if grid_input.strip().isdigit() else 200

bound = 5.0
dV = (2 * bound / grid_size)**3
cache_filename = f"cache_experiment_phases_grid_{grid_size}.npz"

print(f"Разрешение сетки : {grid_size}x{grid_size}x{grid_size}")
print(f"Квантов объема   : {grid_size**3:,}")

A_core = 0.8168
alpha = 0.00729735
B_tail = np.sqrt(alpha)

# Фазы скручивания для тестирования
test_phases = [0.0, np.pi/8, np.pi/4, 3*np.pi/8, np.pi/2, 1.0]
phase_labels = ["0 (Протон)", "π/8", "π/4", "3π/8", "π/2", "1.0 (Нейтрон v6)"]

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

# =====================================================================
# УНИВЕРСАЛЬНЫЕ ФУНКЦИИ-РАБОЧИЕ (Считают 1 переданную кривую)
# =====================================================================
def worker_cpu(task_q, res_q, X_flat, Y_flat, z_g, curve):
    tree = cKDTree(curve)
    cpu_workers = max(1, multiprocessing.cpu_count() - 2) 
    
    while True:
        try: i = task_q.get_nowait()
        except queue.Empty: break
        
        Z_flat = np.full_like(X_flat, z_g[i])
        slice_pts = np.vstack((X_flat, Y_flat, Z_flat)).T
        
        d, _ = tree.query(slice_pts, workers=cpu_workers)
        d = np.maximum(d, 1e-12)
        u = (A_core / d) * np.exp(-B_tail * d)
        f = 2 * np.arctan(u)
        df_dd = (2 * u / (1 + u**2)) * (-1/d - B_tail)
        sin_f = np.sin(f)
        sin_f_dd = np.where(d < 1e-8, -df_dd, sin_f / d)
        E = (sin_f**2 * (2 * df_dd**2 + sin_f_dd**2)) + (d**2 * df_dd**2 + 2 * sin_f**2) + (3 * alpha * (1 - np.cos(f)) * d**2)
        
        res_q.put((i, 'CPU', np.sum(E)))
        task_q.task_done()

def worker_gpu(device_id, task_q, res_q, X_flat, Y_flat, z_g, curve):
    device = torch.device(f'cuda:{device_id}')
    curve_tc = torch.tensor(curve, dtype=torch.float32, device=device)
    CHUNK_SIZE = 50000 
    
    while True:
        try: i = task_q.get_nowait()
        except queue.Empty: break
        
        Z_flat = np.full_like(X_flat, z_g[i])
        slice_pts = np.vstack((X_flat, Y_flat, Z_flat)).T
        
        E_sum = 0.0
        for c in range(0, len(X_flat), CHUNK_SIZE):
            chunk = slice_pts[c:c+CHUNK_SIZE]
            pts_tc = torch.tensor(chunk, dtype=torch.float32, device=device)
            
            d_tc = torch.clamp(torch.cdist(pts_tc, curve_tc).min(dim=1)[0].to(torch.float64), min=1e-12)
            u = (A_core / d_tc) * torch.exp(-B_tail * d_tc)
            f = 2 * torch.atan(u)
            df_dd = (2 * u / (1 + u**2)) * (-1/d_tc - B_tail)
            sin_f = torch.sin(f)
            sin_f_dd = torch.where(d_tc < 1e-8, -df_dd, sin_f / d_tc)
            E = (sin_f**2 * (2 * df_dd**2 + sin_f_dd**2)) + (d_tc**2 * df_dd**2 + 2 * sin_f**2) + (3 * alpha * (1 - torch.cos(f)) * d_tc**2)
            
            E_sum += E.sum().item()
            
        res_q.put((i, f'GPU:{device_id}', E_sum))
        task_q.task_done()

# =====================================================================
# ДВИЖОК РАСЧЕТА ОДНОЙ ФАЗЫ
# =====================================================================
def compute_mass_for_curve(curve, grid_size, X_flat, Y_flat, z_g):
    task_queue = queue.Queue()
    res_queue = queue.Queue()
    
    for i in range(grid_size):
        task_queue.put(i)
        
    threads = []
    t_cpu = threading.Thread(target=worker_cpu, args=(task_queue, res_queue, X_flat, Y_flat, z_g, curve))
    t_cpu.start()
    threads.append(t_cpu)
    
    if HAS_GPU:
        for dev_id in range(NUM_GPUS):
            t_gpu = threading.Thread(target=worker_gpu, args=(dev_id, task_queue, res_queue, X_flat, Y_flat, z_g, curve))
            t_gpu.start()
            threads.append(t_gpu)

    processed = 0
    E_total_raw = 0.0
    
    while processed < grid_size:
        i, worker_name, e_sum = res_queue.get()
        E_total_raw += e_sum
        processed += 1
        sys.stdout.write(f"\r  -> Прогресс слоя: [{processed}/{grid_size}] ({int(processed/grid_size*100)}%)    ")
        sys.stdout.flush()

    task_queue.join()
    for t in threads: t.join()
    print() 
    return E_total_raw

# =====================================================================
# ГЛАВНЫЙ ЦИКЛ ЭКСПЕРИМЕНТА
# =====================================================================
start_time = time.time()

x_g = np.linspace(-bound, bound, grid_size)
y_g = np.linspace(-bound, bound, grid_size)
z_g = np.linspace(-bound, bound, grid_size)
X, Y = np.meshgrid(x_g, y_g, indexing='ij')
X_flat, Y_flat = X.ravel(), Y.ravel()

results_mass = {}

if os.path.exists(cache_filename):
    print(f"\n[!] Найден кэш '{cache_filename}'. Загрузка результатов эксперимента...")
    data = np.load(cache_filename)
    cached_phases = data['phases']
    cached_masses = data['masses']
    results_mass = dict(zip(cached_phases, cached_masses))
    exec_time = time.time() - start_time
else:
    print("\nВНИМАНИЕ: Запущен мульти-фазный расчет. Это займет время, пропорциональное количеству фаз.")
    
    for phase, label in zip(test_phases, phase_labels):
        print(f"\n[Расчет] Фаза скручивания: {label} (beta = {phase:.4f})")
        curve = generate_nucleon_curve(phase)
        raw_mass = compute_mass_for_curve(curve, grid_size, X_flat, Y_flat, z_g)
        results_mass[phase] = raw_mass * dV
        
    exec_time = time.time() - start_time
    print("\nСохранение результатов эксперимента в кэш...")
    np.savez_compressed(cache_filename, phases=test_phases, masses=[results_mass[p] for p in test_phases])

# =====================================================================
# АНАЛИТИКА И ВЫВОД
# =====================================================================
Mass_p_MeV = 938.272088
Exp_defect = 1.293332 

mass_proton_base = results_mass[0.0] # Базовая масса при beta=0

print("\n" + "="*75)
print(f" РЕЗУЛЬТАТЫ ЭКСПЕРИМЕНТА (Сетка: {grid_size}^3)")
print("="*75)
print(f"{'Фаза (Угол)':<18} | {'Масса (у.е.)':<15} | {'Дефект (МэВ)':<15}")
print("-" * 55)

plot_phases = []
plot_defects = []

for phase, label in zip(test_phases, phase_labels):
    mass = results_mass[phase]
    delta = mass - mass_proton_base
    ratio = delta / mass_proton_base
    defect_mev = Mass_p_MeV * ratio
    
    plot_phases.append(phase)
    plot_defects.append(defect_mev)
    
    print(f"{label:<18} | {mass:<15.5f} | {defect_mev:<15.5f}")

print("="*75)
print(f"Время выполнения эксперимента: {exec_time:.2f} сек")

# =====================================================================
# ВИЗУАЛИЗАЦИЯ
# =====================================================================
plt.style.use('dark_background')
fig = plt.figure(figsize=(16, 10), facecolor='black')
fig.suptitle(f'Зависимость Дефекта Масс от степени скручивания (Сетка: {grid_size}^3)', fontsize=18, fontweight='bold', color='white')

# График 1: Дефект масс от фазы
ax1 = fig.add_subplot(2, 2, 1)
ax1.set_facecolor('black')
ax1.plot(plot_phases, plot_defects, marker='o', color='#00ffff', lw=3, markersize=8, label='Расчетная $\Delta M$')
ax1.axhline(y=Exp_defect, color='#ff00ff', linestyle='--', lw=2, label=f'Эксперимент ({Exp_defect} МэВ)')
ax1.set_xlabel("Степень скручивания (beta / радианы)", color='white', fontsize=12)
ax1.set_ylabel("Дефект Масс (МэВ)", color='white', fontsize=12)
ax1.tick_params(colors='white')
ax1.grid(True, color='#333333', linestyle=':')
ax1.legend(facecolor='black', edgecolor='white', labelcolor='white', fontsize=11)

# Вспомогательная функция для отрисовки 3D узлов
def plot_3d_knot(ax, phase, title, color):
    ax.set_facecolor('black')
    ax.xaxis.set_pane_color((0,0,0,0)); ax.yaxis.set_pane_color((0,0,0,0)); ax.zaxis.set_pane_color((0,0,0,0))
    curve = generate_nucleon_curve(phase)
    ax.plot(curve[:,0], curve[:,1], curve[:,2], color=color, lw=2, alpha=0.8)
    ax.set_title(title, color='white', fontsize=14)
    ax.set_axis_off()

# График 2: Узел при pi/4
ax2 = fig.add_subplot(2, 2, 2, projection='3d')
plot_3d_knot(ax2, np.pi/4, "Скручивание π/4", '#ffaa00')

# График 3: Узел при pi/2
ax3 = fig.add_subplot(2, 2, 3, projection='3d')
plot_3d_knot(ax3, np.pi/2, "Скручивание π/2", '#00ffaa')

# График 4: Узел Нейтрона (1.0)
ax4 = fig.add_subplot(2, 2, 4, projection='3d')
plot_3d_knot(ax4, 1.0, "Нейтрон (beta=1.0)", '#ff00ff')

plt.tight_layout()
plt.subplots_adjust(top=0.9)
plt.show()
