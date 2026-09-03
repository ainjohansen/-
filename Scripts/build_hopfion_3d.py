import torch
import numpy as np
import time
import os
import threading
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
try:
    from skimage.measure import marching_cubes
    HAS_SKIMAGE = True
except ImportError:
    HAS_SKIMAGE = False
    print("[-] Установите scikit-image для 3D рендера.")

print("="*80)
print(" ШАГ 1: ИДЕАЛЬНЫЙ ПОЛЫЙ ХОПФИОН (MULTI-GPU + R^2 ПРОФИЛИРОВАНИЕ)")
print("="*80)

# =====================================================================
# 1. ПАРАМЕТРЫ СЕТКИ И ПОИСК GPU
# =====================================================================
NUM_GPUS = torch.cuda.device_count() if torch.cuda.is_available() else 0
if NUM_GPUS == 0:
    print("[-] ОШИБКА: CUDA не найдена. Нужен GPU.")
    sys.exit()

grid_input = input("Размер сетки N (смело вводите 512): ")
N = int(grid_input) if grid_input.strip().isdigit() else 256

L = 8.0 
dx = 2 * L / N
dV = dx**3
scale_a = 1.5 
filename = f"hopfion_q1_grid_{N}.npz"

C2, C4, C0 = 1.0, 1.0, 0.1 

print(f"\nВыделение {N}x{N}x{N} в системной RAM...")
n_field_cpu = np.zeros((3, N, N, N), dtype=np.float32)

# =====================================================================
# 2. ПОСТРОЕНИЕ ПОЛЯ ХОПФА (ПАРАЛЛЕЛЬНО НА ВСЕХ GPU)
# =====================================================================
start_time = time.time()

def worker_build_field(gpu_id, z_start, z_end):
    device = torch.device(f'cuda:{gpu_id}')
    x_ts = torch.linspace(-L, L, N, device=device, dtype=torch.float32)
    X, Y = torch.meshgrid(x_ts, x_ts, indexing='ij')
    
    xs, ys = X / scale_a, Y / scale_a
    
    for z_idx in range(z_start, z_end):
        z_val = -L + z_idx * dx
        zs = torch.full_like(X, z_val / scale_a)
        
        R2 = xs**2 + ys**2 + zs**2
        R = torch.sqrt(R2 + 1e-12)
        
        # ТОПОЛОГИЧЕСКОЕ ПРОФИЛИРОВАНИЕ: g(R) = R^2
        # Гарантирует E = 0 в центре и полый бублик, сохраняя Q=1
        g_R2 = R2**2 
        denom = 1.0 + g_R2
        
        X1 = 2 * xs * R / denom
        X2 = 2 * ys * R / denom
        X3 = 2 * zs * R / denom
        X4 = (1.0 - g_R2) / denom
        
        n1 = 2 * (X1 * X4 + X2 * X3)
        n2 = 2 * (X2 * X4 - X1 * X3)
        n3 = X4**2 + X3**2 - X1**2 - X2**2
        
        n_field_cpu[0, :, :, z_idx] = n1.cpu().numpy()
        n_field_cpu[1, :, :, z_idx] = n2.cpu().numpy()
        n_field_cpu[2, :, :, z_idx] = n3.cpu().numpy()

print("Генерация топологического поля...")
threads =[]
chunk_size_z = N // NUM_GPUS
for i in range(NUM_GPUS):
    z_s = i * chunk_size_z
    z_e = N if i == NUM_GPUS - 1 else (i + 1) * chunk_size_z
    t = threading.Thread(target=worker_build_field, args=(i, z_s, z_e))
    t.start()
    threads.append(t)
for t in threads: t.join()

# =====================================================================
# 3. ВЫЧИСЛЕНИЕ ЭНЕРГИИ (СЛОЙ ЗА СЛОЕМ ДЛЯ ЗАЩИТЫ VRAM)
# =====================================================================
print("Вычисление функционалов энергии...")

total_I2, total_I4, total_I0 = 0.0, 0.0, 0.0
lock = threading.Lock()

def worker_calc_energy(gpu_id, z_start, z_end):
    global total_I2, total_I4, total_I0
    device = torch.device(f'cuda:{gpu_id}')
    loc_I2, loc_I4, loc_I0 = 0.0, 0.0, 0.0
    
    BATCH = 16
    for b_start in range(z_start, z_end, BATCH):
        b_end = min(z_end, b_start + BATCH)
        pad_s = max(0, b_start - 1)
        pad_e = min(N, b_end + 1)
        
        chunk = torch.from_numpy(n_field_cpu[:, :, :, pad_s:pad_e]).to(device)
        
        dn_dx = torch.zeros_like(chunk)
        dn_dy = torch.zeros_like(chunk)
        dn_dz = torch.zeros_like(chunk)
        
        dn_dx[:, 1:-1, :, :] = (chunk[:, 2:, :, :] - chunk[:, :-2, :, :]) / (2 * dx)
        dn_dy[:, :, 1:-1, :] = (chunk[:, :, 2:, :] - chunk[:, :, :-2, :]) / (2 * dx)
        dn_dz[:, :, :, 1:-1] = (chunk[:, :, :, 2:] - chunk[:, :, :, :-2]) / (2 * dx)
        
        valid_s = 1 if b_start > 0 else 0
        valid_e = chunk.shape[3] - 1 if b_end < N else chunk.shape[3]
        
        chunk_v = chunk[:, :, :, valid_s:valid_e]
        dx_v = dn_dx[:, :, :, valid_s:valid_e]
        dy_v = dn_dy[:, :, :, valid_s:valid_e]
        dz_v = dn_dz[:, :, :, valid_s:valid_e]
        
        i2_dens = torch.sum(dx_v**2 + dy_v**2 + dz_v**2, dim=0)
        c_xy = torch.cross(dx_v, dy_v, dim=0)
        c_yz = torch.cross(dy_v, dz_v, dim=0)
        c_zx = torch.cross(dz_v, dx_v, dim=0)
        i4_dens = torch.sum(c_xy**2 + c_yz**2 + c_zx**2, dim=0)
        i0_dens = 1.0 - chunk_v[2]
        
        loc_I2 += torch.sum(i2_dens.to(torch.float64)).item() * dV
        loc_I4 += torch.sum(i4_dens.to(torch.float64)).item() * dV
        loc_I0 += torch.sum(i0_dens.to(torch.float64)).item() * dV

    with lock:
        total_I2 += loc_I2
        total_I4 += loc_I4
        total_I0 += loc_I0

threads =[]
for i in range(NUM_GPUS):
    z_s = i * chunk_size_z
    z_e = N if i == NUM_GPUS - 1 else (i + 1) * chunk_size_z
    t = threading.Thread(target=worker_calc_energy, args=(i, z_s, z_e))
    t.start()
    threads.append(t)
for t in threads: t.join()

E_tot = C2 * total_I2 + C4 * total_I4 + C0 * total_I0

# =====================================================================
# 4. ИЗВЛЕЧЕНИЕ ПЛОТНОСТИ ЭНЕРГИИ ДЛЯ ГРАФИКА
# =====================================================================
print("Извлечение точной плотности энергии экватора...")
mid = N // 2
device_0 = torch.device('cuda:0')
slice_chunk = torch.from_numpy(n_field_cpu[:, :, :, mid-1:mid+2]).to(device_0)

dx_v = torch.zeros((3, N, N), device=device_0)
dy_v = torch.zeros((3, N, N), device=device_0)
dz_v = (slice_chunk[:, :, :, 2] - slice_chunk[:, :, :, 0]) / (2 * dx)

dx_v[:, 1:-1, :] = (slice_chunk[:, 2:, :, 1] - slice_chunk[:, :-2, :, 1]) / (2 * dx)
dy_v[:, :, 1:-1] = (slice_chunk[:, :, 2:, 1] - slice_chunk[:, :, :-2, 1]) / (2 * dx)

i2 = torch.sum(dx_v**2 + dy_v**2 + dz_v**2, dim=0)
c_xy = torch.cross(dx_v, dy_v, dim=0)
c_yz = torch.cross(dy_v, dz_v, dim=0)
c_zx = torch.cross(dz_v, dx_v, dim=0)
i4 = torch.sum(c_xy**2 + c_yz**2 + c_zx**2, dim=0)
i0 = 1.0 - slice_chunk[2, :, :, 1]

E_slice = C2 * i2 + C4 * i4 + C0 * i0
E_slice_cpu = E_slice.cpu().numpy()

calc_time = time.time() - start_time

print("\n" + "="*60)
print(" ТОПОЛОГИЧЕСКИЕ ИНТЕГРАЛЫ ХОПФИОНА (Q=1)")
print("="*60)
print(f"Интеграл Дирихле (I2) : {total_I2:.6f}")
print(f"Интеграл Скирма  (I4) : {total_I4:.6f}")
print(f"Массовый член    (I0) : {total_I0:.6f}")
print("-" * 60)
print(f"ПОЛНАЯ ЭНЕРГИЯ        : {E_tot:.6f}")
print(f"Время вычислений (2xGPU): {calc_time:.3f} сек")
print("="*60)

np.savez_compressed(filename, n_field=n_field_cpu, dx=dx, L=L, scale=scale_a)

# =====================================================================
# 5. ВИЗУАЛИЗАЦИЯ ИСТИННОЙ ЭНЕРГИИ
# =====================================================================
print("\nРендеринг (закройте окно для выхода)...")

plt.style.use('dark_background')
fig = plt.figure(figsize=(16, 8), facecolor='black')
fig.suptitle(f'Идеальный Хопфион (Сетка {N}^3)', fontsize=18, fontweight='bold', color='white')

# График 1: Истинная Плотность Энергии (Срез)
ax1 = fig.add_subplot(1, 2, 1)
ax1.set_facecolor('black')

# Обрезаем 1% самых ярких пиков, чтобы кольцо светилось равномерно
vmax = np.percentile(E_slice_cpu, 99.5)
img = ax1.imshow(E_slice_cpu.T, extent=[-L, L, -L, L], origin='lower', cmap='inferno', vmax=vmax)
ax1.set_title("Плотность Энергии (Срез Z=0)", fontsize=14, color='white')
ax1.set_xlabel("X"); ax1.set_ylabel("Y")
fig.colorbar(img, ax=ax1, fraction=0.046, pad=0.04).ax.yaxis.set_tick_params(color='white')

# Векторы фазы
step = max(1, N // 30)
x_coords = np.linspace(-L, L, N)
X_m, Y_m = np.meshgrid(x_coords, x_coords, indexing='ij')
ax1.quiver(X_m[::step, ::step], Y_m[::step, ::step], 
           n_field_cpu[0, ::step, ::step, mid], n_field_cpu[1, ::step, ::step, mid], 
           color='#00ffff', alpha=0.6, scale=25)

# Подпись в центре
ax1.text(0, 0, "E = 0\n(Вакуум)", color='white', ha='center', va='center', fontweight='bold')

# График 2: 3D Бублик
if HAS_SKIMAGE:
    ax2 = fig.add_subplot(1, 2, 2, projection='3d')
    ax2.set_facecolor('black')
    verts, faces, normals, values = marching_cubes(n_field_cpu[2], level=-0.5)
    verts_scaled = verts * dx - L
    mesh = Poly3DCollection(verts_scaled[faces], alpha=0.8, facecolor='#ff00ff', edgecolor='#550055', linewidth=0.5)
    ax2.add_collection3d(mesh)
    
    ax2.set_xlim(-L/2, L/2); ax2.set_ylim(-L/2, L/2); ax2.set_zlim(-L/2, L/2)
    ax2.set_title("Ядро бублика (Стереографическая проекция S^3 -> S^2)", fontsize=14, color='white')
    ax2.set_axis_off()
else:
    ax2 = fig.add_subplot(1, 2, 2)
    ax2.axis('off')

plt.tight_layout()
plt.show()
