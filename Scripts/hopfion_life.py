import torch
import numpy as np
import time
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
try:
    from skimage.measure import marching_cubes
    HAS_SKIMAGE = True
except ImportError:
    HAS_SKIMAGE = False

print("="*80)
print(" ЖИЗНЬ ЭЛЕКТРОНА: ДЫХАНИЕ ХОПФИОНА (ZITTERBEWEGUNG)")
print("="*80)

# =====================================================================
# 1. ПАРАМЕТРЫ И ИНИЦИАЛИЗАЦИЯ
# =====================================================================
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
N = 120  # Небольшая сетка для быстрого сканирования
L = 8.0 
dx = 2 * L / N
dV = dx**3

C2, C4, C0 = 1.0, 1.0, 0.1 

# Функция генерации поля Хопфиона с заданным масштабом (радиусом)
def generate_hopfion_energy(scale):
    x_ts = torch.linspace(-L, L, N, device=device, dtype=torch.float32)
    Y, X, Z = torch.meshgrid(x_ts, x_ts, x_ts, indexing='ij')
    
    xs, ys, zs = X / scale, Y / scale, Z / scale
    R2 = xs**2 + ys**2 + zs**2
    R = torch.sqrt(R2 + 1e-12)
    
    g_R2 = R2**2 
    denom = 1.0 + g_R2
    
    X1 = 2 * xs * R / denom
    X2 = 2 * ys * R / denom
    X3 = 2 * zs * R / denom
    X4 = (1.0 - g_R2) / denom
    
    n_field = torch.stack([
        2 * (X1 * X4 + X2 * X3),
        2 * (X2 * X4 - X1 * X3),
        X4**2 + X3**2 - X1**2 - X2**2
    ], dim=0)
    
    # Градиенты
    dn_dx = torch.zeros_like(n_field)
    dn_dy = torch.zeros_like(n_field)
    dn_dz = torch.zeros_like(n_field)
    
    dn_dx[:, 1:-1, :, :] = (n_field[:, 2:, :, :] - n_field[:, :-2, :, :]) / (2 * dx)
    dn_dy[:, :, 1:-1, :] = (n_field[:, :, 2:, :] - n_field[:, :, :-2, :]) / (2 * dx)
    dn_dz[:, :, :, 1:-1] = (n_field[:, :, :, 2:] - n_field[:, :, :, :-2]) / (2 * dx)
    
    # Энергии
    i2 = torch.sum(dn_dx**2 + dn_dy**2 + dn_dz**2, dim=0)
    c_xy = torch.cross(dn_dx, dn_dy, dim=0)
    c_yz = torch.cross(dn_dy, dn_dz, dim=0)
    c_zx = torch.cross(dn_dz, dn_dx, dim=0)
    i4 = torch.sum(c_xy**2 + c_yz**2 + c_zx**2, dim=0)
    i0 = 1.0 - n_field[2]
    
    E_tot = (C2 * torch.sum(i2) + C4 * torch.sum(i4) + C0 * torch.sum(i0)) * dV
    
    return E_tot.item(), n_field[2].cpu().numpy()

# =====================================================================
# 2. СКАНИРОВАНИЕ ДИАПАЗОНА УСТОЙЧИВОСТИ (ЭНЕРГЕТИЧЕСКАЯ ЯМА)
# =====================================================================
print("Поиск диапазона баланса (Энергетической ямы)...")
scales = np.linspace(0.8, 3.5, 40)
energies =[]

for s in scales:
    e, _ = generate_hopfion_energy(s)
    energies.append(e)
    
energies = np.array(energies)

# Находим точку абсолютного баланса
min_idx = np.argmin(energies)
optimal_scale = scales[min_idx]
min_energy = energies[min_idx]

print(f"Идеальный масштаб: {optimal_scale:.2f} | Энергия покоя: {min_energy:.2f}")

# Генерируем 3 фазы "дыхания" для 3D рендера
_, field_compressed = generate_hopfion_energy(optimal_scale * 0.6) # Сжат
_, field_optimal    = generate_hopfion_energy(optimal_scale)       # Идеал
_, field_expanded   = generate_hopfion_energy(optimal_scale * 1.5) # Раздут

# =====================================================================
# 3. ВИЗУАЛИЗАЦИЯ ЖИЗНИ ЭЛЕКТРОНА
# =====================================================================
print("\nРендеринг Жизни Электрона...")

plt.style.use('dark_background')
fig = plt.figure(figsize=(16, 10), facecolor='black')
fig.suptitle('Истинная жизнь Электрона (Hopfion Zitterbewegung)', fontsize=18, fontweight='bold', color='white')

# ГРАФИК 1: Энергетическая яма (Диапазон баланса)
ax1 = fig.add_subplot(2, 1, 1)
ax1.set_facecolor('black')
ax1.plot(scales, energies, color='#00ffff', lw=4, label='Полная Энергия (Баланс Скирма и Дирихле)')
ax1.axvline(optimal_scale, color='#ff00ff', linestyle='--', lw=2, label='Точка абсолютного равновесия')

# Выделяем "зону пульсации" (например +15% энергии от минимума)
threshold_energy = min_energy * 1.15
ax1.axhline(threshold_energy, color='#aaaaaa', linestyle=':', lw=2)
ax1.fill_between(scales, energies, threshold_energy, where=(energies < threshold_energy), color='#00ffff', alpha=0.2, label='Диапазон пульсации (Zitterbewegung)')

ax1.set_title("Энергетическая Яма (Диапазон Устойчивости)", color='white', fontsize=14)
ax1.set_xlabel("Масштаб (Физический размер тора)", color='white')
ax1.set_ylabel("Энергия вакуума", color='white')
ax1.set_ylim(min_energy * 0.95, min_energy * 1.5)
ax1.grid(True, color='#333333', linestyle=':')
ax1.legend(facecolor='black', edgecolor='white', labelcolor='white', loc='upper right')

# ГРАФИКИ 2, 3, 4: Три состояния "Дыхания"
titles =["Фаза Сжатия (Отскок)", "Фаза Баланса (Точка покоя)", "Фаза Расширения (Растяжение)"]
fields = [field_compressed, field_optimal, field_expanded]
colors =['#ff0000', '#ff00ff', '#0000ff'] # От красного (горячего) к синему (холодному)

for i in range(3):
    if HAS_SKIMAGE:
        ax = fig.add_subplot(2, 3, 4 + i, projection='3d')
        ax.set_facecolor('black')
        
        verts, faces, normals, values = marching_cubes(fields[i], level=-0.5)
        verts_scaled = verts * dx - L
        mesh = Poly3DCollection(verts_scaled[faces], alpha=0.7, facecolor=colors[i], edgecolor='black', linewidth=0.2)
        ax.add_collection3d(mesh)
        
        ax.set_xlim(-L/2, L/2); ax.set_ylim(-L/2, L/2); ax.set_zlim(-L/2, L/2)
        ax.set_title(titles[i], color=colors[i], fontsize=12, fontweight='bold')
        ax.set_axis_off()
    else:
        ax = fig.add_subplot(2, 3, 4 + i)
        ax.set_facecolor('black')
        mid = N // 2
        ax.imshow(fields[i][:, :, mid].T, extent=[-L, L, -L, L], origin='lower', cmap='magma')
        ax.set_title(titles[i] + "\n(Срез)", color=colors[i], fontsize=12)
        ax.axis('off')

plt.tight_layout()
plt.subplots_adjust(top=0.92)
plt.show()
