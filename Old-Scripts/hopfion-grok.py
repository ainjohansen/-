import numpy as np
import time
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

print("="*80)
print(" ТОПОЛОГИЧЕСКАЯ МОДЕЛЬ: 3D ХОПФИОН Q=1 (полное поле)")
print("="*80)

# =====================================================================
# ПАРАМЕТРЫ
# =====================================================================
grid_input = input("Размер сетки N (рекомендуется 128-256 для начала): ")
N = int(grid_input) if grid_input.strip().isdigit() else 192
bound = 8.0
cache_file = f"cache_hopfion_Q1_{N}.npz"

alpha = 0.00729735
A_core = 0.8168          # из твоего BVP
B_tail = np.sqrt(alpha)

# =====================================================================
# 1. СОЗДАНИЕ 3D-СЕТКИ
# =====================================================================
print(f"Создаём сетку {N}³ ...")
x = np.linspace(-bound, bound, N)
X, Y, Z = np.meshgrid(x, x, x, indexing='ij')
r2 = X**2 + Y**2 + Z**2 + 1e-12

# =====================================================================
# 2. СТАНДАРТНЫЙ АНЗАЦ ХОПФИОНА Q=1 (Hopf map)
# =====================================================================
# Это классический стереографический проекционный хопфион с зарядом Q=1
D = 1.0 + r2
nx = 2 * (X*Z + Y) / D
ny = 2 * (Y*Z - X) / D
nz = (1.0 - r2) / D

# Нормируем на S²
norm = np.sqrt(nx**2 + ny**2 + nz**2)
n = np.stack([nx/norm, ny/norm, nz/norm], axis=-1)   # shape (N,N,N,3)

print("Хопфион построен (Q=1)")

# =====================================================================
# 3. ВЫЧИСЛЕНИЕ ЭНЕРГЕТИЧЕСКИХ ИНТЕГРАЛОВ (полный 3D)
# =====================================================================
if os.path.exists(cache_file):
    print("Загружаем кэш...")
    data = np.load(cache_file)
    I2 = float(data['I2'])
    I4 = float(data['I4'])
    I0 = float(data['I0'])
else:
    print("Вычисляем градиенты и энергию (может занять 30-90 сек)...")
    start = time.time()

    # Численные производные
    dx = 2 * bound / (N - 1)
    dn_dx = np.gradient(n, dx, axis=(0,1,2))          # (3, N,N,N,3)

    # |∇n|²
    grad_sq = np.sum(dn_dx**2, axis=(0,4))            # shape (N,N,N)

    # n · (∇n × ∇n)
    cross = np.cross(dn_dx[0], dn_dx[1], axis=0) + \
            np.cross(dn_dx[1], dn_dx[2], axis=0) + \
            np.cross(dn_dx[2], dn_dx[0], axis=0)
    skyrme = np.sum(n * cross, axis=-1)**2

    I2_raw = grad_sq
    I4_raw = skyrme
    I0_raw = (1.0 - n[..., 2])

    I2 = np.sum(I2_raw) * dx**3
    I4 = np.sum(I4_raw) * dx**3
    I0 = np.sum(I0_raw) * dx**3

    print(f"Время расчёта: {time.time()-start:.1f} сек")
    np.savez_compressed(cache_file, I2=I2, I4=I4, I0=I0, n=n)

print(f"\nИНТЕГРАЛЫ ХОПФИОНА Q=1 (N={N})")
print(f"I₂ (упругость)      : {I2:10.4f}")
print(f"I₄ (жесткость)     : {I4:10.4f}")
print(f"I₀ (масса хвоста)  : {I0:10.4f}")
print(f"α из баланса       : {(I4 - I2)/(3*I0):.8f}   (ожидаемо ~0.007297)")

# =====================================================================
# 4. ВИЗУАЛИЗАЦИЯ (показываем "рыхлый бублик")
# =====================================================================
plt.style.use('dark_background')
fig = plt.figure(figsize=(16, 8))

# Срез энергии в плоскости XY (z=0)
energy_density = I2_raw + I4_raw + 3*alpha*I0_raw

ax1 = fig.add_subplot(1, 2, 1)
im = ax1.imshow(energy_density[N//2, :, :].T, origin='lower',
                extent=[-bound, bound, -bound, bound],
                cmap='magma', norm=LogNorm())
ax1.set_title('Плотность энергии (срез z=0)\nТороидальный "бублик" + минимум в центре')
ax1.set_xlabel('x')
ax1.set_ylabel('y')
plt.colorbar(im, ax=ax1, label='Плотность энергии')

# 3D-изоповерхность (энергия > 30% от max)
ax2 = fig.add_subplot(1, 2, 2, projection='3d')
ax2.set_facecolor('black')
threshold = 0.3 * energy_density.max()
Xg, Yg, Zg = np.meshgrid(x, x, x, indexing='ij')
ax2.scatter(Xg[energy_density > threshold],
            Yg[energy_density > threshold],
            Zg[energy_density > threshold],
            c='cyan', s=1, alpha=0.3)
ax2.set_title('Изоповерхность энергии (30% от максимума)\nРыхлый тор + минимум в центре')
ax2.set_xlabel('x'); ax2.set_ylabel('y'); ax2.set_zlabel('z')

plt.suptitle(f'3D Хопфион Q=1  (сетка {N}³)', fontsize=16, color='white')
plt.tight_layout()
plt.show()

print("\n✅ Хопфион готов и сохранён в", cache_file)
print("   Теперь можно использовать массив 'n' для сборки атома водорода или нейтрона.")
