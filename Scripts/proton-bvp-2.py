import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.spatial import cKDTree
from skimage.measure import marching_cubes

print("--- ТОПОЛОГИЧЕСКОЕ 3D-МОДЕЛИРОВАНИЕ ПРОТОНА ---")
print("1. Генерация оси узла-трилистника (Q=3)...")

# =====================================================================
# 1. ТОПОЛОГИЯ ТРИЛИСТНИКА (Каркас Протона)
# =====================================================================
t = np.linspace(0, 2 * np.pi, 2000)
R_torus = 2.0
a_torus = 0.8

x_c = (R_torus + a_torus * np.cos(3 * t)) * np.cos(2 * t)
y_c = (R_torus + a_torus * np.cos(3 * t)) * np.sin(2 * t)
z_c = a_torus * np.sin(3 * t)
curve_points = np.vstack((x_c, y_c, z_c)).T

# =====================================================================
# 2. РАЗВЕРТЫВАНИЕ 3D-ПОЛЯ
# =====================================================================
print("2. Развертывание 3D-поля континуума (сетка высокого разрешения)...")

A_core = 0.8168
B_tail = 0.08542

grid_size = 80
bound = 3.5
x_g = np.linspace(-bound, bound, grid_size)
y_g = np.linspace(-bound, bound, grid_size)
z_g = np.linspace(-bound, bound, grid_size)
X, Y, Z = np.meshgrid(x_g, y_g, z_g, indexing='ij')
grid_points = np.vstack((X.ravel(), Y.ravel(), Z.ravel())).T

tree = cKDTree(curve_points)
distances, _ = tree.query(grid_points)
distances = distances.reshape((grid_size, grid_size, grid_size))
distances = np.maximum(distances, 1e-12)

u = (A_core / distances) * np.exp(-B_tail * distances)
f_phase = 2 * np.arctan(u)

# =====================================================================
# 3. ВИЗУАЛИЗАЦИЯ "КОЖИ" ПРОТОНА
# =====================================================================
print("3. Рендеринг поверхности узла...")
threshold_phase = 1.2
verts, faces, normals, values = marching_cubes(f_phase, level=threshold_phase,
                                               spacing=(bound*2/grid_size, bound*2/grid_size, bound*2/grid_size))
verts = verts - bound

fig = plt.figure(figsize=(12, 12))
plt.style.use('dark_background')
ax = fig.add_subplot(111, projection='3d')
mesh = Poly3DCollection(verts[faces], alpha=0.8)
mesh.set_facecolor('magenta')
mesh.set_edgecolor('black')
mesh.set_linewidth(0.5)
ax.add_collection3d(mesh)
ax.plot(x_c, y_c, z_c, color='cyan', lw=2, linestyle='--', alpha=0.8, label='Ось фазового вихря')
ax.set_xlim(-bound, bound)
ax.set_ylim(-bound, bound)
ax.set_zlim(-bound, bound)
ax.set_title("ПРОТОН\nТрилистник (Q=3)", fontsize=16, pad=10, color='white')
ax.set_axis_off()
ax.view_init(elev=45, azim=60)
plt.tight_layout()
plt.show()

# =====================================================================
# 4. ПОСТРОЕНИЕ ПРОФИЛЕЙ
# =====================================================================
print("4. Построение профиля f(ρ) и плотностей энергии...")

dist_flat = distances.ravel()
f_flat = f_phase.ravel()

# Сортируем по расстоянию
idx = np.argsort(dist_flat)
dist_sorted = dist_flat[idx]
f_sorted = f_flat[idx]

# Биннинг
bins = np.linspace(0, bound, 200)
bin_centers = (bins[:-1] + bins[1:]) / 2
f_binned = np.zeros(len(bin_centers))
for i in range(len(bin_centers)):
    mask = (dist_sorted >= bins[i]) & (dist_sorted < bins[i+1])
    if np.sum(mask) > 0:
        f_binned[i] = np.mean(f_sorted[mask])
    else:
        f_binned[i] = np.nan

valid = ~np.isnan(f_binned)
rho_bin = bin_centers[valid]
f_bin = f_binned[valid]

# Производная
df_drho = np.gradient(f_bin, rho_bin)

# Плотности энергии (без учёта множителя 4π, который сокращается в α)
rho2 = rho_bin**2
sin_f = np.sin(f_bin)
cos_f = np.cos(f_bin)

dens2 = (df_drho**2 + 2 * sin_f**2 / rho_bin**2) * rho2
term = sin_f**2 / rho_bin**2
dens4 = term * (2 * df_drho**2 + term) * rho2
dens0 = (1 - cos_f) * rho2

# Интегрирование
I2 = np.trapezoid(dens2, rho_bin)
I4 = np.trapezoid(dens4, rho_bin)
I0 = np.trapezoid(dens0, rho_bin)
alpha_calc = (I4 - I2) / (3 * I0)

print(f"\nИнтегралы из 3D-модели (приближенные):")
print(f"I2 = {I2:.4f}")
print(f"I4 = {I4:.4f}")
print(f"I0 = {I0:.4f}")
print(f"α  = {alpha_calc:.10f}")

# =====================================================================
# 5. ГРАФИКИ
# =====================================================================
plt.style.use('default')
fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

ax1.plot(rho_bin, f_bin, 'k-', linewidth=2)
ax1.axhline(np.pi, color='gray', linestyle='--', alpha=0.7, label=r'$\pi$')
ax1.axhline(0, color='gray', linestyle='--', alpha=0.7)
ax1.axvline(1, color='red', linestyle=':', alpha=0.7, label=r'$\rho=1$')
ax1.set_xlabel(r'$\rho = r/a$')
ax1.set_ylabel(r'$f(\rho)$')
ax1.set_title('Профиль поля протона (трилистник)')
ax1.legend()
ax1.grid(True, alpha=0.3)

ax2.plot(rho_bin, dens2, 'b-', label='Упругость')
ax2.plot(rho_bin, dens4, 'k-', label='Жёсткость')
ax2.plot(rho_bin, dens0, 'r-', label='Масса')
ax2.set_xlabel(r'$\rho = r/a$')
ax2.set_ylabel('Плотность энергии')
ax2.set_title('Распределение энергии в протоне')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# =====================================================================
# 6. СРАВНЕНИЕ С ЭЛЕКТРОНОМ
# =====================================================================
def electron_profile(rho):
    A = 0.8168
    B = 0.08542
    u = (A / rho) * np.exp(-B * rho)
    return 2 * np.arctan(u)

rho_plot = np.linspace(1e-4, 6, 500)
f_e = electron_profile(rho_plot)
f_p_interp = np.interp(rho_plot, rho_bin, f_bin, left=np.pi, right=0)

fig3, ax = plt.subplots(figsize=(8, 5))
ax.plot(rho_plot, f_e, 'b-', label='Электрон (хопфион)', linewidth=2)
ax.plot(rho_plot, f_p_interp, 'r-', label='Протон (трилистник)', linewidth=2)
ax.axhline(np.pi, color='gray', linestyle='--', alpha=0.7)
ax.axhline(0, color='gray', linestyle='--', alpha=0.7)
ax.set_xlabel(r'$\rho = r/a$')
ax.set_ylabel(r'$f(\rho)$')
ax.set_title('Сравнение профилей электрона и протона')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

print("--- ЗАВЕРШЕНО ---")
