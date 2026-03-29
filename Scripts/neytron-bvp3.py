import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.spatial import cKDTree
from skimage.measure import marching_cubes
from scipy.interpolate import interp1d
from scipy.integrate import solve_bvp, quad

# =====================================================================
# 1. Сначала получим точный профиль f(r) для протона (как в proton-bvp-3.py)
# =====================================================================
alpha_exp = 0.0072973525693

def ode_system_proton(r, y):
    f, fp = y
    sin_f = np.sin(f)
    cos_f = np.cos(f)
    sin2 = sin_f**2
    denom = r**2 + 2 * sin2
    term1 = 2 * sin_f * cos_f * (1 - fp**2 + sin2/r**2)
    term2 = (alpha_exp / 2) * r**2 * sin_f
    term3 = -2 * r * fp
    fpp = (term1 + term2 + term3) / denom
    return np.vstack((fp, fpp))

def boundary_conditions(ya, yb):
    return np.array([ya[0] - np.pi, yb[0] - 0.0])

# Решаем BVP для протона
r_min = 1e-4
r_max = 120.0
r_nodes = np.logspace(np.log10(r_min), np.log10(r_max), 1000)
A_core = 0.8168
B_tail = 0.08542
u_guess = (A_core / r_nodes) * np.exp(-B_tail * r_nodes)
f_guess = 2 * np.arctan(u_guess)
fp_guess = (2 * u_guess / (1 + u_guess**2)) * (-1/r_nodes - B_tail)
y_guess = np.vstack((f_guess, fp_guess))

sol_proton = solve_bvp(ode_system_proton, boundary_conditions, r_nodes, y_guess, tol=1e-8, max_nodes=50000)
if not sol_proton.success:
    raise RuntimeError("BVP для протона не сошёлся")

# Создаём интерполятор f(r) для быстрого вычисления в 3D
r_vals = np.logspace(np.log10(r_min), np.log10(20.0), 2000)
f_vals = sol_proton.sol(r_vals)[0]
f_interp = interp1d(r_vals, f_vals, kind='cubic', bounds_error=False, fill_value=0.0)

# =====================================================================
# 2. Генерация оси нейтрона (трилистник с локальной полускруткой)
# =====================================================================
print("Генерация оси нейтрона с полускруткой...")
t = np.linspace(0, 2*np.pi, 2500)
R_torus = 2.0
a_torus = 0.8

# Базовый трилистник (протон)
x_p = (R_torus + a_torus * np.cos(3*t)) * np.cos(2*t)
y_p = (R_torus + a_torus * np.cos(3*t)) * np.sin(2*t)
z_p = a_torus * np.sin(3*t)

# Локальная полускрутка в районе t = pi
defect_center = np.pi
envelope = np.exp(-((t - defect_center) / 0.3)**2)
twist_freq = 12
twist_amplitude = 0.45
x_n = x_p + twist_amplitude * envelope * np.cos(twist_freq * t)
y_n = y_p + twist_amplitude * envelope * np.sin(twist_freq * t)
z_n = z_p + twist_amplitude * envelope * np.cos(twist_freq * t + np.pi/2)

curve_points = np.vstack((x_n, y_n, z_n)).T

# =====================================================================
# 3. Построение 3D поля f(x,y,z) = f_interp(расстояние до оси)
# =====================================================================
print("Построение 3D сетки...")
grid_size = 80
bound = 4.0
x_g = np.linspace(-bound, bound, grid_size)
y_g = np.linspace(-bound, bound, grid_size)
z_g = np.linspace(-bound, bound, grid_size)
X, Y, Z = np.meshgrid(x_g, y_g, z_g, indexing='ij')
grid_points = np.vstack((X.ravel(), Y.ravel(), Z.ravel())).T

tree = cKDTree(curve_points)
distances, _ = tree.query(grid_points)
distances = distances.reshape((grid_size, grid_size, grid_size))
distances = np.maximum(distances, 1e-8)

# Вычисляем f(r) через интерполятор
f_phase = f_interp(distances)

# =====================================================================
# 4. Визуализация изоповерхности
# =====================================================================
print("Рендеринг поверхности...")
threshold = 1.2   # уровень f, при котором строим поверхность
verts, faces, normals, values = marching_cubes(f_phase, level=threshold,
                                               spacing=(bound*2/grid_size, bound*2/grid_size, bound*2/grid_size))
verts = verts - bound

fig = plt.figure(figsize=(12, 12))
plt.style.use('dark_background')
ax = fig.add_subplot(111, projection='3d')
mesh = Poly3DCollection(verts[faces], alpha=0.85)
mesh.set_facecolor('coral')
mesh.set_edgecolor('black')
mesh.set_linewidth(0.4)
ax.add_collection3d(mesh)

# Ось нейтрона (для наглядности)
ax.plot(x_n, y_n, z_n, color='yellow', lw=1.5, alpha=0.9, label='Фрустрированная ось (полускрутка)')

ax.set_xlim(-bound, bound)
ax.set_ylim(-bound, bound)
ax.set_zlim(-bound, bound)
ax.set_title("НЕЙТРОН\nТрилистник с локальной полускруткой (Tw=1/2)", fontsize=16, color='white')
ax.set_axis_off()
ax.view_init(elev=35, azim=110)
plt.legend(loc='upper right', facecolor='black', edgecolor='white', labelcolor='white')
plt.tight_layout()
plt.savefig("neutron_3d.png", dpi=150)
plt.show()

print("Готово. Визуализация сохранена в neutron_3d.png")
