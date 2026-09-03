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
R_torus = 2.0  # Большой радиус
a_torus = 0.8  # Малый радиус (амплитуда переплетения)

x_c = (R_torus + a_torus * np.cos(3 * t)) * np.cos(2 * t)
y_c = (R_torus + a_torus * np.cos(3 * t)) * np.sin(2 * t)
z_c = a_torus * np.sin(3 * t)
curve_points = np.vstack((x_c, y_c, z_c)).T

# =====================================================================
# 2. РАЗВЕРТЫВАНИЕ 3D-ПОЛЯ
# =====================================================================
print("2. Развертывание 3D-поля континуума (сетка высокого разрешения)...")

A_core = 0.8168      # Идеальная толщина ядра (из электрона)
B_tail = 0.08542     # Хвост вакуума (sqrt(1/137))

grid_size = 80  # Плотная сетка для красивого 3D
bound = 3.5
x_g = np.linspace(-bound, bound, grid_size)
y_g = np.linspace(-bound, bound, grid_size)
z_g = np.linspace(-bound, bound, grid_size)
X, Y, Z = np.meshgrid(x_g, y_g, z_g, indexing='ij')

grid_points = np.vstack((X.ravel(), Y.ravel(), Z.ravel())).T

# Ищем кратчайшее расстояние от каждой точки пространства до оси узла
tree = cKDTree(curve_points)
distances, _ = tree.query(grid_points)
distances = distances.reshape((grid_size, grid_size, grid_size))
distances = np.maximum(distances, 1e-12)

# ФРАКТАЛЬНЫЙ ПЕРЕНОС: Вычисляем фазу f(rho) точно как у электрона
u = (A_core / distances) * np.exp(-B_tail * distances)
f_phase = 2 * np.arctan(u)

# =====================================================================
# 3. ВИЗУАЛИЗАЦИЯ "КОЖИ" ПРОТОНА
# =====================================================================
print("3. Рендеринг поверхности узла...")

# Строим поверхность там, где фаза равна 1.2 радиана (это физическая граница ядра)
threshold_phase = 1.2 
verts, faces, normals, values = marching_cubes(f_phase, level=threshold_phase, 
                                               spacing=(bound*2/grid_size, bound*2/grid_size, bound*2/grid_size))

# Центрируем координаты после marching_cubes
verts = verts - bound

fig = plt.figure(figsize=(12, 12))
plt.style.use('dark_background')
ax = fig.add_subplot(111, projection='3d')

# Рендерим 3D-сетку (Кибер-физический стиль)
mesh = Poly3DCollection(verts[faces], alpha=0.8)
mesh.set_facecolor('magenta')    # Тело ядра
mesh.set_edgecolor('black')      # Подчеркиваем полигоны для объема
mesh.set_linewidth(0.5)
ax.add_collection3d(mesh)

# Рисуем осевую линию (невидимый струнный каркас)
ax.plot(x_c, y_c, z_c, color='cyan', lw=2, linestyle='--', alpha=0.8, label='Ось фазового вихря (Writhe $\\approx$ 3)')

ax.set_xlim(-bound, bound)
ax.set_ylim(-bound, bound)
ax.set_zlim(-bound, bound)
ax.set_title("ПРОТОН\nФрактальная топология Узла-Трилистника ($Q=3$)", fontsize=16, pad=10, color='white')

ax.set_axis_off() # Выключаем скучные оси
ax.view_init(elev=45, azim=60) # Идеальный ракурс

plt.tight_layout()
plt.show()
print("--- ЗАВЕРШЕНО ---")
