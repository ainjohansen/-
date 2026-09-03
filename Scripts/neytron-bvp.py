import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.spatial import cKDTree
from skimage.measure import marching_cubes

print("--- ТОПОЛОГИЧЕСКОЕ 3D-МОДЕЛИРОВАНИЕ НЕЙТРОНА ---")
print("1. Формирование базового каркаса (Трилистник)...")

t = np.linspace(0, 2 * np.pi, 2500)
R_torus = 2.0  
a_torus = 0.8  

# Гладкий протон
x_p = (R_torus + a_torus * np.cos(3 * t)) * np.cos(2 * t)
y_p = (R_torus + a_torus * np.cos(3 * t)) * np.sin(2 * t)
z_p = a_torus * np.sin(3 * t)

print("2. Вплетение Хопфиона (Локальная топологическая фрустрация)...")
# Создаем дефект на одном из лепестков (t = pi)
# Gaussian envelope локализует дефект только в одной зоне
defect_center = np.pi
envelope = np.exp(-((t - defect_center) / 0.3)**2)

# Хопфион вплетается как высокочастотная винтовая скрутка (кинк) поверх основной трубки
twist_freq = 12
twist_amplitude = 0.45

# Смещаем координаты оси, добавляя локальную пружину (Twist)
x_n = x_p + twist_amplitude * envelope * np.cos(twist_freq * t)
y_n = y_p + twist_amplitude * envelope * np.sin(twist_freq * t)
z_n = z_p + twist_amplitude * envelope * np.cos(twist_freq * t + np.pi/2)

curve_points = np.vstack((x_n, y_n, z_n)).T

print("3. Развертывание 3D-поля континуума...")
A_core = 0.8168      # Идеальная толщина ядра 
B_tail = 0.08542     # Хвост вакуума 

grid_size = 80  
bound = 4.0
x_g = np.linspace(-bound, bound, grid_size)
y_g = np.linspace(-bound, bound, grid_size)
z_g = np.linspace(-bound, bound, grid_size)
X, Y, Z = np.meshgrid(x_g, y_g, z_g, indexing='ij')

grid_points = np.vstack((X.ravel(), Y.ravel(), Z.ravel())).T

# Ищем расстояния до фрустрированной оси нейтрона
tree = cKDTree(curve_points)
distances, _ = tree.query(grid_points)
distances = distances.reshape((grid_size, grid_size, grid_size))
distances = np.maximum(distances, 1e-12)

# Фазовое поле
u = (A_core / distances) * np.exp(-B_tail * distances)
f_phase = 2 * np.arctan(u)

print("4. Рендеринг поверхности Нейтрона...")
threshold_phase = 1.2 
verts, faces, normals, values = marching_cubes(f_phase, level=threshold_phase, 
                                               spacing=(bound*2/grid_size, bound*2/grid_size, bound*2/grid_size))

verts = verts - bound

fig = plt.figure(figsize=(12, 12))
plt.style.use('dark_background')
ax = fig.add_subplot(111, projection='3d')

# Рисуем нейтрон. Делаем его цвет слегка нестабильным (оранжево-красным)
mesh = Poly3DCollection(verts[faces], alpha=0.85)
mesh.set_facecolor('coral')    # Цвет напряженной энергии
mesh.set_edgecolor('black')    # Полигоны для глубины
mesh.set_linewidth(0.4)
ax.add_collection3d(mesh)

# Рисуем ось (чтобы было видно, как хопфион запутался внутри)
ax.plot(x_n, y_n, z_n, color='yellow', lw=2, linestyle='-', alpha=0.9, label='Фрустрированная ось (Трилистник + Хопфион)')

ax.set_xlim(-bound, bound)
ax.set_ylim(-bound, bound)
ax.set_zlim(-bound, bound)
ax.set_title("НЕЙТРОН\nНарушенная симметрия $\\mathbb{Z}_3$ (Вплетенный дефект Хопфа)", fontsize=16, pad=10, color='white')

ax.set_axis_off() 
ax.view_init(elev=35, azim=110) # Ракурс прямо на дефект

plt.legend(loc='upper right', facecolor='black', edgecolor='white', labelcolor='white')
plt.tight_layout()
plt.show()
print("--- ЗАВЕРШЕНО ---")
