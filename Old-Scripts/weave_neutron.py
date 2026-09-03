import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection

print("="*80)
print("РОЖДЕНИЕ НЕЙТРОНА: ТРУБКА С ЦВЕТНЫМИ НИТЯМИ (ВИДИМОЕ КРУЧЕНИЕ)")
print("="*80)

# ---------------------------------------------------------------------
# 1. Трилистник T(2,3)
# ---------------------------------------------------------------------
N_curve = 300
t = np.linspace(0, 2*np.pi, N_curve)
R, a = 2.0, 0.8

x = (R + a * np.cos(3*t)) * np.cos(2*t)
y = (R + a * np.cos(3*t)) * np.sin(2*t)
z = a * np.sin(3*t)
curve = np.vstack([x, y, z]).T

# ---------------------------------------------------------------------
# 2. Репер Френе
# ---------------------------------------------------------------------
def frenet_frame(curve):
    N_pts = len(curve)
    T = np.zeros_like(curve)
    for i in range(1, N_pts-1):
        T[i] = curve[i+1] - curve[i-1]
    T[0] = curve[1] - curve[0]
    T[-1] = curve[-1] - curve[-2]
    T = T / (np.linalg.norm(T, axis=1)[:, np.newaxis] + 1e-8)

    dT = np.zeros_like(curve)
    for i in range(1, N_pts-1):
        dT[i] = T[i+1] - T[i-1]
    dT[0] = T[1] - T[0]
    dT[-1] = T[-1] - T[-2]
    dT_norm = np.linalg.norm(dT, axis=1)
    N_vec = dT / (dT_norm[:, np.newaxis] + 1e-8)
    B_vec = np.cross(T, N_vec)
    return T, N_vec, B_vec

T, N, B = frenet_frame(curve)

# ---------------------------------------------------------------------
# 3. Построение трубок и цветных нитей
# ---------------------------------------------------------------------
def make_tube_with_threads(curve, N, B, twist_phase, radius, n_threads=3):
    """
    Возвращает:
      - vertices, faces для трубки (однотонной)
      - список нитей: каждая нить = массив точек (N_curve, 3)
    twist_phase: массив длины N_curve, добавочный угол поворота сечения.
    """
    N_pts = len(curve)
    # Углы нитей в сечении (равномерно)
    thread_angles = np.linspace(0, 2*np.pi, n_threads, endpoint=False)
    threads = [[] for _ in range(n_threads)]

    # Для трубки строим полную сетку
    n_circ = 24
    theta = np.linspace(0, 2*np.pi, n_circ, endpoint=False)
    vertices = []
    for i in range(N_pts):
        phase = twist_phase[i]
        N_rot = np.cos(phase)*N[i] + np.sin(phase)*B[i]
        B_rot = -np.sin(phase)*N[i] + np.cos(phase)*B[i]
        # Точки окружности для трубки
        for th in theta:
            vert = curve[i] + radius * (np.cos(th)*N_rot + np.sin(th)*B_rot)
            vertices.append(vert)
        # Точки для нитей (те же углы, но только несколько)
        for k, ang in enumerate(thread_angles):
            point = curve[i] + radius * (np.cos(ang)*N_rot + np.sin(ang)*B_rot)
            threads[k].append(point)

    vertices = np.array(vertices)
    # Грани трубки
    faces = []
    for i in range(N_pts-1):
        for j in range(n_circ):
            i_next = i+1
            j_next = (j+1) % n_circ
            v0 = i*n_circ + j
            v1 = i*n_circ + j_next
            v2 = i_next*n_circ + j_next
            v3 = i_next*n_circ + j
            faces.append([v0, v1, v2, v3])
    # Преобразуем нити в массивы
    threads_arrays = [np.array(th) for th in threads]
    return vertices, faces, threads_arrays

radius = 0.12  # тонкая трубка

# Протон: twist_phase = 0
twist_proton = np.zeros(N_curve)
verts_p, faces_p, threads_p = make_tube_with_threads(curve, N, B, twist_proton, radius, n_threads=3)

# Нейтрон: twist_phase от 0 до 2π
twist_neutron = 2 * np.pi * (t / (2*np.pi))
verts_n, faces_n, threads_n = make_tube_with_threads(curve, N, B, twist_neutron, radius, n_threads=3)

# ---------------------------------------------------------------------
# 4. Визуализация
# ---------------------------------------------------------------------
plt.style.use('dark_background')
fig = plt.figure(figsize=(16, 8), facecolor='black')
fig.suptitle('Захват Электрона: Трансмутация Фазы (видимое кручение хопфиона)', 
             fontsize=16, fontweight='bold', color='white')

# ПРОТОН
ax1 = fig.add_subplot(1, 2, 1, projection='3d')
ax1.set_facecolor('black')
ax1.plot(x, y, z, color='white', lw=0.8, linestyle='--', alpha=0.3)
# Трубка (полупрозрачная, серая)
tube_mesh = Poly3DCollection([verts_p[face] for face in faces_p], 
                             facecolor='gray', alpha=0.2, edgecolor='none')
ax1.add_collection3d(tube_mesh)
# Цветные нити (красная, зелёная, синяя)
colors = ['red', 'lime', 'cyan']
for i, thread in enumerate(threads_p):
    ax1.plot(thread[:,0], thread[:,1], thread[:,2], color=colors[i], lw=2.5, alpha=0.9)
ax1.set_xlim(-3, 3); ax1.set_ylim(-3, 3); ax1.set_zlim(-3, 3)
ax1.set_title("ПРОТОН\n(Нити прямые, кручения нет)", color='#00ffff', fontsize=14)
ax1.set_axis_off()

# НЕЙТРОН
ax2 = fig.add_subplot(1, 2, 2, projection='3d')
ax2.set_facecolor('black')
ax2.plot(x, y, z, color='white', lw=0.8, linestyle='--', alpha=0.3)
tube_mesh_n = Poly3DCollection([verts_n[face] for face in faces_n], 
                               facecolor='gray', alpha=0.2, edgecolor='none')
ax2.add_collection3d(tube_mesh_n)
for i, thread in enumerate(threads_n):
    ax2.plot(thread[:,0], thread[:,1], thread[:,2], color=colors[i], lw=2.5, alpha=0.9)
ax2.set_xlim(-3, 3); ax2.set_ylim(-3, 3); ax2.set_zlim(-3, 3)
ax2.set_title("НЕЙТРОН (Протон + Электрон)\nХопфион вплетён: нити закручены (Q=1)",
              color='#ff00ff', fontsize=14)
ax2.set_axis_off()

plt.tight_layout()
plt.show()
