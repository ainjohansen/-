import numpy as np
import matplotlib.pyplot as plt

# Параметры
k = 25.0
N_theta = 300
N_phi = 300

# Генерация трилистника (5000 точек)
t = np.linspace(0, 2*np.pi, 5000)
R, r = 2.0, 0.8
x = (R + r*np.cos(3*t)) * np.cos(2*t)
y = (R + r*np.cos(3*t)) * np.sin(2*t)
z = r * np.sin(3*t)
curve = np.vstack([x, y, z]).T

# Угловая сетка
theta = np.linspace(0, np.pi, N_theta)
phi = np.linspace(0, 2*np.pi, N_phi)
TH, PH = np.meshgrid(theta, phi, indexing='ij')

# q-вектор
qx = k * (np.sin(TH)*np.cos(PH) - 1)
qy = k * (np.sin(TH)*np.sin(PH))
qz = k * (np.cos(TH) - 0)

# Векторизованный расчёт амплитуды
q_flat = np.stack([qx.ravel(), qy.ravel(), qz.ravel()], axis=1)
phases = np.exp(1j * np.dot(curve, q_flat.T))
amp_flat = np.sum(phases, axis=0)
intensity = np.abs(amp_flat.reshape(N_theta, N_phi))**2
intensity_norm = intensity / np.max(intensity)
intensity_log = np.log10(intensity_norm + 1e-12)

# Графики
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14,6))

# 3D трилистник
ax1 = fig.add_subplot(121, projection='3d')
ax1.plot(x, y, z, 'cyan', linewidth=2, alpha=0.7)
t_cross = [0, 2*np.pi/3, 4*np.pi/3]
cross_pts = []
for tc in t_cross:
    xc = (R + r*np.cos(3*tc)) * np.cos(2*tc)
    yc = (R + r*np.cos(3*tc)) * np.sin(2*tc)
    zc = r * np.sin(3*tc)
    cross_pts.append([xc, yc, zc])
cross_pts = np.array(cross_pts)
ax1.scatter(cross_pts[:,0], cross_pts[:,1], cross_pts[:,2], color='red', s=100, label='Перехлёсты петель')
ax1.set_title('Трилистник (протон) и три точки перехлёста')
ax1.set_xlabel('x'); ax1.set_ylabel('y'); ax1.set_zlabel('z')
ax1.legend()

# Угловое распределение с контурами
ax2 = fig.add_subplot(122)
im = ax2.pcolormesh(phi, theta, intensity_log, shading='auto', cmap='hot', vmin=-4, vmax=0)
plt.colorbar(im, ax=ax2, label='log10(Интенсивность)')
ax2.contour(phi, theta, intensity_norm, levels=[0.5, 0.7, 0.9], colors='white', linewidths=1)
ax2.set_xlabel('Азимут φ (рад)')
ax2.set_ylabel('Полярный угол θ (рад)')
ax2.set_title('Угловое распределение рассеяния\n(три максимума от перехлёстов)')

plt.tight_layout()
plt.show()
