import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
import time
import pandas as pd

# Константы
A_core = 0.8168
B_tail = 0.08542
alpha = 0.00729735
bound = 5.0
grid_size = 300
R_torus = 2.0
a_torus = 0.8
twist_freq = 15

# Генерация сетки один раз
x = np.linspace(-bound, bound, grid_size)
y = np.linspace(-bound, bound, grid_size)
z = np.linspace(-bound, bound, grid_size)
X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
grid_pts = np.vstack((X.ravel(), Y.ravel(), Z.ravel())).T
dV = (2 * bound / grid_size)**3

# Функция вычисления энергии
def compute_energy(curve_pts):
    tree = cKDTree(curve_pts)
    d, _ = tree.query(grid_pts, workers=-1)
    d = np.maximum(d, 1e-12)
    u = (A_core / d) * np.exp(-B_tail * d)
    f = 2 * np.arctan(u)
    df_dd = (2 * u / (1 + u**2)) * (-1/d - B_tail)
    sin_f = np.sin(f)
    sin_f_d = np.where(d < 1e-8, -df_dd, sin_f / d)
    dens_I2 = d**2 * df_dd**2 + 2 * sin_f**2
    dens_I4 = sin_f**2 * (2 * df_dd**2 + sin_f_d**2)
    dens_I0 = (1 - np.cos(f)) * d**2
    E_total = dens_I4 + dens_I2 + 3 * alpha * dens_I0
    return np.sum(E_total) * dV

# Генерация протона (один раз)
t = np.linspace(0, 2 * np.pi, 5000)
xp = (R_torus + a_torus * np.cos(3 * t)) * np.cos(2 * t)
yp = (R_torus + a_torus * np.cos(3 * t)) * np.sin(2 * t)
zp = a_torus * np.sin(3 * t)
proton_curve = np.vstack([xp, yp, zp]).T
Mass_p = compute_energy(proton_curve)
print(f"Масса протона: {Mass_p:.4f} (безразм.)")

# Поиск оптимального twist_amp
twist_amps = np.linspace(0.0860, 0.0875, 30)  # небольшой диапазон вокруг 0.0867
results = []
for twist_amp in twist_amps:
    start = time.time()
    defect_center = 0.0
    dt_ang = np.pi - np.abs(np.pi - np.abs(t - defect_center))
    defect = np.exp(-(dt_ang / 0.25)**2)
    xn = xp + twist_amp * defect * np.cos(twist_freq * t)
    yn = yp + twist_amp * defect * np.sin(twist_freq * t)
    zn = zp + twist_amp * defect * np.cos(twist_freq * t + np.pi/2)
    neutron_curve = np.vstack([xn, yn, zn]).T
    Mass_n = compute_energy(neutron_curve)
    delta = (Mass_n - Mass_p) / Mass_p * 938.272
    results.append((twist_amp, delta))
    print(f"twist_amp={twist_amp:.5f}: Δm = {delta:.3f} МэВ ({time.time()-start:.1f} сек)")

# Сохранение и график
df = pd.DataFrame(results, columns=['twist_amp', 'delta_MeV'])
df.to_csv('twist_tuning.csv', index=False)

plt.plot(df['twist_amp'], df['delta_MeV'], 'o-')
plt.axhline(y=1.293, color='r', linestyle='--', label='Эксперимент')
plt.xlabel('twist_amp')
plt.ylabel('Δm (МэВ)')
plt.legend()
plt.grid()
plt.savefig('twist_tuning.png')
plt.show()
