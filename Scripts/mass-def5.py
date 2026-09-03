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
grid_size = 300  # будем использовать сетку 300³ для высокой точности
R_torus = 2.0
a_torus = 0.8
twist_amp = 0.0867
twist_freq = 15

def generate_curves(t, relax_amp=0.0):
    """Генерирует кривые протона и нейтрона с возможной релаксацией радиуса трубки."""
    # Протон (без релаксации)
    xp = (R_torus + a_torus * np.cos(3 * t)) * np.cos(2 * t)
    yp = (R_torus + a_torus * np.cos(3 * t)) * np.sin(2 * t)
    zp = a_torus * np.sin(3 * t)
    proton_curve = np.vstack([xp, yp, zp]).T

    # Нейтрон
    defect_center = 0.0
    dt_ang = np.pi - np.abs(np.pi - np.abs(t - defect_center))
    defect = np.exp(-(dt_ang / 0.25)**2)
    # Локальное увеличение радиуса трубки
    a_local = a_torus * (1 + relax_amp * defect)
    xn = (R_torus + a_local * np.cos(3 * t)) * np.cos(2 * t)
    yn = (R_torus + a_local * np.cos(3 * t)) * np.sin(2 * t)
    zn = a_local * np.sin(3 * t)
    # Добавляем кручение (дефект Хопфа)
    xn += twist_amp * defect * np.cos(twist_freq * t)
    yn += twist_amp * defect * np.sin(twist_freq * t)
    zn += twist_amp * defect * np.cos(twist_freq * t + np.pi/2)
    neutron_curve = np.vstack([xn, yn, zn]).T

    return proton_curve, neutron_curve

def compute_energy_field(curve_points, grid_pts, dV):
    """Вычисляет энергию поля для заданной кривой и сетки."""
    tree = cKDTree(curve_points)
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
    Mass = np.sum(E_total) * dV
    return Mass

# Сетка (один раз для всех расчётов)
print("Генерация сетки...")
x_g = np.linspace(-bound, bound, grid_size)
y_g = np.linspace(-bound, bound, grid_size)
z_g = np.linspace(-bound, bound, grid_size)
X, Y, Z = np.meshgrid(x_g, y_g, z_g, indexing='ij')
grid_pts = np.vstack((X.ravel(), Y.ravel(), Z.ravel())).T
dV = (2 * bound / grid_size)**3

t = np.linspace(0, 2 * np.pi, 5000)

# Сначала вычислим массу протона (не зависит от relax_amp)
proton_curve, _ = generate_curves(t, relax_amp=0.0)
Mass_p = compute_energy_field(proton_curve, grid_pts, dV)
print(f"Масса протона (безразм.): {Mass_p:.4f}")

# Исследуем зависимость дефекта от relax_amp
relax_amps = np.linspace(0, 0.05, 11)  # от 0 до 0.05
results = []
for ra in relax_amps:
    start = time.time()
    _, neutron_curve = generate_curves(t, relax_amp=ra)
    Mass_n = compute_energy_field(neutron_curve, grid_pts, dV)
    delta = Mass_n - Mass_p
    delta_MeV = delta / Mass_p * 938.272
    results.append([ra, Mass_n, delta, delta_MeV])
    elapsed = time.time() - start
    print(f"relax_amp={ra:.3f}: Δm = {delta_MeV:.3f} МэВ ({elapsed:.1f} сек)")

df = pd.DataFrame(results, columns=['relax_amp', 'Mass_n', 'Delta', 'Delta_MeV'])
df.to_csv('neutron_relaxation.csv', index=False)

# Построение графика
plt.style.use('dark_background')
fig, ax = plt.subplots(figsize=(10,6))
ax.plot(df['relax_amp'], df['Delta_MeV'], 'o-', color='cyan', linewidth=2, markersize=6, label='Расчёт (сетка 300³)')
ax.axhline(y=1.293, color='red', linestyle='--', linewidth=1.5, label='Эксперимент (1.293 МэВ)')
ax.set_xlabel('Амплитуда релаксации $\\varepsilon$ (локальное увеличение радиуса трубки)', fontsize=12)
ax.set_ylabel('Дефект масс Δm (МэВ)', fontsize=12)
ax.set_title('Зависимость дефекта масс нейтрона от релаксации трилистника', fontsize=14)
ax.grid(color='white', alpha=0.2)
ax.legend()
plt.tight_layout()
plt.savefig('neutron_relaxation.png', dpi=150)
plt.show()

# Найдём relax_amp, при котором достигается эксперимент
from scipy.interpolate import interp1d
f_interp = interp1d(df['Delta_MeV'], df['relax_amp'], kind='linear')
try:
    ra_exp = f_interp(1.293)
    print(f"\nДля достижения Δm = 1.293 МэВ требуется relax_amp ≈ {ra_exp:.4f}")
except:
    print("\nЭкспериментальное значение не достигнуто в данном диапазоне, но тренд показывает необходимость небольшой релаксации.")
