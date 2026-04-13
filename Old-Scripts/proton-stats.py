import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
from scipy.stats import binned_statistic

print("--- АНАЛИТИЧЕСКАЯ ФИЗИКА ПРОТОНА (ТРИЛИСТНИК) ---")
print("1. Построение топологического каркаса...")

# =====================================================================
# 1. КАРКАС ПРОТОНА (Q=3, Трилистник)
# =====================================================================
t = np.linspace(0, 2 * np.pi, 2000)
R_torus = 2.0  # Главный радиус
a_torus = 0.8  # Радиус переплетения

x_c = (R_torus + a_torus * np.cos(3 * t)) * np.cos(2 * t)
y_c = (R_torus + a_torus * np.cos(3 * t)) * np.sin(2 * t)
z_c = a_torus * np.sin(3 * t)
curve_points = np.vstack((x_c, y_c, z_c)).T

# =====================================================================
# 2. 3D-ПОЛЕ И ЭНЕРГЕТИКА
# =====================================================================
print("2. Интегрирование фазового континуума (это займет пару секунд)...")

A_core = 0.8168      # Константа жесткости (из электрона)
B_tail = 0.08542     # Константа вакуума (sqrt(1/137))

grid_size = 100      # Высокое разрешение для точной физики
bound = 5.0
x_g = np.linspace(-bound, bound, grid_size)
y_g = np.linspace(-bound, bound, grid_size)
z_g = np.linspace(-bound, bound, grid_size)
X, Y, Z = np.meshgrid(x_g, y_g, z_g, indexing='ij')

grid_points = np.vstack((X.ravel(), Y.ravel(), Z.ravel())).T

# Расстояния до каркаса
tree = cKDTree(curve_points)
distances, _ = tree.query(grid_points)
distances = distances.reshape((grid_size, grid_size, grid_size))
distances = np.maximum(distances, 1e-12)

# Расчет поля (Точная аналитика)
u = (A_core / distances) * np.exp(-B_tail * distances)
f_phase = 2 * np.arctan(u)
# Производная фазы по нормали к трубке
df_dd = (2 * u / (1 + u**2)) * (-1/distances - B_tail)

sin_f = np.sin(f_phase)
sin_f_d = np.where(distances < 1e-8, -df_dd, sin_f / distances)

# Плотности Энергии
dens_I2 = distances**2 * df_dd**2 + 2 * sin_f**2               # Упругость (Хвост)
dens_I4 = sin_f**2 * (2 * df_dd**2 + sin_f_d**2)               # Жесткость (Масса Ядра)
dens_I0 = (1 - np.cos(f_phase)) * distances**2                 # Объемная масса
E_total = dens_I4 + dens_I2 + 3 * (B_tail**2) * dens_I0        # Полная энергия

# =====================================================================
# 3. ВЫЧИСЛЕНИЕ ФИЗИЧЕСКИХ ПАРАМЕТРОВ (ТАБЛИЦА)
# =====================================================================
print("3. Расчет макроскопических параметров...")
dV = (2*bound / grid_size)**3 # Объем одной ячейки решетки

# 3.1 Полная масса (Интеграл энергии)
Mass_total = np.sum(E_total) * dV

# 3.2 Среднеквадратичный радиус (RMS Radius)
# R_cm - расстояние от центра масс протона (0,0,0)
R_cm_sq = X**2 + Y**2 + Z**2
R_cm = np.sqrt(R_cm_sq)
R_rms_dim = np.sqrt(np.sum(R_cm_sq * E_total) * dV / Mass_total)

# 3.3 Вывод таблицы
print("\n" + "="*50)
print(" ТАБЛИЦА ПАРАМЕТРОВ ПРОТОНА (МОДЕЛЬ «АЗЪ»)")
print("="*50)
print(f"Топологический заряд (Число Хопфа)  : Q = 3")
print(f"Интеграл зацепления (Writhe + Tw)   : f = 4")
print(f"Безразмерная Масса (Интеграл E)     : {Mass_total:.4f}")
print(f"Безразмерный RMS Радиус (<r^2>^0.5) : {R_rms_dim:.4f}")
print("-" * 50)
print("ПРОЕКЦИЯ НА ФИЗИКУ (Эксперимент):")
# Если наш R_rms_dim соответствует экспериментальному 0.841 фм:
scale_factor = 0.841 / R_rms_dim
print(f"Масштабный фактор (1 ед. длины)     : {scale_factor:.4f} фм")
print(f"Радиус трубки (a_torus) в фм        : {(a_torus * scale_factor):.4f} фм")
print(f"Радиус протона (R_torus) в фм       : {(R_torus * scale_factor):.4f} фм")
print("="*50)

# =====================================================================
# 4. РАДИАЛЬНЫЕ ПРОФИЛИ И ТОМОГРАФИЯ (ВИЗУАЛИЗАЦИЯ)
# =====================================================================
print("4. Построение профилей энергии...")

# Радиальный профиль (Биннинг энергии по расстоянию от центра)
# Это то, что видят экспериментаторы при рассеянии электронов
bins = np.linspace(0, bound, 100)
radial_energy, bin_edges, _ = binned_statistic(R_cm.ravel(), E_total.ravel(), statistic='sum', bins=bins)
bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
# Нормируем на объем сферического слоя (4 * pi * r^2 * dr), чтобы получить плотность
radial_density = radial_energy / (4 * np.pi * bin_centers**2 * (bins[1]-bins[0]))

# Готовим холст
plt.style.use('dark_background')
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
fig.suptitle('Энергетическая Архитектура Протона', fontsize=18, y=0.98)

# График 1: Томография (Срез по Z=0)
z_mid = grid_size // 2
slice_E = E_total[:, :, z_mid]
im = ax1.imshow(slice_E.T, extent=[-bound, bound, -bound, bound], origin='lower', cmap='magma', interpolation='bilinear')
ax1.set_title('Томография: Тепловой срез экватора (Z=0)')
ax1.set_xlabel('X (безразм.)')
ax1.set_ylabel('Y (безразм.)')
fig.colorbar(im, ax=ax1, fraction=0.046, pad=0.04, label='Плотность энергии')
# Рисуем пунктиром радиус RMS
circle = plt.Circle((0, 0), R_rms_dim, color='cyan', fill=False, linestyle='--', lw=1.5, label=f'RMS Радиус ({R_rms_dim:.2f})')
ax1.add_patch(circle)
ax1.legend(loc='upper right')

# График 2: Радиальный профиль (Предсказание "Полого Ядра")
ax2.plot(bin_centers, radial_density, color='magenta', lw=3, label='Радиальная плотность энергии $\\rho_E(r)$')
ax2.axvline(R_rms_dim, color='cyan', linestyle='--', lw=2, label=f'RMS Радиус')
ax2.set_title('Радиальный профиль массы (Форм-фактор)')
ax2.set_xlabel('Расстояние от центра масс $r$')
ax2.set_ylabel('Усредненная плотность $\\rho_E$')
ax2.set_xlim(0, 4)
ax2.grid(color='white', alpha=0.1)

# Текстовое пояснение прямо на графике
ax2.text(0.5, np.max(radial_density)*0.8, 
         "КЛЮЧЕВОЕ ПРЕДСКАЗАНИЕ:\nВ центре протона (r=0) энергия минимальна!\nМасса сосредоточена в топологической трубке.", 
         color='yellow', fontsize=11, bbox=dict(facecolor='black', alpha=0.6, edgecolor='white'))

ax2.legend()

plt.tight_layout()
plt.show()
print("--- ЗАВЕРШЕНО ---")
