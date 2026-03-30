import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree

print("--- ТОМОГРАФИЯ УЗЛОВ: ПРОТОН vs НЕЙТРОН ---")

# =====================================================================
# 1. ПАРАМЕТРЫ КОНТИНУУМА И СЕТКИ
# =====================================================================
A_core = 0.8168      # Идеальная толщина ядра (выведено из альфы)
B_tail = 0.08542     # Хвост вакуума

grid_size = 500      # Высокое разрешение для томографии
bound = 4.0
x = np.linspace(-bound, bound, grid_size)
y = np.linspace(-bound, bound, grid_size)
X, Y = np.meshgrid(x, y)

# Томографический срез строго по экватору (Z = 0)
grid_pts = np.vstack([X.ravel(), Y.ravel(), np.zeros_like(X.ravel())]).T

# =====================================================================
# 2. ФОРМИРОВАНИЕ ОСЕЙ (ПРОТОН И НЕЙТРОН)
# =====================================================================
print("Генерация топологических каркасов...")
t = np.linspace(0, 2 * np.pi, 5000)
R_torus, a_torus = 2.0, 0.8

# ПРОТОН (Идеальный симметричный трилистник)
xp = (R_torus + a_torus * np.cos(3 * t)) * np.cos(2 * t)
yp = (R_torus + a_torus * np.cos(3 * t)) * np.sin(2 * t)
zp = a_torus * np.sin(3 * t)
proton_curve = np.vstack([xp, yp, zp]).T

# НЕЙТРОН (Трилистник с вплетенным Хопфионом)
# Делаем дефект точно на t = pi (эта точка лежит в плоскости Z=0)
defect = np.exp(-((t - np.pi) / 0.25)**2) 
twist_freq = 15
twist_amp = 0.5

xn = xp + twist_amp * defect * np.cos(twist_freq * t)
yn = yp + twist_amp * defect * np.sin(twist_freq * t)
zn = zp + twist_amp * defect * np.cos(twist_freq * t + np.pi/2)
neutron_curve = np.vstack([xn, yn, zn]).T

# =====================================================================
# 3. РАСЧЕТ ПОЛЯ НА СРЕЗЕ (БЫСТРОЕ ДЕРЕВО ОТРЕЗКОВ)
# =====================================================================
print("Сканирование фазового поля на экваториальном срезе (Z=0)...")

tree_p = cKDTree(proton_curve)
dp, _ = tree_p.query(grid_pts)
dp = dp.reshape((grid_size, grid_size))

tree_n = cKDTree(neutron_curve)
dn, _ = tree_n.query(grid_pts)
dn = dn.reshape((grid_size, grid_size))

# =====================================================================
# 4. ФИЗИКА ТОПОЛОГИЧЕСКОЙ ЭНЕРГИИ
# =====================================================================
def calc_energy_density(d):
    """
    Вычисляет профиль энергии. 
    В центре струны (d=0) фаза f=pi, энергия падает в ноль ("глаз бури").
    Максимум энергии образует трубчатую "кожу" вокруг струны.
    """
    d = np.maximum(d, 1e-3)
    u = (A_core / d) * np.exp(-B_tail * d)
    f = 2 * np.arctan(u)
    
    # Модель плотности энергии I4 (жесткость скрутки)
    # sin(f)^2 обеспечивает черные центры (где f=pi)
    energy = (np.sin(f)**2) * (A_core**2 / (d**2 + 0.05)**2)
    return energy

E_proton = calc_energy_density(dp)
E_neutron = calc_energy_density(dn)

# =====================================================================
# 5. КИБЕР-ФИЗИЧЕСКАЯ ВИЗУАЛИЗАЦИЯ
# =====================================================================
print("Рендеринг томограмм...")

plt.style.use('dark_background')
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))
fig.suptitle('Томография Узлов: Экваториальный срез (Z=0)', fontsize=18, y=0.96)

# ПРОТОН
im1 = ax1.imshow(E_proton, extent=[-bound, bound, -bound, bound], origin='lower', cmap='magma', vmax=np.max(E_proton)*0.8)
ax1.set_title('ПРОТОН ($p^+$)\nИдеальная $\\mathbb{Z}_3$ симметрия', fontsize=14, color='cyan')
ax1.set_xlabel('X (безразм.)')
ax1.set_ylabel('Y (безразм.)')
ax1.contour(X, Y, E_proton, levels=[np.max(E_proton)*0.1], colors='cyan', linestyles='dashed', linewidths=1)

# НЕЙТРОН
im2 = ax2.imshow(E_neutron, extent=[-bound, bound, -bound, bound], origin='lower', cmap='magma', vmax=np.max(E_proton)*0.8)
ax2.set_title('НЕЙТРОН ($n^0$)\nВплетенный дефект (Левосторонняя хиральность)', fontsize=14, color='coral')
ax2.set_xlabel('X (безразм.)')
ax2.contour(X, Y, E_neutron, levels=[np.max(E_proton)*0.1], colors='coral', linestyles='dashed', linewidths=1)

# Отмечаем зону фрустрации
ax2.annotate('Область фрустрации\n(Запертый электрон)', xy=(1, 0), xytext=(1, 3),
             arrowprops=dict(facecolor='yellow', shrink=0.05, width=1.5, headwidth=8),
             color='yellow', fontsize=11, fontweight='bold', bbox=dict(boxstyle="round,pad=0.3", fc="black", ec="yellow", lw=1))

# Добавляем цветовые шкалы
cbar1 = fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
cbar1.set_label('Плотность энергии $\\rho_E$', rotation=270, labelpad=15)
cbar2 = fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
cbar2.set_label('Плотность энергии $\\rho_E$', rotation=270, labelpad=15)

plt.tight_layout()
plt.show()

print("--- ГОТОВО ---")
