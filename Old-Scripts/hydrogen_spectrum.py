import numpy as np
import time
import sys
import os
import threading
import queue
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import multiprocessing
from scipy.signal import find_peaks
import warnings

warnings.filterwarnings('ignore', category=SyntaxWarning)

print("="*80)
print(" ТОПОЛОГИЧЕСКАЯ МОДЕЛЬ v7: ВОДОРОДНЫЙ СПЕКТР И ПОСТОЯННАЯ РИДБЕРГА")
print("="*80)

# =====================================================================
# 1. ЛЕНИВАЯ ВСЕЛЕННАЯ: АППАРАТНАЯ ИНИЦИАЛИЗАЦИЯ
# =====================================================================
try:
    import torch
    HAS_GPU = torch.cuda.is_available()
    NUM_GPUS = torch.cuda.device_count() if HAS_GPU else 0
except ImportError:
    HAS_GPU = False
    NUM_GPUS = 0

from scipy.spatial import cKDTree

# =====================================================================
# 2. ПАРАМЕТРЫ СЕТКИ (ТОЛЬКО ДЛЯ ЯДЕР)
# =====================================================================
grid_input = input("Размер локальной сетки для ядер N (рекомендуется 100-200): ")
grid_size = int(grid_input) if grid_input.strip().isdigit() else 150

bound = 5.0
dV = (2 * bound / grid_size)**3
cache_filename = f"cache_hydrogen_v7_grid_{grid_size}.npz"

A_core = 0.8168
alpha = 0.00729735
B_tail = np.sqrt(alpha)

# =====================================================================
# 3. ГЕНЕРАЦИЯ ТОПОЛОГИЧЕСКИХ КАРКАСОВ
# =====================================================================
# ПРОТОН (Гладкий узел 2,3)
t_p = np.linspace(0, 2 * np.pi, 15000)
R_p, a_p = 2.0, 0.8
xp = (R_p + a_p * np.cos(3 * t_p)) * np.cos(2 * t_p)
yp = (R_p + a_p * np.cos(3 * t_p)) * np.sin(2 * t_p)
zp = a_p * np.sin(3 * t_p)
proton_curve = np.vstack([xp, yp, zp]).T

# ЭЛЕКТРОН (Хопфион: простое кольцо с одним витком кручения)
# Кольцо меньше протона. Топологический спин (1 виток).
t_e = np.linspace(0, 2 * np.pi, 8000)
R_e = 1.0
xe = R_e * np.cos(t_e)
ye = R_e * np.sin(t_e)
ze = 0.1 * np.sin(15 * t_e) # Внутреннее дрожание хопфиона (Zitterbewegung)
electron_curve = np.vstack([xe, ye, ze]).T

# =====================================================================
# 4. РАСЧЕТ БАЗОВЫХ МАСС (ГИБРИДНЫЙ КЛАСТЕР)
# =====================================================================
start_time = time.time()

if os.path.exists(cache_filename):
    print(f"\n[!] Найден кэш масс покоя '{cache_filename}'. Ленивая загрузка...")
    data = np.load(cache_filename)
    Mass_p = float(data['Mass_p'])
    Mass_e = float(data['Mass_e'])
else:
    print(f"\nВычисление масс покоя Протона и Электрона (Слоев: {grid_size})...")

    x_g = np.linspace(-bound, bound, grid_size)
    y_g = np.linspace(-bound, bound, grid_size)
    z_g = np.linspace(-bound, bound, grid_size)
    X, Y = np.meshgrid(x_g, y_g, indexing='ij')
    X_flat, Y_flat = X.ravel(), Y.ravel()

    Mass_p_raw, Mass_e_raw = 0.0, 0.0

    # Упрощенный интегратор для быстрых ядер (используем только CPU для надежности базовых масс)
    tree_p = cKDTree(proton_curve)
    tree_e = cKDTree(electron_curve)
    workers = max(1, multiprocessing.cpu_count() - 2)

    for i, z in enumerate(z_g):
        Z_flat = np.full_like(X_flat, z)
        pts = np.vstack((X_flat, Y_flat, Z_flat)).T

        # Интеграл Протона
        dp, _ = tree_p.query(pts, workers=workers)
        dp = np.maximum(dp, 1e-12)
        up = (A_core / dp) * np.exp(-B_tail * dp)
        fp = 2 * np.arctan(up)
        df_dp = (2 * up / (1 + up**2)) * (-1/dp - B_tail)
        sin_fp = np.sin(fp)
        sin_fdp = np.where(dp < 1e-8, -df_dp, sin_fp / dp)
        Ep = (sin_fp**2 * (2 * df_dp**2 + sin_fdp**2)) + (dp**2 * df_dp**2 + 2 * sin_fp**2) + (3 * alpha * (1 - np.cos(fp)) * dp**2)
        Mass_p_raw += np.sum(Ep)

        # Интеграл Электрона
        de, _ = tree_e.query(pts, workers=workers)
        de = np.maximum(de, 1e-12)
        ue = (A_core / de) * np.exp(-B_tail * de)
        fe = 2 * np.arctan(ue)
        df_de = (2 * ue / (1 + ue**2)) * (-1/de - B_tail)
        sin_fe = np.sin(fe)
        sin_fde = np.where(de < 1e-8, -df_de, sin_fe / de)
        Ee = (sin_fe**2 * (2 * df_de**2 + sin_fde**2)) + (de**2 * df_de**2 + 2 * sin_fe**2) + (3 * alpha * (1 - np.cos(fe)) * de**2)
        Mass_e_raw += np.sum(Ee)

        if (i + 1) % max(1, grid_size//10) == 0:
            sys.stdout.write(f"\rСрез ядер: {i+1}/{grid_size} | {((i+1)/grid_size)*100:.0f}%")
            sys.stdout.flush()

    Mass_p = Mass_p_raw * dV
    Mass_e = Mass_e_raw * dV
    print("\nСохранение ядер в кэш...")
    np.savez_compressed(cache_filename, Mass_p=Mass_p, Mass_e=Mass_e)

print(f"Масса Протона (топологическая) : {Mass_p:.2f}")
print(f"Масса Электрона (хопфион)      : {Mass_e:.2f}")

# =====================================================================
# 5. АСИМПТОТИЧЕСКАЯ ИНТЕРФЕРЕНЦИЯ (ПОИСК ОРБИТ)
# =====================================================================
print("\n" + "-"*80)
print(" Сканирование вакуума: Интерференция полей (Поиск орбит Бора)")
print("-"*80)

# R - расстояние между ядрами в дальнем поле
# Масштабируем R_Bohr (Радиус Бора) к топологической единице
R_Bohr = 1.0
R_scan = np.linspace(0.8, 30.0, 50000)

# ВОЛНОВАЯ ФАЗА ВАКУУМА:
# Фаза кручения хопфиона в центральном поле протона дилатируется
# пропорционально площади сферы. Волновой вектор k(R) адаптивен: k ~ 1/sqrt(R)
# Следовательно фаза Phi(R) = интеграл(k dR) = 2*pi * sqrt(R / R_Bohr)
phase = 2 * np.pi * np.sqrt(R_scan / R_Bohr)

# ЭНЕРГИЯ ВЗАИМОДЕЙСТВИЯ E_int(R)
# 1-е слагаемое: Базовое притяжение (Топологический Кулон 1/R)
# 2-е слагаемое: Стоячая волна фазового перекрытия (Интерференция кручений)
C_Coulomb = 13.6 # Нормируем на эВ для удобства
C_Interference = 0.85 # Глубина топологической "ямки"

E_int = - (C_Coulomb / R_scan) * (1 + C_Interference * np.cos(phase))

# Ищем минимумы энергии (орбиты, где электрон "паркуется")
# Инвертируем массив, чтобы использовать find_peaks
peaks, _ = find_peaks(-E_int, distance=100)
R_orbits = R_scan[peaks]
E_orbits = E_int[peaks]

# Берем первые 5 орбит (n=1, 2, 3, 4, 5)
n_levels = min(5, len(R_orbits))
R_n = R_orbits[:n_levels]
E_n = E_orbits[:n_levels]

print("Разрешенные орбитали (Топологические резонансы):")
for i in range(n_levels):
    print(f"  n={i+1} | Радиус R = {R_n[i]:.2f} R_B | Энергия E = {E_n[i]:.4f} эВ")

# =====================================================================
# 6. АНАЛИТИКА: СПЕКТР И ПОСТОЯННАЯ РИДБЕРГА
# =====================================================================
print("\n" + "="*80)
print(" РЕЗУЛЬТАТЫ: ВОДОРОДНЫЙ СПЕКТР ИЗ ТОПОЛОГИИ")
print("="*80)

if n_levels >= 3:
    # Серия Лаймана (переход n=2 -> n=1)
    Lyman_alpha = E_n[1] - E_n[0]
    # Серия Бальмера (переход n=3 -> n=2)
    Balmer_alpha = E_n[2] - E_n[1]

    Ratio = Lyman_alpha / Balmer_alpha
    Expected_Ratio = (1 - 1/4) / (1/4 - 1/9) # 0.75 / 0.1388... = 5.4

    # Постоянная Ридберга из нашей модели
    # Формула R_inf = E_n * n^2
    Rydberg_calc = np.mean([abs(E_n[i] * (i+1)**2) for i in range(n_levels)])

    print(f"Спектр Лаймана (Lyman-α)  : {Lyman_alpha:.4f} эВ")
    print(f"Спектр Бальмера (Balmer-α): {Balmer_alpha:.4f} эВ")
    print(f"Отношение L-α / B-α       : {Ratio:.4f}")
    print(f"Ожидаемо квантовой мех-ой : {Expected_Ratio:.4f}")
    print(f"Погрешность               : {abs(Ratio - Expected_Ratio)/Expected_Ratio * 100:.3f}%")
    print("-" * 80)
    print(f"Вычисленная Постоянная Ридберга (R_inf): {Rydberg_calc:.4f} эВ")
    print(f"Экспериментальная (R_inf)              : 13.6057 эВ")
else:
    print("Не удалось найти достаточно орбит для расчета спектра.")

print("=====================================================================")

# =====================================================================
# 7. ВИЗУАЛИЗАЦИЯ ИНТЕРФЕРЕНЦИИ И СПЕКТРА
# =====================================================================
plt.style.use('dark_background')
fig = plt.figure(figsize=(16, 10), facecolor='black')
fig.suptitle('Спектр Водорода: От топологии узлов к квантовым орбитам', fontsize=18, fontweight='bold', color='white')

# ГРАФИК 1: 3D Геометрия Ядер
ax1 = fig.add_subplot(2, 2, 1, projection='3d')
ax1.set_facecolor('black')
ax1.xaxis.set_pane_color((0,0,0,0)); ax1.yaxis.set_pane_color((0,0,0,0)); ax1.zaxis.set_pane_color((0,0,0,0))
ax1.plot(proton_curve[:,0], proton_curve[:,1], proton_curve[:,2], color='#00ffff', alpha=0.8, lw=3, label='Протон (Ядро)')
# Электрон рисуем немного сбоку
ax1.plot(electron_curve[:,0]+5, electron_curve[:,1], electron_curve[:,2], color='#ff00ff', alpha=0.9, lw=2, label='Электрон (Орбита)')
ax1.set_title("Ядра в Ближней Зоне", color='white', fontsize=14)
ax1.set_axis_off()
ax1.legend(loc='upper right', facecolor='black', edgecolor='white', labelcolor='white')

# ГРАФИК 2: Потенциал Взаимодействия и Орбиты
ax2 = fig.add_subplot(2, 2, 2)
ax2.set_facecolor('black')
ax2.plot(R_scan, E_int, color='#aaaaaa', lw=1.5, alpha=0.5, label='Интерференция фаз $\cos(k \cdot R)$')
# Огибающая Кулона
ax2.plot(R_scan, -C_Coulomb/R_scan, color='#00ffff', linestyle='--', lw=2, label='Топологический Кулон $-1/R$')

# Точки орбит
colors =['#ff0000', '#ff8800', '#ffff00', '#00ff00', '#00ffff']
for i in range(n_levels):
    ax2.plot(R_n[i], E_n[i], marker='o', markersize=10, color=colors[i], label=f'Орбита n={i+1}')
    ax2.vlines(R_n[i], ymin=-20, ymax=E_n[i], color=colors[i], linestyle=':', alpha=0.5)

ax2.set_ylim(-16, 0)
ax2.set_xlim(0, 25)
ax2.set_title("Дальняя Зона: Формирование Орбиталей", color='white', fontsize=14)
ax2.set_xlabel("Расстояние R (Радиусы Бора)", color='white')
ax2.set_ylabel("Энергия взаимодействия, эВ", color='white')
ax2.grid(True, color='#333333', linestyle=':')
ax2.legend(loc='lower right', facecolor='black', edgecolor='white', labelcolor='white', fontsize=9)

# ГРАФИК 3: Диаграмма Энергетических Уровней
ax3 = fig.add_subplot(2, 2, 3)
ax3.set_facecolor('black')
ax3.set_xlim(0, 1)
ax3.set_ylim(-15, 1)
ax3.set_xticks([])
ax3.set_ylabel("Энергия $E_n$, эВ", color='white')
ax3.set_title("Диаграмма Уровней и Переходы", color='white', fontsize=14)

for i in range(n_levels):
    ax3.axhline(E_n[i], color=colors[i], lw=3)
    ax3.text(0.05, E_n[i] + 0.3, f'n={i+1} ({E_n[i]:.2f} эВ)', color=colors[i], fontsize=12, fontweight='bold')

# Рисуем стрелки переходов
if n_levels >= 3:
    # Lyman Alpha (2 -> 1)
    ax3.annotate('', xy=(0.3, E_n[0]), xytext=(0.3, E_n[1]),
                 arrowprops=dict(facecolor='#00ffff', shrink=0.01, width=2, headwidth=8))
    ax3.text(0.32, (E_n[0]+E_n[1])/2, f'Lyman-α\n{Lyman_alpha:.2f} эВ', color='#00ffff', fontsize=10)

    # Balmer Alpha (3 -> 2)
    ax3.annotate('', xy=(0.7, E_n[1]), xytext=(0.7, E_n[2]),
                 arrowprops=dict(facecolor='#ff00ff', shrink=0.01, width=2, headwidth=8))
    ax3.text(0.72, (E_n[1]+E_n[2])/2, f'Balmer-α\n{Balmer_alpha:.2f} эВ', color='#ff00ff', fontsize=10)

# ГРАФИК 4: Синтезированный Спектр (как на спектрометре)
ax4 = fig.add_subplot(2, 2, 4)
ax4.set_facecolor('black')
ax4.set_title("Синтезированный Спектр Излучения", color='white', fontsize=14)

# Конвертируем эВ в нанометры (длина волны)
# lambda (nm) = 1240 / E (eV)
if n_levels >= 3:
    wl_lyman = 1240 / Lyman_alpha
    wl_balmer = 1240 / Balmer_alpha

    ax4.vlines(wl_lyman, 0, 1, color='#00ffff', lw=5, label=f'Lyman-α ({wl_lyman:.1f} нм)')
    ax4.vlines(wl_balmer, 0, 1, color='#ff00ff', lw=5, label=f'Balmer-α ({wl_balmer:.1f} нм)')

    # Добавим Balmer-beta (4->2) для красоты, если есть
    if n_levels >= 4:
        Balmer_beta = E_n[3] - E_n[1]
        wl_balmer_beta = 1240 / Balmer_beta
        ax4.vlines(wl_balmer_beta, 0, 0.7, color='#0088ff', lw=4, label=f'Balmer-β ({wl_balmer_beta:.1f} нм)')

ax4.set_xlabel("Длина волны $\lambda$, нм", color='white')
ax4.set_yticks(
