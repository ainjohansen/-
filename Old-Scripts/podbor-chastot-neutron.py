import numpy as np
import time
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
import multiprocessing
import warnings
warnings.filterwarnings('ignore')

# =====================================================================
# ПАРАМЕТРЫ МОДЕЛИ (фиксированы, как в первом скрипте)
# =====================================================================
grid_size = 150          # 150³ = 3.375 млн узлов (быстро, достаточно для тренда)
bound = 5.0
dV = (2 * bound / grid_size) ** 3

A_core = 0.8168
alpha = 0.00729735
B_tail = np.sqrt(alpha)

Mp_exp = 938.272088      # масса протона, МэВ
Delta_exp = 1.293332     # экспериментальный дефект масс нейтрона, МэВ

# Базовые параметры трилистника (без модуляции)
R_torus, a_torus = 2.0, 0.8

# =====================================================================
# ГЕНЕРАЦИЯ КРИВОЙ С ВЫСОКОЧАСТОТНОЙ МОДУЛЯЦИЕЙ
# =====================================================================
def generate_curve(freq, amp, num_points=15000):
    """
    Возвращает точки (N,3) кривой трилистника с добавленной вибрацией.
    freq – частота модуляции (целое число),
    amp  – амплитуда модуляции.
    """
    t = np.linspace(0, 2 * np.pi, num_points)
    x0 = (R_torus + a_torus * np.cos(3*t)) * np.cos(2*t)
    y0 = (R_torus + a_torus * np.cos(3*t)) * np.sin(2*t)
    z0 = a_torus * np.sin(3*t)
    if amp == 0.0:
        return np.vstack([x0, y0, z0]).T
    # Добавляем осцилляции
    x = x0 + amp * np.cos(freq * t)
    y = y0 + amp * np.sin(freq * t)
    z = z0 + amp * np.cos(freq * t + np.pi/2)
    return np.vstack([x, y, z]).T

# =====================================================================
# ИНТЕГРИРОВАНИЕ ПОЛНОЙ ЭНЕРГИИ (БЕЗРАЗМЕРНОЙ)
# =====================================================================
def compute_mass(curve, X_flat, Y_flat, z_g):
    """Возвращает безразмерную массу (интеграл плотности энергии * dV)"""
    tree = cKDTree(curve)
    workers = max(1, multiprocessing.cpu_count() - 1)
    E_total = 0.0
    total = grid_size * len(X_flat)
    processed = 0
    for iz in range(grid_size):
        Z_flat = np.full_like(X_flat, z_g[iz])
        pts = np.vstack([X_flat, Y_flat, Z_flat]).T
        d, _ = tree.query(pts, workers=workers)
        d = np.maximum(d, 1e-12)
        u = (A_core / d) * np.exp(-B_tail * d)
        f = 2 * np.arctan(u)
        df_dd = (2*u/(1+u**2)) * (-1/d - B_tail)
        sin_f = np.sin(f)
        sin_f_dd = np.where(d < 1e-8, -df_dd, sin_f / d)
        E = (sin_f**2 * (2*df_dd**2 + sin_f_dd**2)) + (d**2 * df_dd**2 + 2*sin_f**2) + (3*alpha * (1 - np.cos(f)) * d**2)
        E_total += np.sum(E)
        processed += len(X_flat)
        # Прогресс (каждый слой)
        if iz % 10 == 0:
            print(f"\r  {processed/total*100:.1f}%", end="")
    print()
    return E_total * dV

# =====================================================================
# ПОДГОТОВКА СЕТКИ (один раз)
# =====================================================================
print("Подготовка 3D сетки...")
x = np.linspace(-bound, bound, grid_size)
y = np.linspace(-bound, bound, grid_size)
z = np.linspace(-bound, bound, grid_size)
X, Y = np.meshgrid(x, y, indexing='ij')
X_flat, Y_flat = X.ravel(), Y.ravel()

# Базовый протон (amp=0)
print("\nРасчёт массы протона (базовая кривая, amp=0)...")
curve_proton = generate_curve(freq=0, amp=0)
M0_raw = compute_mass(curve_proton, X_flat, Y_flat, z)
print(f"Безразмерная масса протона: {M0_raw:.6f}")
scale = Mp_exp / M0_raw
print(f"Калибровочный коэффициент: {scale:.6f} МэВ/ед")

# =====================================================================
# ПЕРЕБОР ЧАСТОТ И ПОДБОР АМПЛИТУДЫ
# =====================================================================
frequencies = [3, 6, 9, 12, 15, 18]
target_delta = Delta_exp
results = {}   # {freq: amp_target}

for freq in frequencies:
    print(f"\n=== Частота {freq} ===")
    # Попробуем несколько амплитуд (малых, чтобы квадратичность работала)
    amps_try = np.array([0.005, 0.010, 0.015, 0.020, 0.025, 0.030])
    deltas_mev = []
    for amp in amps_try:
        print(f"  Амплитуда {amp:.4f} ... ", end="")
        curve = generate_curve(freq, amp)
        M_raw = compute_mass(curve, X_flat, Y_flat, z)
        delta_raw = M_raw - M0_raw
        delta_mev = delta_raw * scale
        deltas_mev.append(delta_mev)
        print(f"ΔM = {delta_mev:.5f} МэВ")
    # Аппроксимация ΔM = k * amp^2
    # Используем все точки, взвешивая по точности (можно и просто МНК)
    amps2 = amps_try ** 2
    # Линейная регрессия через начало координат: delta = k * amp^2
    k = np.sum(deltas_mev * amps2) / np.sum(amps2**2)
    # Предсказанные значения
    delta_pred = k * amps2
    # Ошибка аппроксимации (среднеквадратичная)
    rms = np.sqrt(np.mean((deltas_mev - delta_pred)**2))
    print(f"  Коэффициент k = {k:.3f} МэВ, RMS = {rms:.4f} МэВ")
    # Необходимая амплитуда для target_delta
    amp_target = np.sqrt(target_delta / k)
    results[freq] = amp_target
    print(f"  -> Амплитуда, дающая ΔM = {target_delta} МэВ: {amp_target:.6f}")

# =====================================================================
# ВЫВОД И ВИЗУАЛИЗАЦИЯ
# =====================================================================
print("\n" + "="*60)
print("РЕЗУЛЬТАТЫ ПОДБОРА АМПЛИТУДЫ ДЛЯ ΔM = 1.293332 МэВ")
print("="*60)
for freq in frequencies:
    print(f"freq = {freq:2d} : amp_target = {results[freq]:.6f}")

# График зависимости amp_target от частоты
freqs_list = list(results.keys())
amps_list = list(results.values())

plt.figure(figsize=(8,5))
plt.plot(freqs_list, amps_list, 'o-', color='cyan', linewidth=2, markersize=8)
plt.axhline(y=0.0247371, color='magenta', linestyle='--', label='amp=0.02474 (из первого скрипта)')
plt.xlabel('Частота модуляции freq', fontsize=12)
plt.ylabel('Требуемая амплитуда amp_target', fontsize=12)
plt.title('Амплитуда, необходимая для достижения ΔM = 1.293 МэВ')
plt.grid(True, alpha=0.3)
plt.legend()
plt.show()

# Дополнительно: проверим, что для freq=15 амплитуда близка к 0.02474
if 15 in results:
    print(f"\nПроверка: для freq=15 amp_target = {results[15]:.6f}, ожидалось ~0.0247371")
    print(f"Относительное отличие: {abs(results[15]-0.0247371)/0.0247371*100:.2f}%")
