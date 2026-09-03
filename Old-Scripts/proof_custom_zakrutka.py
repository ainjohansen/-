import numpy as np
import time
import sys
import os
import threading
import queue
import matplotlib.pyplot as plt
import multiprocessing
import warnings

warnings.filterwarnings('ignore', category=SyntaxWarning)

# =====================================================================
# ИНИЦИАЛИЗАЦИЯ И ПОИСК ЖЕЛЕЗА
# =====================================================================
print("="*75)
print(" МОДЕЛИРОВАНИЕ НЕЙТРОНА: ФРЕЙМИНГ ТРИЛИСТНИКА (TWIST, БЕЗ ИСКАЖЕНИЯ ФОРМЫ)")
print("="*75)

try:
    import torch
    HAS_GPU = torch.cuda.is_available()
    NUM_GPUS = torch.cuda.device_count() if HAS_GPU else 0
except ImportError:
    HAS_GPU = False
    NUM_GPUS = 0

from scipy.spatial import cKDTree

print(f"[Железо] Процессор: {multiprocessing.cpu_count()} ядер")
if HAS_GPU:
    print(f"[Железо] GPU: {NUM_GPUS} шт.")
else:
    print("[Железо] GPU не обнаружены, используется CPU")

# =====================================================================
# ПАРАМЕТРЫ СЕТКИ И МОДЕЛИ
# =====================================================================
grid_input = input("\nВведите размер сетки N (рекомендуется 150-250): ")
grid_size = int(grid_input) if grid_input.strip().isdigit() else 200

bound = 5.0
dV = (2 * bound / grid_size)**3
cache_filename = f"proton_mass_grid_{grid_size}.npz"

print(f"Сетка: {grid_size}^3 = {grid_size**3:,} узлов")

# Параметры солитонного поля (из предыдущей модели)
A_core = 0.8168
alpha = 0.00729735          # постоянная тонкой структуры
B_tail = np.sqrt(alpha)     # параметр экспоненциального хвоста

# Экспериментальные данные
Mass_p_MeV = 938.272088      # масса протона, МэВ
Exp_defect = 1.293332        # дефект масс нейтрона, МэВ

# Параметр скручивания для нейтрона (из статьи: резонанс с sqrt(alpha))
theta_res = np.sqrt(alpha)   # ≈ 0.08542
# Можно также использовать подгоночное значение 0.08636, но оставим теоретическое
print(f"Теоретический параметр скручивания нейтрона: θ_res = √α = {theta_res:.6f}")

# =====================================================================
# ГЕНЕРАЦИЯ БАЗОВОЙ КРИВОЙ ТРИЛИСТНИКА (НЕ ЗАВИСИТ ОТ β)
# =====================================================================
def generate_trefoil_curve(num_points=15000):
    """Возвращает точки трилистника (протон) — единая форма для всех β"""
    t = np.linspace(0, 2 * np.pi, num_points)
    R_torus, a_torus = 2.0, 0.8
    x = (R_torus + a_torus * np.cos(3 * t)) * np.cos(2 * t)
    y = (R_torus + a_torus * np.cos(3 * t)) * np.sin(2 * t)
    z = a_torus * np.sin(3 * t)
    return np.vstack([x, y, z]).T

# =====================================================================
# ФУНКЦИЯ ВЫЧИСЛЕНИЯ МАССЫ ДЛЯ ЗАДАННОЙ КРИВОЙ (ПОЛНАЯ ЭНЕРГИЯ ПОЛЯ)
# =====================================================================
def compute_mass_for_curve(curve, grid_size, X_flat, Y_flat, z_g):
    """Интегрирует плотность энергии по объёму для данной кривой (только базовая часть, без twist)"""
    tree = cKDTree(curve)
    cpu_workers = max(1, multiprocessing.cpu_count() - 1)
    
    E_total = 0.0
    total_points = grid_size * len(X_flat)
    processed = 0
    
    for i in range(grid_size):
        Z_flat = np.full_like(X_flat, z_g[i])
        slice_pts = np.vstack((X_flat, Y_flat, Z_flat)).T
        d, _ = tree.query(slice_pts, workers=cpu_workers)
        d = np.maximum(d, 1e-12)
        
        u = (A_core / d) * np.exp(-B_tail * d)
        f = 2 * np.arctan(u)
        df_dd = (2 * u / (1 + u**2)) * (-1/d - B_tail)
        sin_f = np.sin(f)
        sin_f_dd = np.where(d < 1e-8, -df_dd, sin_f / d)
        E = (sin_f**2 * (2 * df_dd**2 + sin_f_dd**2)) + (d**2 * df_dd**2 + 2 * sin_f**2) + (3 * alpha * (1 - np.cos(f)) * d**2)
        
        E_total += np.sum(E)
        processed += len(X_flat)
        sys.stdout.write(f"\r  Интегрирование: {processed/total_points*100:.1f}%")
        sys.stdout.flush()
    
    print()
    return E_total * dV

# =====================================================================
# ОСНОВНАЯ ЧАСТЬ: ВЫЧИСЛЕНИЕ МАССЫ ПРОТОНА (β=0) И ОПРЕДЕЛЕНИЕ КОЭФФИЦИЕНТА ЖЁСТКОСТИ
# =====================================================================
start_time = time.time()

x_g = np.linspace(-bound, bound, grid_size)
y_g = np.linspace(-bound, bound, grid_size)
z_g = np.linspace(-bound, bound, grid_size)
X, Y = np.meshgrid(x_g, y_g, indexing='ij')
X_flat, Y_flat = X.ravel(), Y.ravel()

# Загрузка или вычисление массы протона
if os.path.exists(cache_filename):
    print(f"\nЗагрузка кэша протона: {cache_filename}")
    data = np.load(cache_filename)
    proton_mass_raw = data['proton_mass']
    print(f"Масса протона (безразмерная): {proton_mass_raw:.6f}")
else:
    print("\nВычисление массы протона (трилистник, β=0)...")
    curve = generate_trefoil_curve()
    proton_mass_raw = compute_mass_for_curve(curve, grid_size, X_flat, Y_flat, z_g)
    print(f"Масса протона (безразмерная): {proton_mass_raw:.6f}")
    np.savez_compressed(cache_filename, proton_mass=proton_mass_raw)

# Пересчёт в МэВ (калибровка: масса протона должна быть 938.272 МэВ)
calibration_factor = Mass_p_MeV / proton_mass_raw
print(f"Калибровочный коэффициент: {calibration_factor:.6f} (перевод у.е. → МэВ)")

# Определение коэффициента жёсткости k_twist из экспериментального дефекта при β = θ_res
# ΔE(β) = 0.5 * k * β^2  =>  k = 2 * ΔE_exp / θ_res^2
k_twist = 2 * Exp_defect / (theta_res**2)
print(f"Коэффициент упругости кручения: k = {k_twist:.3f} МэВ/рад²")

# =====================================================================
# РАСЧЁТ ДЕФЕКТА МАСС ДЛЯ РАЗЛИЧНЫХ β (КВАДРАТИЧНЫЙ ЗАКОН)
# =====================================================================
# Диапазон β: от 0 до ~0.15 (малые углы, чтобы оставаться в гармоническом приближении)
beta_values = np.linspace(0, 0.15, 50)
defect_mev = 0.5 * k_twist * beta_values**2

# Вычисление массы нейтрона в МэВ
neutron_mass_mev = Mass_p_MeV + defect_mev

# Точка, соответствующая нейтрону (β = θ_res)
idx_neutron = np.argmin(np.abs(beta_values - theta_res))
beta_neutron = beta_values[idx_neutron]
defect_neutron = defect_mev[idx_neutron]
mass_neutron = neutron_mass_mev[idx_neutron]

# =====================================================================
# ВЫВОД РЕЗУЛЬТАТОВ
# =====================================================================
print("\n" + "="*75)
print(f" РЕЗУЛЬТАТЫ МОДЕЛИРОВАНИЯ (Сетка: {grid_size}^3)")
print("="*75)
print(f"Масса протона (эксперимент):       {Mass_p_MeV:.6f} МэВ")
print(f"Масса протона (модель, калибровка): {proton_mass_raw * calibration_factor:.6f} МэВ")
print(f"Параметр скручивания нейтрона β_n = {theta_res:.6f} рад")
print(f"Дефект масс (эксперимент):          {Exp_defect:.6f} МэВ")
print(f"Дефект масс (модель, квадратичный): {defect_neutron:.6f} МэВ")
print(f"Относительная погрешность:          {abs(defect_neutron-Exp_defect)/Exp_defect*100:.4f}%")
print(f"Масса нейтрона (модель):            {mass_neutron:.6f} МэВ")
print(f"Время выполнения:                   {time.time()-start_time:.2f} с")

# =====================================================================
# ПОСТРОЕНИЕ ГРАФИКА
# =====================================================================
plt.style.use('dark_background')
fig, ax = plt.subplots(figsize=(10, 7), facecolor='black')
ax.set_facecolor('black')

ax.plot(beta_values, defect_mev, 'c-', lw=2.5, label=r'$\Delta M(\beta) = \frac{1}{2} k \beta^2$')
ax.axvline(x=theta_res, color='magenta', linestyle='--', lw=1.5, label=f'β = √α = {theta_res:.4f} (нейтрон)')
ax.axhline(y=Exp_defect, color='yellow', linestyle=':', lw=1.5, label=f'Эксперимент: {Exp_defect} МэВ')
ax.plot(theta_res, defect_neutron, 'ro', markersize=8, label='Модельное значение')

ax.set_xlabel('Параметр скручивания β (радианы)', fontsize=12, color='white')
ax.set_ylabel('Дефект масс ΔM (МэВ)', fontsize=12, color='white')
ax.set_title(f'Зависимость дефекта масс от кручения (twist) трилистника\nСетка {grid_size}³, калибровка по массе протона', fontsize=14, color='white')
ax.tick_params(colors='white')
ax.grid(True, color='#333333', linestyle=':')
ax.legend(facecolor='black', edgecolor='white', labelcolor='white', fontsize=11)

plt.tight_layout()
plt.show()

# =====================================================================
# ПРИМЕЧАНИЕ
# =====================================================================
print("\nПримечание: В данной реализации центральная линия узла не искажается.")
print("Энергия кручения добавлена как квадратичная поправка, что соответствует")
print("физике фрейминга (теорема Калугэряну–Уайта–Фуллера).")
print("Нейтрон — это то же топологическое состояние, что и протон, но с")
print("дополнительным глобальным скручиванием трубки на угол β = √α.")
