import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import pandas as pd
import os

# =====================================================================
# 1. ЗАГРУЗКА ДАННЫХ (или запуск расчётов, если файла нет)
# =====================================================================

print("="*60)
print(" ЧЕСТНЫЙ 3D-ИНТЕГРАЛ: СХОДИМОСТЬ ПО СЕТКЕ")
print("="*60)

csv_file = 'mass_convergence.csv'

if not os.path.exists(csv_file):
    # Если файла нет, нужно запустить расчёты (здесь вставлен код из предыдущей версии без сгущения)
    # Для краткости предполагаем, что файл уже есть. Если нет — можно добавить вызов функции run_calculation.
    print("Файл mass_convergence.csv не найден. Сначала запустите расчёт на сетках.")
    exit(1)

df = pd.read_csv(csv_file)
print("Загружены данные из", csv_file)
print(df.to_string(index=False))

# =====================================================================
# 2. ЭКСТРАПОЛЯЦИЯ
# =====================================================================

def fit_func(N, a, b):
    return a + b / N

N_vals = df['grid_size'].values
dM_vals = df['Delta_Mass_MeV'].values

# Используем точки с N >= 150 (сходимость уже хорошая)
mask = N_vals >= 150
popt, pcov = curve_fit(fit_func, N_vals[mask], dM_vals[mask], p0=[1.293, 10])
a_fit, b_fit = popt
dM_extrap = a_fit
dM_extrap_err = np.sqrt(pcov[0,0])

print("\nАппроксимация Δm(N) = a + b/N")
print(f"  a = {a_fit:.5f} ± {dM_extrap_err:.5f} МэВ (экстраполированное значение)")
print(f"  b = {b_fit:.5f} МэВ")

# =====================================================================
# 3. ГРАФИК
# =====================================================================

plt.style.use('dark_background')
fig, ax = plt.subplots(figsize=(10, 6))

# Точки расчёта
ax.plot(N_vals, dM_vals, 'o-', color='cyan', linewidth=2, markersize=8, label='Расчётное значение')

# Экспериментальное значение
ax.axhline(y=1.293, color='red', linestyle='--', linewidth=1.5, label='Эксперимент (1.293 МэВ)')

# Экстраполяционная кривая
N_fine = np.linspace(100, 500, 200)
ax.plot(N_fine, fit_func(N_fine, a_fit, b_fit), '--', color='gray', alpha=0.8,
        label=f'Экстраполяция: {a_fit:.3f} МэВ')

# Доверительный интервал (грубо)
sigma = np.sqrt(np.diag(pcov))
y_upper = fit_func(N_fine, a_fit + sigma[0], b_fit + sigma[1])
y_lower = fit_func(N_fine, a_fit - sigma[0], b_fit - sigma[1])
ax.fill_between(N_fine, y_lower, y_upper, color='gray', alpha=0.2, label='1σ доверительный интервал')

ax.set_xlabel('Размер сетки (N³)', fontsize=12)
ax.set_ylabel('Дефект масс Δm (МэВ)', fontsize=12)
ax.set_title('Сходимость дефекта масс нейтрона с увеличением разрешения сетки', fontsize=14)
ax.grid(color='white', alpha=0.2)
ax.legend()

plt.tight_layout()
plt.savefig('mass_convergence.png', dpi=150)
plt.show()

# =====================================================================
# 4. ИТОГОВАЯ ТАБЛИЦА
# =====================================================================

print("\n" + "="*60)
print(" ИТОГОВЫЕ РЕЗУЛЬТАТЫ")
print("="*60)
print(f"Экстраполированное значение дефекта масс: {a_fit:.3f} ± {dM_extrap_err:.3f} МэВ")
print(f"Экспериментальное значение:                1.293 МэВ")
print(f"Расхождение: {abs(a_fit - 1.293):.3f} МэВ ({abs(a_fit/1.293 - 1)*100:.2f}%)")
print("\nГрафик сохранён как mass_convergence.png")
