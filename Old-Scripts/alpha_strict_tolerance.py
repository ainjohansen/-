import numpy as np
import matplotlib.pyplot as plt

print("="*65)
print(" АНАЛИТИЧЕСКИЙ АТТРАКТОР МЕРЫ (СТРОГОЕ МАСШТАБИРОВАНИЕ)")
print("="*65)

# Идеальные значения из точного BVP-решения
I2_base = 5.780647
I4_base = 5.825971
I0_base = 2.070358
alpha_ideal = 0.00729735

# Параметр масштаба 'a'. 
# a = 1.0 : идеальное состояние (0 Кельвинов)
# a > 1.0 : среда растягивает электрон
# a < 1.0 : среда сжимает электрон
a_vals = np.linspace(0.985, 1.01, 1000)

# Строгие аналитические функции деформации
def E_total(a):
    return a * I2_base + (1/a) * I4_base + (a**3) * alpha_ideal * I0_base

def dynamic_alpha(a):
    return (I4_base/a - I2_base*a) / (3 * I0_base * a**3)

energies = E_total(a_vals)
dyn_alphas = dynamic_alpha(a_vals)

# =====================================================================
# АНАЛИТИКА: ИЩЕМ ГРАНИЦУ РАЗРУШЕНИЯ
# =====================================================================
E_min = np.min(energies)
min_idx = np.argmin(energies)

# Граница смерти: когда упругость хвоста разрывает жесткость ядра
# I4 / a == I2 * a  =>  a^2 = I4 / I2
a_death = np.sqrt(I4_base / I2_base)
E_death = E_total(a_death)

print(f"Центр Аттрактора (Идеал): a = 1.0000, α = {alpha_ideal:.6f}")
print(f"Энергия в центре        : {E_min:.6f} (Безразмерных единиц)\n")

print(f"ГРАНИЦА РАЗРУШЕНИЯ (СМЕРТЬ ЭЛЕКТРОНА):")
print(f"Критическое растяжение  : a = {a_death:.6f}")
print(f"Увеличение радиуса на   : {(a_death - 1)*100:.3f}%")
print(f"Динамическая α в этой т.: 0.000000")
print("Вывод: Парковочный слот феноменально узок! Вакуум может растянуть")
print("ядро электрона всего на ~0.4%, прежде чем оно аннигилирует.")
print("="*65)

# =====================================================================
# ВИЗУАЛИЗАЦИЯ
# =====================================================================
plt.style.use('dark_background')
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
fig.suptitle('Строгое топологическое дыхание Электрона (Аналитика)', fontsize=16)

# График 1: Энергетическая Яма
ax1.plot(dyn_alphas, energies, color='cyan', lw=3, label='Энергетический профиль аттрактора')
ax1.scatter([alpha_ideal], [E_min], color='yellow', s=120, zorder=5, label=f'Идеал ($\\alpha$ = {alpha_ideal:.5f})')
ax1.scatter([0], [E_death], color='red', s=120, zorder=5, label='Сингулярность ($\\alpha$ = 0)')

# Закраска парковочного слота
valid_idx = np.where(dyn_alphas >= 0)[0]
ax1.fill_between(dyn_alphas[valid_idx], E_min*0.99, energies[valid_idx], color='yellow', alpha=0.2, label='Зона жизни (Хопфион стабилен)')

ax1.set_title('Бассейн Аттрактора Меры (Энергия vs $\\alpha$)')
ax1.set_xlabel('Динамическая Мера ($\\alpha$)')
ax1.set_ylabel('Полная топологическая энергия')
ax1.set_xlim(-0.001, 0.015)
ax1.set_ylim(11.621, 11.623)
ax1.grid(color='white', alpha=0.1)
ax1.legend()

# График 2: Битва Интегралов
I4_dyn = I4_base / a_vals
I2_dyn = I2_base * a_vals

ax2.plot(a_vals, I4_dyn, color='magenta', lw=2.5, label='Жесткость ядра ($\\mathcal{I}_4/a$)')
ax2.plot(a_vals, I2_dyn, color='cyan', lw=2.5, label='Упругость хвоста ($\\mathcal{I}_2 \\cdot a$)')
ax2.axvline(1.0, color='yellow', linestyle='--', label='Идеальный радиус (a=1)')
ax2.axvline(a_death, color='red', linestyle='-', lw=2, label=f'Разрыв (a={a_death:.4f})')

ax2.set_title('Граница смерти: Натяжение разрывает Узел')
ax2.set_xlabel('Масштабный фактор "дыхания" ($a$)')
ax2.set_ylabel('Значение интегралов')
ax2.grid(color='white', alpha=0.1)
ax2.legend()

# Плашка с выводом
fig.text(0.5, 0.03, "Физический вывод: Электрон находится в жесточайшем резонансе.\nТопологическое ядро может деформироваться лишь на ~0.4%, прежде чем кулоновский хвост разорвет узел.", 
         ha='center', fontsize=12, bbox=dict(facecolor='black', alpha=0.8, edgecolor='cyan'))

plt.tight_layout(rect=[0, 0.08, 1, 0.95])
plt.show()