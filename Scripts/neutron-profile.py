import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import simpson

print("--- АНАЛИТИЧЕСКИЙ РАЗРЕЗ НЕЙТРОНА ---")

# =====================================================================
# 1. ФИЗИЧЕСКИЕ КОНСТАНТЫ (в МэВ)
# =====================================================================
mass_proton = 938.272     # Базовая масса гладкого трилистника
mass_defect = 1.293       # Избыточная энергия фрустрации (Tw = 0.5)
mass_electron = 0.511     # Энергия, уходящая на формирование хопфиона
energy_neutrino = 0.782   # Энергия, уходящая в фазовую волну (нейтрино + кинетика)

# =====================================================================
# 2. МОДЕЛИРОВАНИЕ ПРОДОЛЬНОГО ПРОФИЛЯ
# =====================================================================
# Параметр t - координата вдоль оси трубки узла (от 0 до 2*pi)
t = np.linspace(0, 2 * np.pi, 2000)

# Базовая плотность энергии протона (идеально симметричная, ровная линия)
# Интеграл от E_base по dt от 0 до 2pi даст mass_proton
E_base = np.full_like(t, mass_proton / (2 * np.pi))

# Плотность энергии топологического дефекта (локальный кинк)
# Моделируем как распределение Гаусса с центром в t = pi
defect_center = np.pi
sigma = 0.15 # Ширина дефекта
E_defect = (mass_defect / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((t - defect_center) / sigma)**2)

# Полная плотность энергии нейтрона
E_neutron = E_base + E_defect

# Проверка интегралов (Доказательство сохранения массы)
int_proton = simpson(E_base, x=t)
int_defect = simpson(E_defect, x=t)
int_total = simpson(E_neutron, x=t)

print(f"Интеграл каркаса (Протон) : {int_proton:.3f} МэВ")
print(f"Интеграл дефекта          : {int_defect:.3f} МэВ")
print(f"Полная масса Нейтрона     : {int_total:.3f} МэВ")

# =====================================================================
# 3. ВИЗУАЛИЗАЦИЯ (ИНФОГРАФИКА ДЛЯ МОНОГРАФИИ)
# =====================================================================
plt.style.use('dark_background')
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
fig.suptitle('Энергетический профиль Нейтрона (1D развертка вдоль оси Трилистника)', fontsize=16, y=0.98)

# ---------------------------------------------------------------------
# ГРАФИК 1: Глобальная картина (Нейтрон целиком)
# ---------------------------------------------------------------------
ax1.plot(t, E_neutron, color='coral', lw=2, label='Плотность энергии Нейтрона')
ax1.plot(t, E_base, color='cyan', lw=2, linestyle='--', label='Базовый уровень Протона')

ax1.fill_between(t, 0, E_base, color='cyan', alpha=0.1)
ax1.fill_between(t, E_base, E_neutron, color='coral', alpha=0.6)

ax1.set_title(f'Глобальный масштаб\n(База {mass_proton:.1f} МэВ + Дефект {mass_defect:.3f} МэВ)')
ax1.set_xlabel('Координата вдоль оси трубки $t$ (радианы)')
ax1.set_ylabel('Плотность энергии $dE/dt$ (МэВ / рад)')
ax1.set_xlim(0, 2 * np.pi)
ax1.set_ylim(0, 160)
ax1.set_xticks([0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi])
ax1.set_xticklabels(['0', r'$\pi/2$', r'$\pi$', r'$3\pi/2$', r'$2\pi$'])
ax1.grid(color='white', alpha=0.1)
ax1.legend(loc='lower right')

# Добавляем стрелку, указывающую на дефект
ax1.annotate('Вплетенный\nХопфион', xy=(np.pi, 151), xytext=(np.pi - 1, 155),
             arrowprops=dict(facecolor='yellow', shrink=0.05, width=1, headwidth=6),
             color='yellow', fontsize=10)

# ---------------------------------------------------------------------
# ГРАФИК 2: Макро-ЗУМ (Анатомия Бета-распада)
# ---------------------------------------------------------------------
# Смотрим только на дефект, вычитая базу протона
ax2.plot(t, E_defect, color='magenta', lw=3, label='Энергия фрустрации ($\\Delta m$)')

# Разделяем площадь дефекта на Электрон и Нейтрино по высоте
# Ищем высоту, при которой площадь под кривой (интеграл) равна массе электрона 0.511
# Для визуальной простоты мы закрасим центральную самую плотную часть под электрон
electron_threshold = np.max(E_defect) * (energy_neutrino / mass_defect)

ax2.fill_between(t, 0, E_defect, where=(E_defect > electron_threshold), 
                 color='cyan', alpha=0.8, label=f'Электрон ($e^-$): {mass_electron:.3f} МэВ\n(Зародыш Хопфиона)')
ax2.fill_between(t, 0, E_defect, where=(E_defect <= electron_threshold), 
                 color='yellow', alpha=0.4, label=f'Антинейтрино ($\\bar{{\\nu}}_e$): {energy_neutrino:.3f} МэВ\n(Фазовая волна + кинетика)')

ax2.set_title('Анатомия Дефекта: Подготовка к $\\beta$-распаду')
ax2.set_xlabel('Координата вокруг локального излома $t$')
ax2.set_ylabel('Избыточная плотность энергии (МэВ / рад)')
ax2.set_xlim(np.pi - 0.6, np.pi + 0.6)
ax2.set_ylim(0, np.max(E_defect) * 1.1)
ax2.grid(color='white', alpha=0.1)
ax2.legend(loc='upper right', fontsize=10)

# Подпись механизма
ax2.text(np.pi - 0.55, np.max(E_defect) * 0.5, 
         "Когда пружина лопнет\n(топологическое туннелирование),\nГолубая зона замкнется в тор (электрон),\nЖелтая зона разлетится как рябь (нейтрино).", 
         fontsize=9, color='white', bbox=dict(facecolor='black', alpha=0.6, edgecolor='gray'))

plt.tight_layout()
plt.show()

print("--- ЗАВЕРШЕНО ---")
