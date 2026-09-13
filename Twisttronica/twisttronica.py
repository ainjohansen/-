import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# Установка красивого стиля для научных публикаций
plt.rcParams.update({
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 14,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'figure.titlesize': 14,
    'font.family': 'sans-serif'
})

# =====================================================================
# РИСУНОК 1: Коллапс скорости Ферми и расходимость массы (Раздел 3 и 4)
# =====================================================================
fig1, ax1 = plt.subplots(figsize=(7, 5))

theta = np.linspace(0.8, 1.5, 500)
theta_m = 1.0822
m_0 = 0.08

# Расчет v_F* / v_F и m* / m_e
v_f_ratio = np.abs(1 - (theta_m / theta)**2)
# Ограничим расходимость массы для адекватности графика
m_star = m_0 / np.clip(np.abs(1 - (theta_m / theta)**2), 0.01, None)

color = 'tab:blue'
ax1.set_xlabel(r'Угол поворота $\theta$ (градусы)')
ax1.set_ylabel(r'Скорость Ферми $v_F^* / v_F$', color=color)
line1 = ax1.plot(theta, v_f_ratio, color=color, lw=2.5, label=r'Скорость Ферми $v_F^*$')
ax1.tick_params(axis='y', labelcolor=color)
ax1.grid(True, linestyle='--', alpha=0.5)

ax2 = ax1.twinx()  
color = 'tab:red'
ax2.set_ylabel(r'Эффективная масса $m^* / m_e$', color=color)
line2 = ax2.plot(theta, m_star, color=color, lw=2.5, linestyle='--', label=r'Эффективная масса $m^*$')
ax2.tick_params(axis='y', labelcolor=color)

# Линия магического угла
ax1.axvline(x=theta_m, color='black', linestyle=':', alpha=0.7, lw=1.5)
ax1.text(theta_m + 0.02, 0.4, f'$\\theta_m = {theta_m:.2f}^\\circ$\n(BPS-текучесть)', 
         fontweight='bold', bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))

plt.title('Коллапс фазового потока на границе текучести')
fig1.tight_layout()
plt.savefig('fig1_flat_band_collapse.png', dpi=300)
plt.close()


# =====================================================================
# РИСУНОК 2: Критический гетерострейн и сдвиг магического угла (Раздел 6)
# =====================================================================
fig2, ax = plt.subplots(figsize=(7, 5))

strain = np.linspace(0, 0.25, 300)  # в процентах
eps_crit = 0.2013  # порог прочности по Губеру-фон Мизесу

# Формула анизотропного сдвига угла для разных направлений деформации (phi)
def theta_m_strain(eps, phi):
    val = 1 - (eps / eps_crit) * np.cos(2 * phi) - 0.5 * (eps / eps_crit)**2
    val = np.clip(val, 0, None)  # Физический предел разрушения зоны
    return theta_m * np.sqrt(val)

ax.plot(strain, theta_m_strain(strain, 0), label=r'Натяжение вдоль оси ($\phi = 0$)', lw=2.5, color='darkred')
ax.plot(strain, theta_m_strain(strain, np.pi/4), label=r'Диагональное ($\phi = \pi/4$)', lw=2, color='darkorange', linestyle='--')
ax.plot(strain, theta_m_strain(strain, np.pi/2), label=r'Поперечное ($\phi = \pi/2$)', lw=2.5, color='darkblue')

# Выделение зоны разрушения
ax.axvspan(eps_crit, 0.25, color='gray', alpha=0.15, label='Пластическое разрушение муара')
ax.axvline(x=eps_crit, color='red', linestyle=':', lw=1.5)
ax.text(eps_crit - 0.04, 0.2, f'$\\epsilon_{{crit}} = {eps_crit:.2f}\\%$', color='red', rotation=90, fontweight='bold')

ax.set_xlabel(r'Одноосное натяжение $\epsilon$ (%)')
ax.set_ylabel(r'Перенормированный магический угол $\theta_m(\epsilon)$')
ax.set_title('Разрушение плоских зон под действием гетерострейна')
ax.set_xlim(0, 0.25)
ax.set_ylim(0, 1.2)
ax.grid(True, linestyle='--', alpha=0.5)
ax.legend(loc='lower left')

fig2.tight_layout()
plt.savefig('fig2_heterostrain_limits.png', dpi=300)
plt.close()


# =====================================================================
# РИСУНОК 3: Спектральное дерево многослойных систем Чебышёва (Раздел 7)
# =====================================================================
fig3, ax = plt.subplots(figsize=(7, 5))

# Данные расчетов Чебышёва
layers = [2, 3, 4, 5]
# Карта углов для каждого N
angles_theory = {
    2: [1.082],
    3: [1.530],
    4: [1.751, 0.669],
    5: [1.872, 1.082]
}

# Экспериментальные точки для сопоставления
exp_data = [
    (2, 1.08, 'Bi-layer (Cao 2018)'),
    (3, 1.53, 'Tri-layer (Park 2021)'),
    (4, 1.75, '4-layer (Exp)'),
    (4, 0.67, '4-layer (Exp)'),
]

# Отрисовка теоретических веток
for n in layers:
    for angle in angles_theory[n]:
        ax.scatter(n, angle, color='blue', s=120, zorder=3, edgecolors='darkblue', alpha=0.8)
        ax.text(n + 0.08, angle - 0.03, f'{angle:.2f}$^\\circ$', fontsize=10, color='blue')

# Отрисовка экспериментальных точек
for n, angle, label in exp_data:
    ax.scatter(n, angle, color='red', marker='x', s=100, linewidths=2.5, zorder=4)

# Косметические линии тренда ("дерево муара")
ax.plot([2, 3, 4, 5], [1.082, 1.530, 1.751, 1.872], color='blue', linestyle='-', alpha=0.3, lw=2, label='Главная Чебышёвская мода')
ax.plot([4, 5], [0.669, 1.082], color='indigo', linestyle='-', alpha=0.3, lw=2, label='Высшие гармоники сдвига')

ax.axhline(y=2.164, color='black', linestyle='-.', alpha=0.5, lw=1.5)
ax.text(2.2, 2.20, r'Предел бесконечного твиста $\theta_m^{(\infty)} = 2.16^\circ$', color='black', fontsize=10)
ax.set_xlabel('Количество слоев в стопке ($N$)')
ax.set_ylabel('Магические углы (градусы)')
ax.set_title('Спектр Чебышёва для мультиплетов твистроники')
ax.set_xticks(layers)
ax.set_xlim(1.5, 5.8)
ax.set_ylim(0.4, 2.4)
ax.grid(True, linestyle='--', alpha=0.3)

legend_elements = [
    Line2D([0], [0], marker='o', color='w', label='Теория эластодинамики', markerfacecolor='blue', markersize=10),
    Line2D([0], [0], marker='x', color='red', label='Эксперимент (MIT/Harvard)', markersize=10, linestyle='None', markeredgewidth=2.5),
]
ax.legend(handles=legend_elements, loc='lower left')

fig3.tight_layout()
plt.savefig('fig3_chebyshev_multiplets.png', dpi=300)
plt.close()

print("Успешно сгенерировано 3 графика: fig1_flat_band_collapse.png, fig2_heterostrain_limits.png, fig3_chebyshev_multiplets.png")
