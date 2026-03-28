import numpy as np
from scipy.integrate import solve_bvp, quad
import matplotlib.pyplot as plt

# =====================================================================
# 1. ЗАДАЕМ КОНСТАНТЫ И УРАВНЕНИЕ ПОЛЯ
# =====================================================================
# Альфа - это глобальное свойство вакуума
alpha_exact = 0.0072973525693

def ode_system(r, y):
    """
    Система дифференциальных уравнений для BVP солвера.
    Точный вывод из вариационного принципа для I2, I4, I0.
    """
    f, fp = y
    sin_f = np.sin(f)
    cos_f = np.cos(f)
    sin2 = sin_f**2
    
    # ИСПРАВЛЕНО: добавлена двойка перед sin2
    denom = r**2 + 2 * sin2
    
    # ИСПРАВЛЕНО: строгие коэффициенты Эйлера-Лагранжа
    term1 = 2 * sin_f * cos_f * (1 - fp**2 + sin2/r**2)
    term2 = (alpha_exact / 2) * r**2 * sin_f  # ИСПРАВЛЕНО: деление на 2
    term3 = -2 * r * fp
    
    fpp = (term1 + term2 + term3) / denom
    return np.vstack((fp, fpp))

def boundary_conditions(ya, yb):
    """
    ya - состояние в центре ядра (r -> 0)
    yb - состояние на краю Вселенной (r -> R)
    """
    # f(0) должно быть pi (вывернутый вакуум)
    # f(R) должно быть 0 (чистый вакуум)
    return np.array([ya[0] - np.pi, yb[0] - 0.0])

# =====================================================================
# 2. ИНИЦИАЛИЗАЦИЯ ПОЛЯ (РЕЛАКСАЦИЯ ИЗ АНЗАЦА)
# =====================================================================
print("--- ГЛОБАЛЬНАЯ РЕЛАКСАЦИЯ ПОЛЯ (BVP) ---")
print("Создаем начальное приближение (анзац) и позволяем полю расслабиться...")

# Задаем сетку от центра до глубокого хвоста
r_min = 1e-4
r_max = 120.0
r_nodes = np.logspace(np.log10(r_min), np.log10(r_max), 1000)

# Наш старый добрый анзац служит идеальным стартовым шаблоном!
A_guess = 0.8168
B_guess = np.sqrt(alpha_exact)

u_guess = (A_guess / r_nodes) * np.exp(-B_guess * r_nodes)
f_guess = 2 * np.arctan(u_guess)
fp_guess = (2 * u_guess / (1 + u_guess**2)) * (-1/r_nodes - B_guess)

y_guess = np.vstack((f_guess, fp_guess))

# =====================================================================
# 3. РЕШЕНИЕ УРАВНЕНИЯ
# =====================================================================
# solve_bvp найдет точный профиль, удовлетворяющий ОДУ на всем пространстве
sol = solve_bvp(ode_system, boundary_conditions, r_nodes, y_guess, tol=1e-8, max_nodes=50000)

if sol.success:
    print("\nПоле успешно релаксировало в состояние минимальной энергии!")
    
    # =====================================================================
    # 4. ВЫЧИСЛЕНИЕ ИНТЕГРАЛОВ ИЗ ТОЧНОГО РЕШЕНИЯ
    # =====================================================================
    def I2_integrand(r):
        y = sol.sol(r)
        return r**2 * y[1]**2 + 2 * np.sin(y[0])**2

    def I4_integrand(r):
        y = sol.sol(r)
        sin_f = np.sin(y[0])
        sin_f_r = sin_f / r if r > 1e-8 else -y[1]
        return sin_f**2 * (2 * y[1]**2 + sin_f_r**2)

    def I0_integrand(r):
        y = sol.sol(r)
        return (1 - np.cos(y[0])) * r**2

    I2, _ = quad(I2_integrand, r_min, r_max, limit=500)
    I4, _ = quad(I4_integrand, r_min, r_max, limit=500)
    I0, _ = quad(I0_integrand, r_min, r_max, limit=500)
    
    alpha_check = (I4 - I2) / (3 * I0)
    
    print("\n" + "="*45)
    print(" АНАЛИТИКА ТОЧНОГО ТОПОЛОГИЧЕСКОГО ПРОФИЛЯ")
    print("="*45)
    print(f" Упругость поля (I2): {I2:.6f}")
    print(f" Жесткость ядра (I4): {I4:.6f}")
    print(f" Масса хвоста (I0)  : {I0:.6f}")
    print(f" Дельта (I4 - I2)   : {(I4 - I2):.6f}")
    print("-" * 45)
    print(f" Исходная Альфа вакуума : {alpha_exact:.10f}")
    print(f" Альфа из баланса (ОДУ) : {alpha_check:.10f}")
    print(f" Погрешность Деррика    : {abs(alpha_check - alpha_exact):.2e}")
    print("="*45)

    # =====================================================================
    # 5. ВИЗУАЛИЗАЦИЯ
    # =====================================================================
    r_plot = np.linspace(0.001, 8.0, 1000)
    y_plot = sol.sol(r_plot)
    f_vals = y_plot[0]
    fp_vals = y_plot[1]
    
    sin_f = np.sin(f_vals)
    sin_f_r = np.where(r_plot < 1e-8, -fp_vals, sin_f / r_plot)
    
    dens_I2 = r_plot**2 * fp_vals**2 + 2 * sin_f**2
    dens_I4 = sin_f**2 * (2 * fp_vals**2 + sin_f_r**2)
    dens_I0 = (1 - np.cos(f_vals)) * r_plot**2
    
    plt.style.use('dark_background')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(f'Строгое ОДУ-решение: Электрон-Хопфион', fontsize=16, y=0.98)

    ax1.plot(r_plot, f_vals, color='cyan', lw=2.5, label='Фаза $f(\\rho)$')
    ax1.axhline(np.pi, color='magenta', linestyle='--', alpha=0.5, label='$\\pi$ (Ядро)')
    ax1.axhline(0, color='gray', linestyle='--', alpha=0.5, label='$0$ (Вакуум)')
    ax1.set_title('Профиль фазового следа')
    ax1.set_xlabel('Безразмерное расстояние ($\\rho = r/a$)')
    ax1.set_ylabel('Фаза (радианы)')
    ax1.set_xlim(0, 8)
    ax1.legend()
    ax1.grid(color='white', alpha=0.1)

    ax2.plot(r_plot, dens_I4, color='magenta', lw=2.5, label='$\\mathcal{I}_4$: Жесткость (Скрутка)')
    ax2.plot(r_plot, dens_I2, color='cyan', lw=2.5, label='$\\mathcal{I}_2$: Упругость (Натяжение)')
    ax2.plot(r_plot, 3 * alpha_exact * dens_I0, color='yellow', lw=2, linestyle='--', label='$3\\alpha\\mathcal{I}_0$: Масса хвоста')
    
    ax2.set_title('Идеальный динамический резонанс')
    ax2.set_xlabel('Безразмерное расстояние ($\\rho = r/a$)')
    ax2.set_ylabel('Плотность энергии')
    ax2.set_xlim(0, 4)
    ax2.legend()
    ax2.grid(color='white', alpha=0.1)

    plt.tight_layout()
    plt.show()

else:
    print("\nОшибка: Поле не смогло релаксировать.")
    print(sol.message)
