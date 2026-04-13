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

# ============================================================
# 6. ПОГРУЖЕНИЕ ЭТАЛОННОГО ЭЛЕКТРОНА В ДРУГИЕ ВСЕЛЕННЫЕ
# ============================================================
if sol.success:
    print("\nПоле успешно релаксировало. Вычисляем эталонные интегралы...")
    # Эталонные интегралы (a=1) из исходного решения
    def I2_int(r): 
        y = sol.sol(r)
        return r**2 * y[1]**2 + 2 * np.sin(y[0])**2
    def I4_int(r):
        y = sol.sol(r)
        sin_f = np.sin(y[0])
        sin_f_r = sin_f / r if r > 1e-8 else -y[1]
        return sin_f**2 * (2 * y[1]**2 + sin_f_r**2)
    def I0_int(r):
        y = sol.sol(r)
        return (1 - np.cos(y[0])) * r**2

    I2_ref, _ = quad(I2_int, r_min, r_max, limit=200)
    I4_ref, _ = quad(I4_int, r_min, r_max, limit=200)
    I0_ref, _ = quad(I0_int, r_min, r_max, limit=200)
    print(f"Эталон: I2={I2_ref:.6f}, I4={I4_ref:.6f}, I0={I0_ref:.6f}")

    from scipy.interpolate import interp1d
    f_interp = interp1d(sol.x, sol.y[0], kind='cubic', fill_value='extrapolate')
    fp_interp = interp1d(sol.x, sol.y[1], kind='cubic', fill_value='extrapolate')

    def energy_scaled(a):
        def I2_int_a(r):
            r_scaled = r / a
            f_val = f_interp(r_scaled)
            fp_val = fp_interp(r_scaled) / a
            return r**2 * fp_val**2 + 2 * np.sin(f_val)**2
        def I4_int_a(r):
            r_scaled = r / a
            f_val = f_interp(r_scaled)
            fp_val = fp_interp(r_scaled) / a
            sin_f = np.sin(f_val)
            sin_f_r = sin_f / r if r > 1e-8 else -fp_val
            return sin_f**2 * (2 * fp_val**2 + sin_f_r**2)
        def I0_int_a(r):
            r_scaled = r / a
            f_val = f_interp(r_scaled)
            return (1 - np.cos(f_val)) * r**2

        I2, _ = quad(I2_int_a, r_min, r_max, limit=200)
        I4, _ = quad(I4_int_a, r_min, r_max, limit=200)
        I0, _ = quad(I0_int_a, r_min, r_max, limit=200)
        return I2 + I4 + 3 * alpha_exact * I0

    # Проверка при a=1
    E_ref = I2_ref + I4_ref + 3 * alpha_exact * I0_ref
    E_test = energy_scaled(1.0)
    print(f"Энергия при a=1: эталон {E_ref:.6f}, интерполяция {E_test:.6f}")

    # Расширенное сканирование
    a_vals = np.linspace(0.5, 1.5, 200)
    energies = []
    for a in a_vals:
        energies.append(energy_scaled(a))
        if abs(a - 1.0) < 0.01:
            print(f"a={a:.3f} E={energies[-1]:.6f}")

    # График E(a)
    plt.figure(figsize=(8,5))
    plt.plot(a_vals, energies, 'b-', lw=2)
    plt.axvline(1.0, color='r', linestyle='--', label='Наша Вселенная (a=1)')
    plt.xlabel('Масштабный фактор a')
    plt.ylabel('Полная энергия')
    plt.title('Энергия эталонного электрона при масштабировании')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig('energy_vs_scale.png', dpi=150)
    plt.show()

    # График E(α)
    alphas = alpha_exact / a_vals**2
    plt.figure(figsize=(8,5))
    plt.plot(alphas, energies, 'g-', lw=2)
    plt.axvline(alpha_exact, color='r', linestyle='--', label=f'α = {alpha_exact:.6f}')
    plt.xlabel('α')
    plt.ylabel('Полная энергия')
    plt.title('Энергия эталонного электрона в зависимости от α')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig('energy_vs_alpha.png', dpi=150)
    plt.show()

    # Сохранение данных
    np.savez('scaling_data.npz', a_vals=a_vals, energies=energies, alphas=alphas)
    print("Данные сохранены в 'scaling_data.npz'")

else:
    print("\nОшибка: Поле не смогло релаксировать.")
    print(sol.message)
