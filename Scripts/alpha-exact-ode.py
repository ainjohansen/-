import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import root
import matplotlib.pyplot as plt

# ------------------------------------------------------------
# 1. ОДУ И ИНТЕГРАЛЫ (5 уравнений в одной системе)
# ------------------------------------------------------------
def ode_system(r, y, alpha):
    # y = [f, fp, I2, I4, I0]
    f, fp, i2, i4, i0 = y
    
    # Защита от деления на ноль
    r_safe = max(r, 1e-15)
    
    sin_f = np.sin(f)
    cos_f = np.cos(f)
    sin2 = sin_f * sin_f
    
    # 1. Вычисление второй производной f'' (Уравнение Эйлера-Лагранжа)
    denom = r_safe*r_safe + sin2
    term1 = sin_f * cos_f * (1 - fp*fp + sin2/(r_safe*r_safe))
    term2 = alpha * r_safe*r_safe * sin_f
    term3 = -2 * r_safe * fp
    fpp = (term1 + term2 + term3) / denom
    
    # 2. Вычисление подынтегральных плотностей энергий
    # Предел sin(f)/r при r->0. Так как f ~ pi + fp*r, sin(f) ~ -fp*r
    sin_f_over_r = -fp if r < 1e-8 else sin_f / r_safe
    
    di2 = r_safe**2 * fp**2 + 2 * sin2
    di4 = sin2 * (2 * fp**2 + sin_f_over_r**2)
    di0 = (1 - cos_f) * r_safe**2

    return[fp, fpp, di2, di4, di0]

# ------------------------------------------------------------
# 2. ФУНКЦИЯ ПРИСТРЕЛКИ (SHOOTING)
# ------------------------------------------------------------
def shoot(params, r0=1e-6, R=150.0):
    s, alpha = params
    # Начальные условия: f(0)=pi, f'(0)=s. Интегралы стартуют с 0.
    y0 =[np.pi + s*r0, s, 0.0, 0.0, 0.0]
    
    sol = solve_ivp(ode_system, [r0, R], y0, args=(alpha,),
                    method='LSODA', rtol=1e-11, atol=1e-13)
    if not sol.success:
        return[np.nan, np.nan]
    
    # Возвращаем значения f(R) и f'(R) на краю Вселенной (они должны быть нулями)
    return [sol.y[0, -1], sol.y[1, -1]]

# ------------------------------------------------------------
# 3. ПОИСК ИДЕАЛЬНОГО РЕШЕНИЯ
# ------------------------------------------------------------
# s - это f'(0), то есть отрицательный наклон в центре
s_guess = -7.56267
alpha_guess = 0.00729735

print("--- ТОЧНОЕ ОДУ-МОДЕЛИРОВАНИЕ ЭЛЕКТРОНА ---")
print(f"Старт: s = {s_guess:.5f}, α = {alpha_guess:.8f}")
print("Ищем идеальный корень методом Левенберга-Марквардта (hybr)...")

# Увеличиваем R до 150, чтобы собрать весь хвост I0
R_calc = 150.0 
res = root(shoot,[s_guess, alpha_guess], args=(1e-6, R_calc), method='hybr', tol=1e-11)

if res.success:
    s_opt, alpha_opt = res.x
    
    # Делаем финальный "золотой" прогон для получения всех данных
    y0 =[np.pi + s_opt*1e-6, s_opt, 0.0, 0.0, 0.0]
    sol = solve_ivp(ode_system,[1e-6, R_calc], y0, args=(alpha_opt,),
                    method='LSODA', rtol=1e-12, atol=1e-14, dense_output=True)
    
    # Точные значения интегралов берем прямо из решателя ОДУ
    I2_final = sol.y[2, -1]
    I4_final = sol.y[3, -1]
    I0_final = sol.y[4, -1]
    alpha_check = (I4_final - I2_final) / (3 * I0_final)
    
    print("\n" + "="*45)
    print(" РЕЗУЛЬТАТЫ СТРОГОГО РЕШЕНИЯ УРАВНЕНИЯ ПОЛЯ")
    print("="*45)
    print(f"Наклон в центре (s) : {s_opt:.8f}")
    print(f"Собственная Альфа   : {alpha_opt:.10f}")
    print(f"Обратная Альфа (1/α): {1/alpha_opt:.4f}")
    print("-" * 45)
    print(" АНАЛИТИКА ИНТЕГРАЛОВ (с точностью LSODA):")
    print(f" Упругость поля (I2): {I2_final:.6f}")
    print(f" Жесткость ядра (I4): {I4_final:.6f}")
    print(f" Масса хвоста (I0)  : {I0_final:.6f}")
    print(f" Дельта (I4 - I2)   : {(I4_final - I2_final):.6f}")
    print("-" * 45)
    print(f" Проверка баланса   : {alpha_check:.10f}")
    print(f" Погрешность ОДУ    : {abs(alpha_check - alpha_opt):.2e}")
    print("="*45)

    # ------------------------------------------------------------
    # 4. НАГЛЯДНАЯ ВИЗУАЛИЗАЦИЯ
    # ------------------------------------------------------------
    # Для графиков берем логарифмически-линейную сетку, чтобы показать ядро
    r_plot = np.linspace(0.001, 8.0, 1000)
    
    # Достаем функции из dense_output
    state = sol.sol(r_plot)
    f_vals = state[0]
    fp_vals = state[1]
    
    # Считаем плотности для графиков
    sin_f = np.sin(f_vals)
    sin_f_over_r = np.where(r_plot < 1e-8, -fp_vals, sin_f / r_plot)
    dens_I2 = r_plot**2 * fp_vals**2 + 2 * sin_f**2
    dens_I4 = sin_f**2 * (2 * fp_vals**2 + sin_f_over_r**2)
    dens_I0 = (1 - np.cos(f_vals)) * r_plot**2
    
    plt.style.use('dark_background')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(f'Строгое ОДУ-решение Электрона ($\\alpha$ = {alpha_opt:.6f})', fontsize=16, y=0.98)

    # График 1: Фазовый профиль
    ax1.plot(r_plot, f_vals, color='cyan', lw=2.5, label='Фаза $f(\\rho)$')
    ax1.axhline(np.pi, color='magenta', linestyle='--', alpha=0.5, label='$\\pi$ (Ядро)')
    ax1.axhline(0, color='gray', linestyle='--', alpha=0.5, label='$0$ (Вакуум)')
    ax1.set_title('Профиль фазового следа')
    ax1.set_xlabel('Безразмерное расстояние ($\\rho = r/a$)')
    ax1.set_ylabel('Фаза (радианы)')
    ax1.set_xlim(0, 8)
    ax1.legend()
    ax1.grid(color='white', alpha=0.1)

    # График 2: Баланс энергий (Плотности интегралов)
    ax2.plot(r_plot, dens_I4, color='magenta', lw=2.5, label='$\\mathcal{I}_4$: Жесткость (Скрутка)')
    ax2.plot(r_plot, dens_I2, color='cyan', lw=2.5, label='$\\mathcal{I}_2$: Упругость (Натяжение)')
    ax2.plot(r_plot, 3 * alpha_opt * dens_I0, color='yellow', lw=2, linestyle='--', 
             label='$3\\alpha\\mathcal{I}_0$: Масса хвоста')
    
    ax2.set_title('Идеальный динамический резонанс')
    ax2.set_xlabel('Безразмерное расстояние ($\\rho = r/a$)')
    ax2.set_ylabel('Плотность энергии')
    ax2.set_xlim(0, 4)
    ax2.legend()
    ax2.grid(color='white', alpha=0.1)

    plt.tight_layout()
    plt.show()

else:
    print("Корень не найден:", res.message)
