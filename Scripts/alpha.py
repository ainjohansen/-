import numpy as np
from scipy.integrate import quad
from scipy.optimize import root_scalar
import matplotlib.pyplot as plt

# =====================================================================
# 1. МОДЕЛЬ ПРОФИЛЯ ЭЛЕКТРОНА
# =====================================================================
def profile(rho, A, B):
    # Защита от деления на ноль для массивов numpy
    rho = np.maximum(rho, 1e-12)
    u = (A / rho) * np.exp(-B * rho)
    f = 2 * np.arctan(u)
    df = (2 * u / (1 + u**2)) * (-1/rho - B)
    return f, df

def integrand_I2(rho, A, B):
    f, df = profile(rho, A, B)
    return (rho**2 * df**2 + 2 * np.sin(f)**2)

def integrand_I4(rho, A, B):
    f, df = profile(rho, A, B)
    sin_f = np.sin(f)
    # Векторизованное условие для избежания сингулярности в 0
    sin_f_over_rho = np.where(rho < 1e-8, 2/A, sin_f / rho)
    return (sin_f**2) * (2 * df**2 + sin_f_over_rho**2)

def integrand_I0(rho, A, B):
    f, df = profile(rho, A, B)
    return (1 - np.cos(f)) * rho**2

# =====================================================================
# 2. ВЫЧИСЛИТЕЛЬНОЕ ЯДРО
# =====================================================================
def calculate_alpha(A, B):
    limit = 150.0 / max(B, 1e-4) # Надежный предел для бесконечного хвоста
    
    I2, _ = quad(integrand_I2, 0, limit, args=(A, B), limit=200)
    I4, _ = quad(integrand_I4, 0, limit, args=(A, B), limit=200)
    I0, _ = quad(integrand_I0, 0, limit, args=(A, B), limit=200)
    
    alpha = (I4 - I2) / (3 * I0)
    return alpha, I2, I4, I0

# =====================================================================
# 3. ПОИСК ФОРМЫ УЗЛА ПОД ДИКТАТ ВАКУУМА
# =====================================================================
# Экспериментальная константа (свойства среды)
alpha_exp = 1 / 137.036
B_target = np.sqrt(alpha_exp)

def objective(A):
    alpha_calc, _, _, _ = calculate_alpha(A, B_target)
    return alpha_calc - alpha_exp

print("--- СТАРТ РЕЗОНАНСНОГО ПОИСКА ---")
print(f"Вакуум требует хвост B = sqrt(1/137) = {B_target:.5f}")
print("Ищем идеальную топологическую толщину ядра (A)...")

res = root_scalar(objective, bracket=[0.2, 2.0], method='brentq')

if res.converged:
    A_final = res.root
    alpha_final, I2, I4, I0 = calculate_alpha(A_final, B_target)
    
    print("\n--- ИТОГИ РЕЗОНАНСА ---")
    print(f"Идеальная толщина узла (A) : {A_final:.5f}")
    print(f"Вычисленная Альфа          : {alpha_final:.6f}")
    print(f"Экспериментальная Альфа    : {alpha_exp:.6f}")
    print("-" * 30)
    print(f"Интеграл упругости I2      : {I2:.4f}")
    print(f"Интеграл жесткости I4      : {I4:.4f}")
    print(f"Объемный хвост I0          : {I0:.4f}")
    print(f"Дельта ядра (I4 - I2)      : {(I4 - I2):.4f}")
    print("\nВывод: Топологическое ядро успешно адаптировалось под упругость Вселенной.")
    
    # =====================================================================
    # 4. ВИЗУАЛИЗАЦИЯ (АНАТОМИЯ ЭЛЕКТРОНА)
    # =====================================================================
    print("\nГенерация графиков...")
    
    # Создаем массив координат (от центра ядра до далекого хвоста)
    rho_vals = np.linspace(0.001, 15, 1000)
    
    # Получаем функции
    f_vals, _ = profile(rho_vals, A_final, B_target)
    dens_I2 = integrand_I2(rho_vals, A_final, B_target)
    dens_I4 = integrand_I4(rho_vals, A_final, B_target)
    dens_I0 = integrand_I0(rho_vals, A_final, B_target)
    
    plt.style.use('dark_background') # Темный фон выглядит более "космически"
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(f'Анатомия Электрона-Хопфиона (α = 1/137, A = {A_final:.4f})', fontsize=16, y=0.98)

    # График 1: Фазовый профиль
    ax1.plot(rho_vals, f_vals, color='cyan', lw=2, label='Фазовый угол $f(\\rho)$')
    ax1.axhline(np.pi, color='magenta', linestyle='--', alpha=0.5, label='$\pi$ (Сингулярность свернута)')
    ax1.axhline(0, color='gray', linestyle='--', alpha=0.5, label='$0$ (Чистый вакуум)')
    ax1.axvline(1.0, color='yellow', linestyle=':', label='Радиус ядра ($\\rho=1$)')
    ax1.set_title('Профиль фазового следа')
    ax1.set_xlabel('Безразмерное расстояние ($\\rho = r/a$)')
    ax1.set_ylabel('Фаза (радианы)')
    ax1.set_xlim(0, 10)
    ax1.legend()
    ax1.grid(color='white', alpha=0.1)

    # График 2: Баланс энергий (Плотности интегралов)
    ax2.plot(rho_vals, dens_I4, color='magenta', lw=2.5, label='$\\mathcal{I}_4$: Жесткость ядра (Скрутка)')
    ax2.plot(rho_vals, dens_I2, color='cyan', lw=2.5, label='$\\mathcal{I}_2$: Упругость поля (Натяжение)')
    # Масштабируем I0, чтобы показать его вклад в уравнение баланса (3 * alpha * I0)
    ax2.plot(rho_vals, 3 * alpha_exp * dens_I0, color='yellow', lw=2, linestyle='--', 
             label='$3\\alpha\\mathcal{I}_0$: Масса хвоста (Компенсатор)')
    
    ax2.set_title('Локальный дисбаланс $\\rightarrow$ Глобальный резонанс')
    ax2.set_xlabel('Безразмерное расстояние ($\\rho = r/a$)')
    ax2.set_ylabel('Плотность энергии (подынтегральное выражение)')
    ax2.set_xlim(0, 6) # Приближаем к ядру, чтобы увидеть пересечение
    ax2.legend()
    ax2.grid(color='white', alpha=0.1)

    plt.tight_layout()
    plt.show()

else:
    print("Резонанс не найден.")
