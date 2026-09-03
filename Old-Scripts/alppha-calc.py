import numpy as np
from scipy.integrate import solve_bvp, quad
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

# =================================================================
# УРАВНЕНИЯ ХОПФИОНА (как в alpha-bvp.py)
# =================================================================
def ode_system(r, y, alpha):
    f, fp = y
    sin_f = np.sin(f)
    cos_f = np.cos(f)
    sin2 = sin_f**2
    denom = r**2 + 2 * sin2
    term1 = 2 * sin_f * cos_f * (1 - fp**2 + sin2/r**2)
    term2 = (alpha / 2) * r**2 * sin_f
    term3 = -2 * r * fp
    fpp = (term1 + term2 + term3) / denom
    return np.vstack((fp, fpp))

def boundary_conditions(ya, yb):
    return np.array([ya[0] - np.pi, yb[0] - 0.0])

def solve_for_alpha(alpha, prev_sol=None):
    """
    Решает BVP для заданного alpha.
    Если prev_sol есть, используем его как начальное приближение.
    Возвращает (success, I2, I4, I0, alpha_calc, energy, sol_object)
    """
    r_min = 1e-4
    r_max = min(800.0, 40.0 / np.sqrt(alpha))
    r_nodes = np.logspace(np.log10(r_min), np.log10(r_max), 2000)
    
    if prev_sol is not None and hasattr(prev_sol, 'sol'):
        # Используем интерполяцию предыдущего решения как начальное приближение
        f_guess = prev_sol.sol(r_nodes)[0]
        fp_guess = prev_sol.sol(r_nodes)[1]
        y_guess = np.vstack((f_guess, fp_guess))
    else:
        # Анзац для первого приближения
        B_guess = np.sqrt(alpha)
        A_guess = 0.8168
        u_guess = (A_guess / r_nodes) * np.exp(-B_guess * r_nodes)
        f_guess = 2 * np.arctan(u_guess)
        fp_guess = (2 * u_guess / (1 + u_guess**2)) * (-1/r_nodes - B_guess)
        y_guess = np.vstack((f_guess, fp_guess))
    
    try:
        sol = solve_bvp(lambda r, y: ode_system(r, y, alpha),
                        boundary_conditions,
                        r_nodes, y_guess, tol=1e-7, max_nodes=30000)
        if not sol.success:
            return (False, 0, 0, 0, 0, 0, None)
    except:
        return (False, 0, 0, 0, 0, 0, None)
    
    # Вычисляем интегралы
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
    
    try:
        I2, _ = quad(I2_int, r_min, r_max, limit=300, epsabs=1e-8)
        I4, _ = quad(I4_int, r_min, r_max, limit=300, epsabs=1e-8)
        I0, _ = quad(I0_int, r_min, r_max, limit=300, epsabs=1e-8)
    except:
        return (False, 0, 0, 0, 0, 0, None)
    
    if I0 == 0:
        return (False, 0, 0, 0, 0, 0, None)
    
    alpha_calc = (I4 - I2) / (3 * I0)
    energy = I2 + I4 + 3 * alpha * I0
    return (True, I2, I4, I0, alpha_calc, energy, sol)

# =================================================================
# СКАНИРОВАНИЕ В УЗКОМ ДИАПАЗОНЕ ВОКРУГ 1/137
# =================================================================
alpha_exp = 0.0072973525693
alpha_vals = np.linspace(0.0070, 0.0076, 25)  # 25 точек
results = []
prev_sol = None

print("Сканирование α в диапазоне [0.0070, 0.0076]...")
for i, alpha in enumerate(alpha_vals):
    print(f"  α = {alpha:.6f}...", end=" ")
    ok, I2, I4, I0, ac, E, sol = solve_for_alpha(alpha, prev_sol)
    if ok:
        print(f"успех, E = {E:.6f}, α_calc = {ac:.8f}")
        results.append((alpha, I2, I4, I0, ac, E))
        prev_sol = sol  # для продолжения
    else:
        print("неудача")
        # Продолжаем без обновления prev_sol

if not results:
    print("Нет успешных решений!")
    exit()

# Преобразуем в массивы
alphas = np.array([r[0] for r in results])
I2s = np.array([r[1] for r in results])
I4s = np.array([r[2] for r in results])
I0s = np.array([r[3] for r in results])
ac_calc = np.array([r[4] for r in results])
energies = np.array([r[5] for r in results])

# Находим минимум энергии
min_idx = np.argmin(energies)
best_alpha = alphas[min_idx]
best_energy = energies[min_idx]
best_ac = ac_calc[min_idx]

print("\n" + "="*70)
print(" РЕЗУЛЬТАТЫ СКАНИРОВАНИЯ")
print("="*70)
print(f"Минимум энергии при α = {best_alpha:.6f}")
print(f"Энергия в минимуме: {best_energy:.6f}")
print(f"Вычисленная α из интегралов в этой точке: {best_ac:.8f}")
print(f"Экспериментальная α = {alpha_exp:.8f}")
print(f"Отклонение: {abs(best_alpha - alpha_exp)/alpha_exp*100:.3f}%")
print("="*70)

# Графики
plt.style.use('dark_background')
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
fig.suptitle('Зависимость интегралов и энергии от α (честное BVP-сканирование)', fontsize=12)

axes[0,0].plot(alphas, energies, 'o-', color='cyan')
axes[0,0].axvline(alpha_exp, color='yellow', linestyle='--', label='1/137')
axes[0,0].axvline(best_alpha, color='red', linestyle=':', label='минимум')
axes[0,0].set_xlabel('α')
axes[0,0].set_ylabel('Энергия')
axes[0,0].legend()
axes[0,0].grid(alpha=0.3)

axes[0,1].plot(alphas, I2s, 'o-', label='I₂', color='lime')
axes[0,1].plot(alphas, I4s, 'o-', label='I₄', color='magenta')
axes[0,1].plot(alphas, I0s, 'o-', label='I₀', color='orange')
axes[0,1].axvline(alpha_exp, color='yellow', linestyle='--')
axes[0,1].set_xlabel('α')
axes[0,1].set_ylabel('Интегралы')
axes[0,1].legend()
axes[0,1].grid(alpha=0.3)

axes[1,0].plot(alphas, ac_calc, 'o-', label='α_calc', color='cyan')
axes[1,0].plot(alphas, alphas, 'r--', label='α_input')
axes[1,0].axvline(alpha_exp, color='yellow', linestyle='--')
axes[1,0].set_xlabel('α')
axes[1,0].set_ylabel('α_calc')
axes[1,0].legend()
axes[1,0].grid(alpha=0.3)

axes[1,1].plot(alphas, I4s/I2s, 'o-', color='white')
axes[1,1].axvline(alpha_exp, color='yellow', linestyle='--')
axes[1,1].set_xlabel('α')
axes[1,1].set_ylabel('I₄/I₂')
axes[1,1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig('bvp_scan_narrow.png', dpi=150)
plt.show()

print("\nГрафик сохранён как bvp_scan_narrow.png")
if abs(best_alpha - alpha_exp) < 0.0001:
    print("*** ПРЕДСКАЗАНИЕ СОВПАДАЕТ С ЭКСПЕРИМЕНТОМ ***")
else:
    print("Минимум не совпадает с 1/137. Проверьте параметры.")
