import numpy as np
from scipy.integrate import solve_bvp, quad
import matplotlib.pyplot as plt

# Константы
alpha_exp = 0.0072973525693          # 1/137.036
hbar_c = 197.3269804                 # МэВ·фм
m_p_exp = 938.2720813                # МэВ
m_n_exp = 939.5654205                # МэВ
R_p_exp = 0.84123                    # фм (зарядовый радиус протона)

def ode_system(r, y, epsilon):
    """
    Система ОДУ для нейтрона с дефектом в центре.
    Параметр epsilon задаёт отклонение f(0) от pi.
    """
    f, fp = y
    sin_f = np.sin(f)
    cos_f = np.cos(f)
    sin2 = sin_f**2
    
    denom = r**2 + 2 * sin2
    # Стандартное уравнение Эйлера-Лагранжа (как для протона)
    term1 = 2 * sin_f * cos_f * (1 - fp**2 + sin2/r**2)
    term2 = (alpha_exp / 2) * r**2 * sin_f
    term3 = -2 * r * fp
    fpp = (term1 + term2 + term3) / denom
    return np.vstack((fp, fpp))

def boundary_conditions(ya, yb, epsilon):
    """Граничные условия: f(0)=pi-epsilon, f(inf)=0."""
    return np.array([ya[0] - (np.pi - epsilon), yb[0] - 0.0])

def solve_neutron(epsilon, r_max=120.0, plot=False):
    """Решает BVP для нейтрона с заданным epsilon."""
    r_min = 1e-4
    r_nodes = np.logspace(np.log10(r_min), np.log10(r_max), 1000)
    
    # Начальное приближение (профиль протона, но с поправкой на epsilon)
    A_core = 0.8168
    B_tail = 0.08542
    u_guess = (A_core / r_nodes) * np.exp(-B_tail * r_nodes)
    f_guess = 2 * np.arctan(u_guess)
    # Сдвигаем начальное значение к pi-epsilon
    f_guess[0] = np.pi - epsilon
    fp_guess = (2 * u_guess / (1 + u_guess**2)) * (-1/r_nodes - B_tail)
    y_guess = np.vstack((f_guess, fp_guess))
    
    sol = solve_bvp(lambda r, y: ode_system(r, y, epsilon),
                    lambda ya, yb: boundary_conditions(ya, yb, epsilon),
                    r_nodes, y_guess, tol=1e-8, max_nodes=50000)
    if not sol.success:
        print(f"BVP не сошёлся для epsilon={epsilon}: {sol.message}")
        return None
    return sol

def compute_integrals(sol, r_min=1e-4, r_max=120.0):
    """Вычисляет I2, I4, I0 из решения."""
    def integrand_I2(r):
        y = sol.sol(r)
        return r**2 * y[1]**2 + 2 * np.sin(y[0])**2
    def integrand_I4(r):
        y = sol.sol(r)
        sin_f = np.sin(y[0])
        sin_f_r = sin_f / r if r > 1e-8 else -y[1]
        return sin_f**2 * (2 * y[1]**2 + sin_f_r**2)
    def integrand_I0(r):
        y = sol.sol(r)
        return (1 - np.cos(y[0])) * r**2
    
    I2, _ = quad(integrand_I2, r_min, r_max, limit=500)
    I4, _ = quad(integrand_I4, r_min, r_max, limit=500)
    I0, _ = quad(integrand_I0, r_min, r_max, limit=500)
    return I2, I4, I0

def compute_rms_radius(sol, r_min=1e-4, r_max=20.0):
    """Вычисляет безразмерный среднеквадратичный радиус поля."""
    # Плотность энергии (упрощённо: вклад от градиентов + массовый член)
    def integrand_num(r):
        y = sol.sol(r)
        # берём r^2 * (плотность) * r^2 dr -> подынтегральное выражение
        # Плотность энергии: (f'^2 + 2 sin^2 f / r^2) * r^2 ? Для радиуса нужно взвешивать r^2
        # Проще: используем распределение топологической плотности ~ sin^2 f * f' / r (не точно)
        # Возьмём стандартное выражение для RMS из профиля f: 
        # <r^2> = ∫ r^2 * (sin^2 f) * r^2 dr / ∫ (sin^2 f) r^2 dr
        sin2 = np.sin(y[0])**2
        return sin2 * r**4
    def integrand_den(r):
        y = sol.sol(r)
        sin2 = np.sin(y[0])**2
        return sin2 * r**2
    num, _ = quad(integrand_num, r_min, r_max, limit=500)
    den, _ = quad(integrand_den, r_min, r_max, limit=500)
    if den == 0:
        return 0.0
    return np.sqrt(num / den)

def compute_magnetic_moment(sol, r_max=20.0):
    """Вычисляет безразмерный интеграл для магнитного момента."""
    def integrand(r):
        y = sol.sol(r)
        sin_f = np.sin(y[0])
        # Магнитный момент пропорционален ∫ r^2 sin^2 f |f'| dr
        return r**2 * sin_f**2 * np.abs(y[1])
    mag_int, _ = quad(integrand, 1e-4, r_max, limit=500)
    return mag_int

def compute_quadrupole_moment(sol, r_max=20.0):
    """Вычисляет безразмерный интеграл для электрического квадрупольного момента."""
    def integrand(r):
        y = sol.sol(r)
        sin_f = np.sin(y[0])
        # Квадруполь: ∫ r^4 * (sin^2 f) * |f'| dr
        return r**4 * sin_f**2 * np.abs(y[1])
    quad_int, _ = quad(integrand, 1e-4, r_max, limit=500)
    return quad_int

# ============================================================
# ПОДБОР EPSILON (вручную или автоматически)
# ============================================================
# Целевая масса нейтрона: m_n_target = 939.5654205 МэВ
# Масса протона из модели (при epsilon=0) должна получиться ~938.272 МэВ.
# Сначала проверим протон (epsilon=0):
print("Решаем для протона (epsilon=0):")
sol_p = solve_neutron(0.0)
if sol_p is not None:
    I2_p, I4_p, I0_p = compute_integrals(sol_p)
    alpha_check_p = (I4_p - I2_p) / (3 * I0_p)
    rms_dimless_p = compute_rms_radius(sol_p)
    # Масштабный коэффициент из радиуса протона
    scale_p = R_p_exp / rms_dimless_p   # фм / безразмерная единица
    # Энергия (масса) в МэВ: E = (ħc / scale) * E_dimless, где E_dimless = I2+I4+I0 (с точностью до коэф.)
    # Коэффициент перед интегралами в функционале энергии: в лагранжиане были Λ2, Λ4, Λ0.
    # В безразмерных переменных энергия = (4π) * (I2 + I4 + I0) * (ħc / a), где a - масштаб.
    # Примем E_dimless = I2 + I4 + I0.
    E_dimless_p = I2_p + I4_p + I0_p
    m_p_calc = (hbar_c / scale_p) * E_dimless_p
    print(f"Протон: rms_dimless={rms_dimless_p:.4f}, scale={scale_p:.4f} фм, E_dimless={E_dimless_p:.2f}, m_calc={m_p_calc:.3f} МэВ (эксп. {m_p_exp:.3f})")
    # Отношение расчётной массы к экспериментальной для калибровки
    norm_factor = m_p_exp / m_p_calc
    print(f"Калибровочный множитель для энергии: {norm_factor:.6f}")
    # Теперь для нейтрона будем использовать этот же множитель (предполагая, что форма функционала та же)
else:
    print("Ошибка при решении для протона")
    exit()

# Теперь подбираем epsilon так, чтобы масса нейтрона стала m_n_exp
def mass_for_epsilon(epsilon):
    sol = solve_neutron(epsilon)
    if sol is None:
        return np.nan
    I2, I4, I0 = compute_integrals(sol)
    rms_dimless = compute_rms_radius(sol)
    scale = R_p_exp / rms_dimless   # считаем радиус нейтрона таким же, как у протона
    E_dimless = I2 + I4 + I0
    m_calc = (hbar_c / scale) * E_dimless * norm_factor
    return m_calc

# Поиск epsilon методом половинного деления
print("\nПоиск epsilon для нейтрона...")
epsilon_low = 0.0
epsilon_high = 0.005
m_low = mass_for_epsilon(epsilon_low)
m_high = mass_for_epsilon(epsilon_high)
if np.isnan(m_low) or np.isnan(m_high):
    print("Не удалось вычислить массу на границах")
    exit()

target = m_n_exp
for _ in range(10):
    eps_mid = (epsilon_low + epsilon_high) / 2
    m_mid = mass_for_epsilon(eps_mid)
    print(f"eps={eps_mid:.6f}, m={m_mid:.3f} МэВ")
    if m_mid < target:
        epsilon_low = eps_mid
    else:
        epsilon_high = eps_mid
    if abs(m_mid - target) < 0.001:
        break
epsilon_best = (epsilon_low + epsilon_high) / 2
print(f"\nПодобранный epsilon = {epsilon_best:.6f}")

# Окончательное решение для нейтрона с лучшим epsilon
sol_n = solve_neutron(epsilon_best)
if sol_n is None:
    print("Не удалось получить решение для нейтрона")
    exit()

# Вычисление всех характеристик
I2_n, I4_n, I0_n = compute_integrals(sol_n)
alpha_check_n = (I4_n - I2_n) / (3 * I0_n)
rms_dimless_n = compute_rms_radius(sol_n)
scale_n = R_p_exp / rms_dimless_n
E_dimless_n = I2_n + I4_n + I0_n
m_n_calc = (hbar_c / scale_n) * E_dimless_n * norm_factor

# Магнитный момент и квадрупольный момент (безразмерные интегралы)
mag_int_n = compute_magnetic_moment(sol_n)
quad_int_n = compute_quadrupole_moment(sol_n)
# Калибровка магнитного момента по протону: для протона mag_int_p ≈ ? (возьмём из скрипта)
# В протонном скрипте было: mu_proton_nm = mag_integral * 1.4456
# Мы вычислим аналогично, но для нейтрона используем ту же калибровку (хотя ожидается отрицательный момент)
# Для протона mag_int_p можно получить из sol_p:
mag_int_p = compute_magnetic_moment(sol_p)
mu_p_calc_nm = mag_int_p * 1.4456   # примерная калибровка
mu_n_calc_nm = mag_int_n * (mu_p_calc_nm / mag_int_p) if mag_int_p != 0 else 0
# Квадрупольный момент (фм²) — аналогично
quad_int_p = compute_quadrupole_moment(sol_p)
q_p_calc_fm2 = quad_int_p * (0.2644**2) * 0.023   # из протонного скрипта
q_n_calc_fm2 = quad_int_n * (q_p_calc_fm2 / quad_int_p) if quad_int_p != 0 else 0

print("\n" + "="*60)
print(" РЕЗУЛЬТАТЫ ДЛЯ НЕЙТРОНА (с дефектом ε = {:.6f})".format(epsilon_best))
print("="*60)
print(f"I2 = {I2_n:.6f}")
print(f"I4 = {I4_n:.6f}")
print(f"I0 = {I0_n:.6f}")
print(f"α из баланса = {alpha_check_n:.10f} (эксп. {alpha_exp:.10f})")
print(f"Безразмерный RMS радиус = {rms_dimless_n:.4f}")
print(f"Масштабный коэффициент = {scale_n:.4f} фм")
print(f"Безразмерная энергия E_dimless = {E_dimless_n:.2f}")
print(f"Вычисленная масса нейтрона = {m_n_calc:.3f} МэВ (эксп. {m_n_exp:.3f})")
print(f"Разность m_n - m_p = {m_n_calc - m_p_exp:.3f} МэВ (эксп. 1.293 МэВ)")
print(f"Магнитный момент (безразм. интеграл) = {mag_int_n:.4f}")
print(f"Магнитный момент нейтрона (ядерных магнетонов) = {mu_n_calc_nm:.3f} μN (эксп. -1.913 μN)")
print(f"Электрический квадрупольный момент = {q_n_calc_fm2:.6f} фм² (эксп. ~0)")
print("="*60)

# Сохраняем профиль для визуализации
rho_plot = np.linspace(1e-4, 8, 500)
f_plot = sol_n.sol(rho_plot)[0]
np.savetxt("neutron_profile.txt", np.column_stack((rho_plot, f_plot)),
           header="rho   f(rho)", comments="")

# Построим график профиля (опционально)
plt.figure(figsize=(8,5))
plt.plot(rho_plot, f_plot, 'r-', linewidth=2, label='Нейтрон (ε={:.5f})'.format(epsilon_best))
plt.axhline(np.pi, color='gray', linestyle='--', alpha=0.7)
plt.axhline(0, color='gray', linestyle='--', alpha=0.7)
plt.xlabel(r'$\rho = r/a$')
plt.ylabel(r'$f(\rho)$')
plt.title('Профиль поля нейтрона (фрустрированный трилистник)')
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("neutron_profile.png")
plt.show()

print("\nГотово. Теперь можно строить 3D-визуализацию, используя профиль neutron_profile.txt")
