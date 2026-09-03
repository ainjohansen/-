import numpy as np
from scipy.integrate import solve_bvp, quad
from scipy.sparse import diags, csr_matrix
from scipy.sparse.linalg import eigsh
import matplotlib.pyplot as plt

# ============================================================
# 1. РЕШЕНИЕ BVP ДЛЯ ПРОФИЛЯ ЭЛЕКТРОНА (как в alpha-ft.py)
# ============================================================
alpha_exact = 0.0072973525693

def ode_system(r, y):
    f, fp = y
    sin_f = np.sin(f)
    cos_f = np.cos(f)
    sin2 = sin_f*sin_f
    denom = r*r + 2*sin2
    term1 = 2 * sin_f * cos_f * (1 - fp*fp + sin2/(r*r))
    term2 = (alpha_exact / 2) * r*r * sin_f
    term3 = -2 * r * fp
    fpp = (term1 + term2 + term3) / denom
    return np.vstack((fp, fpp))

def bc(ya, yb):
    return np.array([ya[0] - np.pi, yb[0]])

r_min = 1e-4
r_max = 120.0
n = 1000
r = np.logspace(np.log10(r_min), np.log10(r_max), n)

A0 = 0.8168
B0 = np.sqrt(alpha_exact)
u0 = (A0 / r) * np.exp(-B0 * r)
f0 = 2 * np.arctan(u0)
fp0 = (2 * u0 / (1 + u0**2)) * (-1/r - B0)
y_init = np.vstack((f0, fp0))

print("Решаем BVP...")
sol = solve_bvp(ode_system, bc, r, y_init, max_nodes=50000, verbose=2)
if not sol.success:
    print("BVP не сошлось:", sol.message)
    exit()
print("BVP успешно решено")

# Функции для профиля и его производной
def f(r):
    return sol.sol(r)[0]
def fp(r):
    return sol.sol(r)[1]

# ============================================================
# 2. ДИСКРЕТИЗАЦИЯ ОПЕРАТОРА ВТОРОЙ ВАРИАЦИИ
# ============================================================
# Выбираем сетку для задачи Штурма-Лиувилля (логарифмическая, с большим числом точек)
r_sl = np.logspace(np.log10(r_min), np.log10(100.0), 2000)  # до 100 достаточно
h = np.diff(r_sl)
# Используем узлы сетки, кроме граничных
N = len(r_sl) - 2   # внутренние точки (без r_min и r_max)
# Построим матрицы: для оператора L = -d/dr (r^2 A d/dr) + r^2 B
# Дискретизация: на сетке r_i (i=1..N) с шагами h_i
# Значения A_i, B_i в узлах
A_vals = np.zeros(N)
B_vals = np.zeros(N)
for i in range(N):
    ri = r_sl[i+1]
    A_vals[i] = 1 + 2*np.sin(f(ri))**2 / ri**2
    sin_f = np.sin(f(ri))
    cos_f = np.cos(f(ri))
    sin2 = sin_f**2
    fpr = fp(ri)
    term1 = np.cos(2*f(ri)) / ri**2 * (1 + 2*fpr**2 + sin2/ri**2)
    term2 = 2*sin2/ri**2 * (fpr**2 + sin2/ri**2)
    term3 = alpha_exact * cos_f
    B_vals[i] = term1 + term2 + term3

# Построение трёхдиагональной матрицы для оператора -d/dr (r^2 A d/dr)
# Используем разностную схему:
# (r^2 A u')' ≈ [ (r_{i+1/2}^2 A_{i+1/2}) (u_{i+1} - u_i)/h_i - (r_{i-1/2}^2 A_{i-1/2}) (u_i - u_{i-1})/h_{i-1} ] / ( (h_{i-1}+h_i)/2 )
# где A_{i+1/2} интерполируется.
# Упростим: предположим равномерную сетку в логарифмическом масштабе? Но у нас r не равномерное.
# Для простоты используем второй порядок точности с учётом неравномерности.

# Создадим списки для диагоналей
diag_main = np.zeros(N)
diag_low = np.zeros(N-1)
diag_up = np.zeros(N-1)

# Вычислим объёмные коэффициенты
for i in range(N):
    ri = r_sl[i+1]
    if i > 0:
        hi_left = ri - r_sl[i]
        hi_right = r_sl[i+2] - ri if i < N-1 else 0
    else:
        hi_left = r_sl[1] - r_sl[0]  # для первого узла
        hi_right = r_sl[2] - r_sl[1]
    # Средние расстояния
    # На самом деле, проще использовать метод конечных элементов или просто перейти к равномерной сетке по t = ln(r).
    # Чтобы избежать сложностей, перейдём к переменной t = ln(r), тогда уравнение упростится.

# Альтернатива: переписать оператор в переменной t = ln(r). Тогда ρ = e^t, d/dρ = e^{-t} d/dt, и объёмный элемент dρ = e^t dt.
# Уравнение Штурма-Лиувилля примет вид:
# - d/dt [ A(t) dη/dt ] + e^{2t} B(t) η = λ e^{2t} η, где A(t) = 1 + 2 sin^2 f / e^{2t}.
# Это стандартная задача с симметричным оператором. Дискретизация на равномерной сетке по t проще.

# Поэтому перейдём к равномерной сетке по t
t_min = np.log(r_min)
t_max = np.log(100.0)  # достаточно
Nt = 2000
t = np.linspace(t_min, t_max, Nt)
dt = t[1] - t[0]
r_grid = np.exp(t)

# Вычислим A(t) и B(t) на узлах
A_t = np.zeros(Nt)
B_t = np.zeros(Nt)
for i in range(Nt):
    ri = r_grid[i]
    sin_f = np.sin(f(ri))
    cos_f = np.cos(f(ri))
    sin2 = sin_f**2
    fpr = fp(ri)
    A_t[i] = 1 + 2*sin2 / ri**2
    term1 = np.cos(2*f(ri)) / ri**2 * (1 + 2*fpr**2 + sin2/ri**2)
    term2 = 2*sin2/ri**2 * (fpr**2 + sin2/ri**2)
    term3 = alpha_exact * cos_f
    B_t[i] = term1 + term2 + term3

# Матрица для оператора L = - d/dt (A d/dt) + e^{2t} B
# Дискретизация: центральные разности
# (A dη/dt) в точке i: (A_{i+1/2} (η_{i+1}-η_i)/dt - A_{i-1/2} (η_i-η_{i-1})/dt) / dt
# где A_{i+1/2} = (A_i + A_{i+1})/2

# Строим трёхдиагональную матрицу
main = np.zeros(Nt)
low = np.zeros(Nt-1)
up = np.zeros(Nt-1)

for i in range(1, Nt-1):
    A_left = (A_t[i-1] + A_t[i]) / 2
    A_right = (A_t[i] + A_t[i+1]) / 2
    main[i] = (A_left + A_right) / dt**2 + np.exp(2*t[i]) * B_t[i]
    low[i-1] = -A_left / dt**2
    up[i] = -A_right / dt**2

# Граничные условия: η=0 на границах (при t=t_min и t=t_max)
# Поэтому исключаем граничные узлы, оставляем внутренние (1..Nt-2)
N_inner = Nt - 2
main_inner = main[1:-1]
low_inner = low[1:-1]  # соответствует связи между i и i-1
up_inner = up[1:-1]    # связь между i и i+1

# Матрица массы M = e^{2t} (для обобщённой задачи L η = λ M η)
# Используем диагональную аппроксимацию: M_i = exp(2*t[i]) * dt (но dt сократится при нормировке)
M_diag = np.exp(2*t[1:-1])  # без dt, так как в L уже есть деление на dt^2, но в собственных значениях масштаб учтётся

# Строим разреженные матрицы
diags_data = [main_inner, low_inner, up_inner]
diags_pos = [0, -1, 1]
L_mat = diags(diags_data, diags_pos, shape=(N_inner, N_inner), format='csr')
M_mat = diags(M_diag, 0, shape=(N_inner, N_inner), format='csr')

# ============================================================
# 3. НАХОЖДЕНИЕ НАИМЕНЬШИХ СОБСТВЕННЫХ ЗНАЧЕНИЙ
# ============================================================
# Решаем обобщённую задачу L η = λ M η
print("Вычисляем наименьшие собственные значения...")
# Используем eigsh для симметричной обобщённой задачи, преобразуя к стандартной
# Поскольку L и M симметричны и M положительна, можно вычислить
try:
    eigenvalues, eigenvectors = eigsh(L_mat, k=5, M=M_mat, sigma=0.0, which='LM')
    # sigma=0.0 ищем собственные значения, близкие к 0, which='LM' даст наибольшие по модулю, но sigma смещает
    # Лучше: ищем наименьшие с помощью shift-invert
    eigenvalues, eigenvectors = eigsh(L_mat, k=5, M=M_mat, sigma=0.0, which='SM', tol=1e-6)
except Exception as e:
    print("Ошибка при решении обобщённой задачи:", e)
    # Попробуем решить стандартную задачу (L - λ M) = 0, используя inverse iteration
    # Для этого можно вычислить наименьшее собственное значение с помощью scipy.sparse.linalg.eigs
    eigenvalues, eigenvectors = eigsh(L_mat, k=1, M=M_mat, sigma=0.0, which='LM', tol=1e-6)
    # Это даст собственное значение, ближайшее к 0 (по модулю), что и требуется

print("Найденные собственные значения (λ):")
for i, val in enumerate(eigenvalues):
    print(f"λ_{i} = {val:.6e}")

if eigenvalues[0] > 0:
    print("\nНаименьшее собственное значение λ_min > 0 → решение устойчиво (локальный минимум).")
else:
    print("\nНаименьшее собственное значение λ_min ≤ 0 → решение неустойчиво.")

# ============================================================
# 4. ВИЗУАЛИЗАЦИЯ СОБСТВЕННЫХ ФУНКЦИЙ
# ============================================================
# Построим первые несколько собственных функций
t_plot = t[1:-1]
r_plot = np.exp(t_plot)

plt.figure()
for i in range(min(3, len(eigenvalues))):
    plt.plot(r_plot, eigenvectors[:, i], label=f'λ = {eigenvalues[i]:.3e}')
plt.xscale('log')
plt.xlabel(r'$\rho$')
plt.ylabel(r'$\eta(\rho)$')
plt.title('Первые собственные функции оператора второй вариации')
plt.legend()
plt.grid()
plt.show()
