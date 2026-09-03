import numpy as np
from scipy.integrate import solve_bvp, quad, solve_ivp
from scipy.sparse import diags
from scipy.sparse.linalg import eigs
import matplotlib.pyplot as plt

alpha = 0.0072973525693

def ode(r, y):
    f, fp = y
    sin_f = np.sin(f)
    cos_f = np.cos(f)
    sin2 = sin_f*sin_f
    denom = r*r + 2*sin2
    term1 = 2*sin_f*cos_f*(1 - fp*fp + sin2/(r*r))
    term2 = (alpha/2)*r*r*sin_f
    term3 = -2*r*fp
    fpp = (term1 + term2 + term3)/denom
    return np.vstack((fp, fpp))

def bc(ya, yb):
    return np.array([ya[0] - 3*np.pi, yb[0]])

# Параметры, аналогичные успешному запуску proton.py
r_min = 1e-4
r_max = 100.0
n = 1000
r = np.logspace(np.log10(r_min), np.log10(r_max), n)

# Начальное приближение: линейное падение от 3π до 0
f_init = 3*np.pi * (1 - r/r_max)
fp_init = -3*np.pi/r_max * np.ones_like(r)
y_init = np.vstack((f_init, fp_init))

print("Решаем BVP для протона...")
sol = solve_bvp(ode, bc, r, y_init, tol=1e-6, max_nodes=50000, verbose=2)
if not sol.success:
    print("BVP не сошлось:", sol.message)
    exit()
print("BVP успешно решено")

# Сохраним решение для дальнейшего использования
np.savez('proton_solution.npz', r=sol.x, f=sol.y[0], fp=sol.y[1])

# Функции для интерполяции
def f(r):
    return sol.sol(r)[0]
def fp(r):
    return sol.sol(r)[1]

# Коэффициенты оператора второй вариации
def A_coef(r):
    sin_f = np.sin(f(r))
    return 1 + 2*sin_f*sin_f/(r*r)

def B_coef(r):
    sin_f = np.sin(f(r))
    cos_f = np.cos(f(r))
    sin2 = sin_f*sin_f
    fpr = fp(r)
    term1 = np.cos(2*f(r))/(r*r) * (1 + 2*fpr*fpr + sin2/(r*r))
    term2 = 2*sin2/(r*r) * (fpr*fpr + sin2/(r*r))
    term3 = alpha * cos_f
    return term1 + term2 + term3

# Дискретизация для решения задачи на собственные значения
r_grid = np.logspace(np.log10(r_min), np.log10(r_max), 2000)
dr = np.diff(r_grid)
r_mid = (r_grid[1:] + r_grid[:-1])/2

# Построение матрицы оператора Штурма-Лиувилля
N = len(r_grid) - 2  # внутренние точки (без границ)
# Матрица A (диагональная) и B (трёхдиагональная) для метода конечных разностей
# Уравнение: -d/dr (r^2 A(r) dη/dr) + r^2 B(r) η = λ η
# Дискретизация: на сетке r_i, i=1..N, с границами η_0=0, η_{N+1}=0

# Коэффициенты на внутренних точках
r_i = r_grid[1:-1]
A_i = A_coef(r_i)
B_i = B_coef(r_i)
# Весовые коэффициенты для производной
# Для i-й внутренней точки:
# -d/dr (r^2 A dη/dr) ≈ -[ (r_{i+1/2}^2 A_{i+1/2} (η_{i+1}-η_i)/Δr_{i+1/2} - r_{i-1/2}^2 A_{i-1/2} (η_i-η_{i-1})/Δr_{i-1/2} ) / (0.5*(Δr_{i-1/2}+Δr_{i+1/2})) ]
# где r_{i+1/2} = (r_i+r_{i+1})/2, A_{i+1/2} = (A_i+A_{i+1})/2
# Δr_{i+1/2} = r_{i+1} - r_i

# Вычислим полуцелые величины
r_half = (r_grid[:-1] + r_grid[1:]) / 2
A_half = (A_coef(r_grid[:-1]) + A_coef(r_grid[1:])) / 2
dr_half = np.diff(r_grid)

# Построение трёхдиагональной матрицы L
# Для i от 1 до N (индексация внутренних точек)
# L_{i,i-1} = - (r_{i-1/2}^2 A_{i-1/2}) / (dr_{i-1/2} * h_i)
# L_{i,i}   =   (r_{i-1/2}^2 A_{i-1/2}) / (dr_{i-1/2} * h_i) + (r_{i+1/2}^2 A_{i+1/2}) / (dr_{i+1/2} * h_i)
# L_{i,i+1} = - (r_{i+1/2}^2 A_{i+1/2}) / (dr_{i+1/2} * h_i)
# где h_i = (dr_{i-1/2} + dr_{i+1/2})/2

h_i = (dr_half[:-1] + dr_half[1:]) / 2

# Диагонали
diag_main = np.zeros(N)
diag_low = np.zeros(N-1)
diag_up = np.zeros(N-1)

for i in range(N):
    # индексы: внутренняя точка i соответствует сеточному индексу i+1
    left_half = i
    right_half = i+1
    r_left_half = r_half[left_half]
    r_right_half = r_half[right_half]
    A_left = A_half[left_half]
    A_right = A_half[right_half]
    dr_left = dr_half[left_half]
    dr_right = dr_half[right_half]
    h = h_i[i]
    
    # Вклад от левой половинки
    coeff_left = (r_left_half**2 * A_left) / (dr_left * h)
    # Вклад от правой половинки
    coeff_right = (r_right_half**2 * A_right) / (dr_right * h)
    
    diag_main[i] = coeff_left + coeff_right
    if i > 0:
        diag_low[i-1] = -coeff_left
    if i < N-1:
        diag_up[i] = -coeff_right

# Матрица масс (диагональная) от члена r^2 B(r)
mass = r_i**2 * B_i

# Формируем разреженную матрицу
L = diags([diag_low, diag_main, diag_up], [-1, 0, 1], format='csr')
M = diags(mass, 0, format='csr')

# Решаем обобщённую задачу на собственные значения L η = λ M η
# Ищем несколько наименьших по модулю собственных значений
print("Вычисляем наименьшие собственные значения...")
try:
    eigvals, eigvecs = eigs(L, k=6, M=M, sigma=0.0, which='LM', tol=1e-6, maxiter=10000)
    # eigs возвращает собственные значения, ближайшие к sigma (0.0)
    eigvals = eigvals.real
    # сортируем по возрастанию
    eigvals = np.sort(eigvals)
    print("Найденные собственные значения (λ):")
    for i, lam in enumerate(eigvals):
        print(f"λ_{i} = {lam:.6e}")
    if eigvals[0] > 0:
        print("Наименьшее собственное значение λ_min > 0 → решение устойчиво (локальный минимум).")
    else:
        print("Наименьшее собственное значение λ_min ≤ 0 → решение неустойчиво.")
except Exception as e:
    print("Ошибка при решении обобщённой задачи:", e)
