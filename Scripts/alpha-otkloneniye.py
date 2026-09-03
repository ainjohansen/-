import numpy as np
import matplotlib.pyplot as plt

# Параметры, полученные из двухпараметрической пристрелки
alpha = 0.0072973525693
s = -7.5626776289
r0 = 1e-8
R = 100.0

# Функция для получения численного профиля (загружаем из файла или повторяем интегрирование)
# Чтобы не дублировать, можно сохранить профиль из предыдущего скрипта в файл,
# но для простоты повторим интегрирование здесь (или используем интерполяцию из sol)
from scipy.integrate import solve_ivp

def ode(r, y):
    f, fp = y
    r = max(r, 1e-12)
    sin_f = np.sin(f)
    cos_f = np.cos(f)
    sin2 = sin_f*sin_f
    denom = r*r + sin2
    term1 = sin_f * cos_f * (1 - fp*fp + sin2/(r*r))
    term2 = alpha * r*r * sin_f
    term3 = -2 * r * fp
    fpp = (term1 + term2 + term3) / denom
    return [fp, fpp]

sol = solve_ivp(ode, [r0, R], [np.pi - s*r0, s],
                method='LSODA', rtol=1e-12, atol=1e-14,
                dense_output=True)

f_num = lambda r: sol.sol(r)[0]

# Параметрическая аппроксимация
A_par = 0.8168
B_par = 0.08542
def f_par(r):
    u = A_par / r * np.exp(-B_par * r)
    return 2 * np.arctan(u)

# Точки для сравнения
r_compare = np.logspace(np.log10(r0), np.log10(R), 500)
f_num_vals = f_num(r_compare)
f_par_vals = f_par(r_compare)

# Ошибки
abs_err = np.abs(f_num_vals - f_par_vals)
rel_err = abs_err / (np.abs(f_num_vals) + 1e-12)
print("Максимальная абсолютная ошибка:", np.max(abs_err))
print("Максимальная относительная ошибка:", np.max(rel_err))
print("Среднеквадратичная ошибка:", np.sqrt(np.mean(abs_err**2)))

# График
plt.figure()
plt.plot(r_compare, f_num_vals, 'b-', label='Численное решение')
plt.plot(r_compare, f_par_vals, 'r--', label='Параметрическая аппроксимация')
plt.xscale('log')
plt.xlabel(r'$\rho$')
plt.ylabel(r'$f(\rho)$')
plt.legend()
plt.title('Сравнение профилей')
plt.grid(True)
plt.show()

# Разность
plt.figure()
plt.plot(r_compare, f_num_vals - f_par_vals)
plt.xscale('log')
plt.xlabel(r'$\rho$')
plt.ylabel(r'$\Delta f(\rho)$')
plt.title('Разность численного и параметрического профилей')
plt.grid(True)
plt.show()
