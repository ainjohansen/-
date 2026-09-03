import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import root
import matplotlib.pyplot as plt

# ------------------------------------------------------------
# 1. ОДУ
# ------------------------------------------------------------
def ode_system(r, y, alpha):
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

# ------------------------------------------------------------
# 2. Функция для shooting, возвращает f(R) и f'(R)
# ------------------------------------------------------------
def shoot(params, r0=1e-6, R=50.0):
    s, alpha = params
    y0 = [np.pi - s*r0, s]
    sol = solve_ivp(ode_system, [r0, R], y0, args=(alpha,),
                    method='LSODA', rtol=1e-10, atol=1e-12)
    if not sol.success:
        return [np.nan, np.nan]
    return [sol.y[0, -1], sol.y[1, -1]]

# ------------------------------------------------------------
# 3. Начальное приближение
# ------------------------------------------------------------
s_guess = -7.5626776289   # из предыдущей пристрелки при фиксированном α
alpha_guess = 0.0072973525693

print("Начальное приближение: s = {:.6f}, α = {:.10f}".format(s_guess, alpha_guess))
# Проверим невязку для начального приближения
fR, fpR = shoot([s_guess, alpha_guess])
print("Невязки: f(R) = {:.3e}, f'(R) = {:.3e}".format(fR, fpR))

# ------------------------------------------------------------
# 4. Поиск корня системы
# ------------------------------------------------------------
print("\nИщем корень системы f(R)=0, f'(R)=0...")
res = root(shoot, [s_guess, alpha_guess], method='hybr', tol=1e-10)

if res.success:
    s_opt, alpha_opt = res.x
    print(f"Найдено: s = {s_opt:.10f}, α = {alpha_opt:.10f}")
    print(f"1/α = {1/alpha_opt:.3f}")
    fR, fpR = shoot([s_opt, alpha_opt])
    print("Невязки после оптимизации: f(R) = {:.3e}, f'(R) = {:.3e}".format(fR, fpR))
    
    # Построим профиль для найденных параметров
    r0 = 1e-6
    R = 50.0
    y0 = [np.pi - s_opt*r0, s_opt]
    sol = solve_ivp(ode_system, [r0, R], y0, args=(alpha_opt,),
                    method='LSODA', rtol=1e-10, atol=1e-12, dense_output=True)
    r_plot = np.logspace(np.log10(r0), np.log10(R), 1000)
    f_plot = sol.sol(r_plot)[0]
    
    plt.figure()
    plt.plot(r_plot, f_plot)
    plt.xscale('log')
    plt.xlabel(r'$\rho$')
    plt.ylabel(r'$f(\rho)$')
    plt.title(f'Профиль электрона-хопфиона, α = {alpha_opt:.8f}')
    plt.grid(True)
    plt.show()
    
    # Вычислим интегралы для проверки
    from scipy.integrate import quad
    
    def integrand_I2(r):
        f = sol.sol(r)[0]
        fp = sol.sol(r)[1]
        sin_f = np.sin(f)
        return r*r*fp*fp + 2*sin_f*sin_f
    
    def integrand_I4(r):
        f = sol.sol(r)[0]
        fp = sol.sol(r)[1]
        sin_f = np.sin(f)
        sin2 = sin_f*sin_f
        sin_f_over_r = sin_f / r if r > 1e-8 else 0.0
        return sin2 * (2*fp*fp + sin_f_over_r*sin_f_over_r)
    
    def integrand_I0(r):
        f = sol.sol(r)[0]
        return (1 - np.cos(f)) * r*r
    
    I2, _ = quad(integrand_I2, r0, R, limit=1000)
    I4, _ = quad(integrand_I4, r0, R, limit=1000)
    I0, _ = quad(integrand_I0, r0, R, limit=1000)
    
    print(f"I2 = {I2:.6f}, I4 = {I4:.6f}, I0 = {I0:.6f}")
    alpha_check = (I4 - I2) / (3 * I0)
    print(f"Проверка по интегралам: α = {alpha_check:.10f}")
    print(f"Отклонение: {abs(alpha_check - alpha_opt):.2e}")
else:
    print("Корень не найден:", res.message)
