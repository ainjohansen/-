import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import root
import matplotlib.pyplot as plt

# Параметры интегрирования
r0 = 1e-8
R  = 50.0   # достаточно большое расстояние

# Система ОДУ
def ode(r, y, alpha):
    f, fp = y
    r = max(r, 1e-12)
    sin_f = np.sin(f)
    cos_f = np.cos(f)
    sin2 = sin_f * sin_f
    denom = r*r + sin2
    term1 = sin_f * cos_f * (1 - fp*fp + sin2/(r*r))
    term2 = alpha * r * r * sin_f
    term3 = -2 * r * fp
    fpp = (term1 + term2 + term3) / denom
    return [fp, fpp]

# Функция, возвращающая невязки на правом конце
def residuals(params):
    s, alpha = params
    # начальные условия: f(r0) = π - s*r0, f'(r0) = s
    y0 = [np.pi - s*r0, s]
    sol = solve_ivp(ode, [r0, R], y0, args=(alpha,),
                    method='LSODA', rtol=1e-12, atol=1e-14)
    if not sol.success:
        return [1e6, 1e6]  # большой штраф
    fR = sol.y[0, -1]
    fpR = sol.y[1, -1]
    return [fR, fpR]

# Начальное приближение (из предыдущего успешного запуска)
s0 = -7.5626776289
alpha0 = 0.0072973525693

print("Ищем корень системы f(R)=0, f'(R)=0...")
res = root(residuals, [s0, alpha0], method='lm', tol=1e-12)

if res.success:
    s_opt, alpha_opt = res.x
    print(f"Оптимальный наклон s = {s_opt:.10f}")
    print(f"Оптимальное α = {alpha_opt:.10f}")
    print(f"1/α = {1/alpha_opt:.6f}")
    print(f"Невязки: f(R) = {residuals([s_opt, alpha_opt])[0]:.2e}, f'(R) = {residuals([s_opt, alpha_opt])[1]:.2e}")
else:
    print("Корень не найден:", res.message)
    exit()
