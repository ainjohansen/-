#!/usr/bin/env python3
import sympy as sp

# Символьные переменные
alpha, Mp = sp.symbols('alpha Mp', positive=True)

# Значения CODATA 2022 и их 1-sigma погрешности
val_alpha = 1.0 / 137.035999084
unc_alpha = 1.1e-10 * (val_alpha**2)  # из неопределенности 1/alpha = 0.000000021
val_Mp = 938.27208816
unc_Mp = 0.00000029


def calc_with_uncertainty(expr, name):
  val = float(expr.subs({alpha: val_alpha, Mp: val_Mp}))
  df_dalpha = float(sp.diff(expr, alpha).subs({alpha: val_alpha, Mp: val_Mp}))
  df_dMp = float(sp.diff(expr, Mp).subs({alpha: val_alpha, Mp: val_Mp}))
  sigma = sp.sqrt((df_dalpha * unc_alpha) ** 2 + (df_dMp * unc_Mp) ** 2)
  print(f"{name:<20}: {val:.8f} ± {float(sigma):.8e}")


# Примеры формул:
# 1. Масса электрона
me_expr = Mp / ((67 / 5) * (1 / alpha) * (1 - (4 / 3) * alpha**2))
calc_with_uncertainty(me_expr, 'm_e (MeV)')

# 2. Масштаб нейтрино
M_nu_expr = (2 * (alpha**5) * (Mp / 3) * 1e9) / (1 + 1 / (6 * sp.pi))
calc_with_uncertainty(M_nu_expr, 'M_nu (meV)')

# 3. Вакуумное среднее v
v_expr = (Mp / alpha) * sp.sqrt(sp.Rational(11, 3)) * 1e-3
calc_with_uncertainty(v_expr, 'vev (GeV)')