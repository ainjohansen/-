#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Верификация числовых утверждений Manifest-v52-rus.tex.
Скрипт пересчитывает каждое спорное значение из канонических констант
и сравнивает с заявленным в .tex. Выводит итоговый список расхождений.
"""
import math

# --- Канонические константы (из манифеста) ---
ALPHA_INV = 137.035999177
ALPHA     = 1.0 / ALPHA_INV
M_E_MeV   = 0.5109989500          # МэВ
M_P_MeV   = 938.272088            # МэВ
R_E_m     = 3.86159e-13           # м, ħ/(m_e c)

def rel(a, b):
    """Относительное отклонение a от b в %."""
    return (a - b) / b * 100.0

print("=" * 78)
print("ВЕРИФИКАЦИЯ Manifest-v52-rus.tex — пересчёт спорных значений")
print("=" * 78)

# (1) a_e: ряд пограничного слоя vs заявленное vs эксперимент
C = [0.5, -0.328478966, 1.181241456, -1.912245, 6.737]   # C1..C5
x = ALPHA / math.pi
a_e_formula = sum(c * x**(i+1) for i, c in enumerate(C))
a_e_claimed = 1.1596521808e-3
a_e_exp     = 1.15965218059e-3
print("\n[1] a_e (аномальный магнитный момент электрона)")
print(f"    формула ряда  = {a_e_formula:.11e}")
print(f"    заявлено в tex= {a_e_claimed:.11e}")
print(f"    эксперимент   = {a_e_exp:.11e}")
print(f"    |формула-заявл| = {abs(a_e_formula-a_e_claimed):.3e}  "
      f"({abs(a_e_formula-a_e_claimed)/a_e_exp*1e9:.2f} ppb)")
print(f"    |формула-эксп | = {abs(a_e_formula-a_e_exp):.3e}  "
      f"({abs(a_e_formula-a_e_exp)/a_e_exp*1e9:.2f} ppb)  <-- НЕ 0.18 ppb")

# (2) E_nu_crit: с фактором погранслоя и без
N4 = 4 * (2*4 - 1)                 # = 28
M4_no   = M_E_MeV * N4**3 / 1000.0 # ГэВ, без фактора
M4_yes  = M4_no * (1 + 1.5*ALPHA/math.pi)
M_P_GeV = M_P_MeV / 1000.0
E_no    = (M4_no**2  - M_P_GeV**2) / (2*M_P_GeV)
E_yes   = (M4_yes**2 - M_P_GeV**2) / (2*M_P_GeV)
print("\n[2] E_nu_crit (порог фрагментации струны)")
print(f"    M4 без фактора = {M4_no:.4f} ГэВ  -> E = {E_no:.3f} ГэВ  (tex Разд.7: 67.055)")
print(f"    M4 с фактором  = {M4_yes:.4f} ГэВ  -> E = {E_yes:.3f} ГэВ  (tex Разд.12: 67.055)")
print(f"    => Разд.7 НЕСООТВЕТСТВУЕТ (нужен фактор 1+1.5α/π)")

# (3) Z_m: две формулы
print("\n[3] Z_m (механический импеданс)")
print("    L~140: sqrt(rho0*G0)  -> размерность кг/(м^2·с)  [ВЕРНО]")
print("    L~360: sqrt(rho0/G0)  -> размерность с/м         [НЕВЕРНО]")

# (4) V_core
V_core = 2*math.pi**2 * ALPHA**2 * R_E_m**3
print("\n[4] V_core (объём керна электрона)")
print(f"    2π²α²R_e³ = {V_core:.3e} м³   (tex: 6.08e-40, расхождение x{6.08e-40/V_core:.1f})")

# (5) Delta m21^2
m_nu2, m_nu1 = 8.67332e-3, 0.24604e-3   # эВ
dm21 = m_nu2**2 - m_nu1**2
print("\n[5] Delta m21^2")
print(f"    m_nu2² - m_nu1² = {dm21:.5e} эВ²   (tex: 7.5226e-5)")

# (6) Коидэ Q
Q_exp = 0.666661
print("\n[6] Отклонение Коидэ Q")
print(f"    2/3 - {Q_exp} = {2/3 - Q_exp:.3e}   (tex: 8.5e-6)")

# (7) H2 R0, De
R0 = 2*ALPHA*R_E_m*math.sqrt(2)*math.log(1/ALPHA) * 1e10  # Å
De = ALPHA**2 * 510998.95 * (1 - ALPHA/math.pi)            # эВ
print("\n[7] H2 (R0, De) — формулы vs заявленные эталоны")
print(f"    R0 = {R0:.4f} Å   (tex эталон: 0.7414 Å)")
print(f"    De = {De:.3f} эВ  (tex эталон: 4.747 эВ)")

print("\n" + "=" * 78)
print("ИТОГ: 3 критических (a_e, E_nu_crit, Z_m), 3 средних (V_core,")
print("      Δm21², Q), 2 подачи (H2, C1-конвенция). См. отчёт выше.")
print("=" * 78)

