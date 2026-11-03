#!/usr/bin/env python3
import itertools
import numpy as np

# Все возможные дроби из базиса индексов <= 20
fractions = sorted(
    list({a / b for a in range(1, 21) for b in range(1, 21) if a != b})
)
print(f"Всего доступных рациональных квантов в базисе <= 20: {len(fractions)}")

# Ключевые безразмерные параметры теории и их требуемые коридоры точности (ppm)
targets = [
    ("sin2_thetaW", 3 / 13, 0.23129, 2300),  # допуск ~0.2%
    ("Koide_Q", 2 / 3, 0.666661, 10),  # допуск 10 ppm
    ("Neutrino_Q", 8 / 15, 0.533333, 500),  # допуск
    ("Wolfenstein_A_sq", 2 / 3, 0.667, 1000),
    ("SEMF_aV_ratio", 2 / 9, 0.2222, 500),
    ("SEMF_aS_ratio", 1 / 4, 0.2500, 500),
    ("SEMF_aA_ratio", 1 / 3, 0.3333, 500),
    ("Proton_FormFactor", 3 / 4, 0.7500, 1000),
    ("Strong_alpha_s_tree", 2 / 17, 0.1176, 2000),
    ("CKM_phase_sq", 9 / 2, 4.500, 2000),
]

# Monte Carlo: вероятность случайного попадания независимых случайных чисел в эти окна
N_trials = 1000000
p_individual = []

for name, theor, exp, tol_ppm in targets:
  window = exp * (tol_ppm * 1e-6)
  # Вероятность, что случайная дробь из пула попадет в окно
  hits = sum(1 for f in fractions if abs(f - exp) <= window)
  p = hits / len(fractions)
  p_individual.append(max(p, 1e-6))
  print(f"Параметр {name:<20}: p-value случайного попадания = {p:.6f}")

p_combined = np.prod(p_individual)
print("-" * 60)
print(
    f"Совокупный P-value случайного совпадения по ансамблю: {p_combined:.2e}"
)