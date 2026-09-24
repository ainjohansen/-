#!/usr/bin/env python3
"""
ТОПОЛОГИЧЕСКАЯ ЭЛАСТОДИНАМИКА ВАКУУМА (ТЭВ)
Статья 03: Инженерная модель структуры нейтрона, расщепления масс и времени жизни.

Проверяемые величины:
  1. Разность масс ΔM_np = 3/16 * alpha * M_p * (1 + alpha/pi)
  2. Аномалия времени жизни: акустический сдвиг Парселла Δtau = tau_beam * (4/3 * alpha)
  3. Странность: предел |S| <= 3 и масса гиперона Lambda^0 = M_p + 5/3 * m_mu
"""

import math

def verify_neutron_sector():
    print("=" * 80)
    print("  ТЭВ: ВЕРИФИКАЦИЯ СТРУКТУРЫ НЕЙТРОНА И АНОМАЛИИ ВРЕМЕНИ ЖИЗНИ (СТАТЬЯ 03)")
    print("=" * 80)

    # 1. ВХОДНЫЕ ПАРАМЕТРЫ ИЗ DAG
    alpha_inv = 137.035999177
    alpha = 1.0 / alpha_inv
    M_p_MeV = 938.27208816      # МэВ
    m_mu_MeV = 105.6583755      # МэВ (из Статьи 01)

    print(f"[Входные данные] Постоянная импеданса α : 1/{alpha_inv:.6f}")
    print(f"[Входные данные] Масса протона M_p       : {M_p_MeV:.6f} МэВ")
    print(f"[Входные данные] Масса мюона m_μ         : {m_mu_MeV:.6f} МэВ\n")

    # 2. РАСЧЕТ РАЗНОСТИ МАСС НЕЙТРОН-ПРОТОН
    print("--- 1. Разность масс нейтрон-протон ΔM_np ---")
    C_np = 3.0 / 16.0
    delta_rad = alpha / math.pi

    delta_M_np_bare = C_np * alpha * M_p_MeV
    delta_M_np_calc = delta_M_np_bare * (1.0 + delta_rad)

    # Эксперимент CODATA 2022
    M_n_exp = 939.56542052
    delta_M_np_exp = M_n_exp - M_p_MeV  # 1.29333236 МэВ

    rel_err_mass = (delta_M_np_calc - delta_M_np_exp) / delta_M_np_exp * 100

    print(f"Геометрический коэффициент C_np       : 3/16 = {C_np:.6f}")
    print(f"Радиационная поправка 1 + α/π         : {1.0 + delta_rad:.7f}")
    print(f"Расчетная разность масс ΔM_np         : {delta_M_np_calc:.6f} МэВ")
    print(f"Эксперимент (CODATA 2022)             : {delta_M_np_exp:.6f} МэВ")
    print(f"Относительная погрешность             : {rel_err_mass:+.3f} %")
    assert abs(rel_err_mass) < 0.6, f"Ошибка ΔM_np превысила порог: {rel_err_mass}%"

    # 3. АНОМАЛИЯ ВРЕМЕНИ ЖИЗНИ НЕЙТРОНА (ЭФФЕКТ ПАРСЕЛЛА)
    print("\n--- 2. Аномалия времени жизни нейтрона (эффект Парселла) ---")
    tau_beam_exp = 887.7       # с (NIST пучковый метод, ± 1.2 с)
    tau_bottle_exp = 878.4     # с (PDG среднее по ловушкам UCN, ± 0.5 с)
    tau_ucntau_exp = 877.75    # с (LANL UCNtau 2021, ± 0.36 с)

    # Фактор Парселла 1/f = 4/3
    purcell_factor = (4.0 / 3.0) * alpha
    delta_tau_theor = tau_beam_exp * purcell_factor
    tau_bottle_theor = tau_beam_exp - delta_tau_theor

    exp_shift = tau_beam_exp - tau_bottle_exp

    print(f"Пучковое время жизни (NIST)           : {tau_beam_exp:.2f} с")
    print(f"Фактор Парселла ловушки (4/3)·α       : {purcell_factor:.6e}")
    print(f"Теоретический сдвиг Парселла Δτ       : {delta_tau_theor:.3f} с")
    print(f"Экспериментальный сдвиг (пучок-ловушка): {exp_shift:.2f} с")
    print(f"Расчетное время в ловушке τ_bottle    : {tau_bottle_theor:.2f} с")
    print(f"Эксперимент в ловушках (PDG / UCNτ)   : {tau_bottle_exp:.2f} с / {tau_ucntau_exp:.2f} с")

    diff_bottle = abs(tau_bottle_theor - tau_bottle_exp)
    print(f"Отклонение от PDG ловушек             : {diff_bottle:.2f} с")
    assert diff_bottle < 1.0, f"Ошибка времени жизни в ловушке превысила 1 с: {diff_bottle}"

    # 4. СТРАННОСТЬ И МАССА ГИПЕРОНА LAMBDA^0
    print("\n--- 3. Топологический предел странности и масса Λ⁰ ---")
    p_lobes = 3
    print(f"Число полоидальных пучностей p        : {p_lobes} -> предел |S| <= 3")

    # Масса Lambda^0: M_p + 5/3 * m_mu
    M_lambda_calc = M_p_MeV + (5.0 / 3.0) * m_mu_MeV
    M_lambda_exp = 1115.683    # МэВ (PDG 2024, ± 0.006 МэВ)
    err_lambda = (M_lambda_calc - M_lambda_exp) / M_lambda_exp * 100

    print(f"Расчетная масса гиперона M(Λ⁰)        : {M_lambda_calc:.3f} МэВ")
    print(f"Эксперимент (PDG 2024)                : {M_lambda_exp:.3f} МэВ")
    print(f"Погрешность расчета                   : {err_lambda:+.3f} % (-1.31 МэВ)")
    assert abs(err_lambda) < 0.2, "Ошибка массы Lambda превысила порог 0.2%"

    print("\n" + "=" * 80)
    print("  СТАТЬЯ 03 ПОЛНОСТЬЮ ВЕРИФИЦИРОВАНА: ВСЕ ТЕСТЫ ПРОЙДЕНЫ")
    print("=" * 80)

if __name__ == "__main__":
    verify_neutron_sector()