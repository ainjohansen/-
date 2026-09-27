#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
ТОПОЛОГИЧЕСКАЯ ЭЛАСТОДИНАМИКА ВАКУУМА (ТЭВ)
Модуль CAE-06: ЧЕСТНЫЙ ГЕОМЕТРИЧЕСКИЙ ВЫВОД ИЗ РАССЛОЕНИЯ ХОПФА S^3 -> S^2
БЕЗ кругового зашивания cos(theta) в вероятности!

Физический механизм:
  1. Векторы детекторов a, b на S^2 поднимаются на 3-сферу S^3 в спиноры xi_a, xi_b.
  2. На S^3 расстояние равно theta/2 (двойное накрытие SU(2) -> SO(3)).
  3. Амплитуда перекрытия спиноров на S^3 вычисляется через эрмитово произведение в C^2.
  4. Плотность энергии ~ квадрату модуля амплитуды: P_|| = |<xi_a|xi_b>|^2 = cos^2(theta/2).
  5. Разность каналов P_perp - P_|| дает в точности -cos(theta) через тригонометрию двойного угла.
================================================================================
"""

import math
import numpy as np

def hopf_lift_to_spinor(nx, ny, nz):
    """
    Точный подъем единичного 3D-вектора n in S^2 на 3-сферу S^3 (спинор Хопфа in C^2).
    Никакой магии: чистая стереографическая параметризация сферы.
    """
    # Углы сферических координат на S^2
    theta = math.acos(np.clip(nz, -1.0, 1.0))
    phi = math.atan2(ny, nx)

    # Спинор на S^3: половинный угол theta/2 !
    xi_1 = complex(math.cos(theta / 2.0), 0.0)
    xi_2 = complex(math.cos(phi) * math.sin(theta / 2.0), math.sin(phi) * math.sin(theta / 2.0))
    return np.array([xi_1, xi_2])

def verify_honest_hopf_derivation():
    print("=" * 82)
    print("  ТЭВ CAE-06: ЧЕСТНЫЙ ВЫВОД КОРРЕЛЯТОРА ИЗ СПИНОРНОЙ МЕТРИКИ ХОПФА (БЕЗ ТАВТОЛОГИЙ)")
    print("=" * 82)

    test_angles = [0.0, np.pi / 4.0, np.pi / 2.0, 3.0 * np.pi / 4.0, np.pi]
    E_calculated = {}

    print("[*] Шаг 1: Подъем направлений детекторов a, b на 3-сферу S^3...")
    print("[*] Шаг 2: Вычисление квадратов модулей спинорных амплитуд |<ξ_a | ξ_b>|² в C²...")
    print("[*] Шаг 3: Разность параллельного и антипараллельного каналов на S^3:\n")

    for th in test_angles:
        # Вектор детектора A вдоль оси Z
        ax, ay, az = 0.0, 0.0, 1.0
        # Вектор детектора B повернут на угол th в плоскости XZ
        bx, by, bz = math.sin(th), 0.0, math.cos(th)
        # Вектор детектора B с инвертированной осью (-b)
        mbx, mby, mbz = -bx, -by, -bz

        # 1. Подъем векторов с S^2 на 3-сферу S^3 (получаем комплексные спиноры)
        xi_a  = hopf_lift_to_spinor(ax, ay, az)
        xi_b  = hopf_lift_to_spinor(bx, by, bz)
        xi_mb = hopf_lift_to_spinor(mbx, mby, mbz)

        # 2. Вычисление эрмитова скалярного произведения спиноров на S^3 (в C^2):
        # <xi_a | xi_b> = conj(xi_a) . xi_b
        overlap_parallel = np.vdot(xi_a, xi_b)
        overlap_antiparallel = np.vdot(xi_a, xi_mb)

        # 3. Энергетические интенсивности (квадраты модулей амплитуд)
        P_parallel = np.abs(overlap_parallel) ** 2
        P_antiparallel = np.abs(overlap_antiparallel) ** 2

        # 4. Коррелятор как разность мод поглощения энергии в упругой среде:
        # Для синглета (где спины при рождении компенсированы):
        E_hopf = P_antiparallel - P_parallel
        E_calculated[th] = E_hopf

        # Сравниваем с ожидаемым значением
        target_cos = -math.cos(th)
        diff = abs(E_hopf - target_cos)

        print(f"  Угол θ = {math.degrees(th):5.1f}° | P_|| = {P_parallel:.5f} | P_⊥ = {P_antiparallel:.5f} | E_hopf = {E_hopf:+.5f} | -cos(θ) = {target_cos:+.5f} | Ошибка = {diff:.1e}")
        assert diff < 1e-12, f"Сбой вывода на угле {math.degrees(th)}°"

    # ВЫЧИСЛЕНИЕ ПАРАМЕТРА БЕЛЛА-CHSH И ПРЕДЕЛА ЦИРЕЛЬСОНА
    print("\n--- Проверка нарушения классического предела и выход на предел Цирельсона ---")
    E_45  = E_calculated[np.pi / 4.0]
    E_135 = E_calculated[3.0 * np.pi / 4.0]

    # S = | E(45°) - E(135°) + E(45°) + E(45°) |
    S_chsh = abs(E_45 - E_135 + E_45 + E_45)
    S_tsirelson = 2.0 * math.sqrt(2.0)
    diff_S = abs(S_chsh - S_tsirelson)

    print(f"  Вычисленный из метрики Хопфа параметр CHSH S : {S_chsh:.8f}")
    print(f"  Аналитический предел Цирельсона 2√2         : {S_tsirelson:.8f}")
    print(f"  Классический локальный предел Белла         : 2.00000000 (Превышение: +41.4%)")
    print(f"  Абсолютная невязка                          : {diff_S:.2e}")

    assert diff_S < 1e-12, "Ошибка достижения предела Цирельсона"

    print("\n" + "=" * 82)
   
if __name__ == "__main__":
    verify_honest_hopf_derivation()
