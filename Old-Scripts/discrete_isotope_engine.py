#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
ТОПОЛОГИЧЕСКАЯ ЭЛАСТОДИНАМИКА ВАКУУМА (ТЭВ)
Дискретный решатель изотопов: 3D-упаковка анизотропных трилистников T(3,2)
================================================================================
Каждый изотоп рассчитывается индивидуально как дискретная кристаллическая 
молекула из Z сплюснутых протонов и N квазисферических нейтронов на ГЦК-решетке.
================================================================================
"""

import numpy as np
import itertools

# Фундаментальные масштабы ТЭВ
ALPHA_INV = 137.035999084
ALPHA = 1.0 / ALPHA_INV
M_E_MEV = 0.51099895
E_0 = ALPHA_INV * M_E_MEV        # Квант Намбу: ~70.0245 МэВ
HBAR_C = 197.3269804             # МэВ * фм
D_0 = 1.125                      # Базисный шаг решетки (фм)

# Базовые кванты связей ТЭВ на узле решетки (r = d_0)
E_BOND_PN = 2.224575             # Базовая квантовая связка p-n (дейтрон), МэВ
E_BOND_NN = 1.812000             # Базовая связка n-n, МэВ
E_BOND_PP = 1.812000             # Ядерная связка p-p (без учета Кулона), МэВ
ETA_ALIGN = 0.25                 # Выигрыш при соосном смыкании сплюснутых граней C_3

def generate_fcc_lattice(num_shells=5):
    """Генерация узлов ГЦК (FCC) решетки плотнейшей упаковки."""
    points = []
    for x in range(-num_shells, num_shells + 1):
        for y in range(-num_shells, num_shells + 1):
            for z in range(-num_shells, num_shells + 1):
                if (x + y + z) % 2 == 0:
                    points.append(np.array([x, y, z]) * (D_0 / np.sqrt(2.0)))
    points.sort(key=lambda p: np.dot(p, p))
    return points

FCC_LATTICE = generate_fcc_lattice(6)

def build_discrete_isotope(Z, N):
    """
    Построение оптимального дискретного кластера из Z протонов и N нейтронов.
    Протоны имеют сплюснутую форму (вектор оси n_i), нейтроны квазисферичны.
    """
    A = Z + N
    if A < 1:
        return 0.0, 0.0, 0.0
    if A == 1:
        return 0.0, 0.0, 0.0

    # 1. Занимаем A центральных узлов ГЦК-решетки (минимизация объема/поверхности)
    coords = FCC_LATTICE[:A]
    
    # 2. Оптимальное распределение протонов и нейтронов по узлам
    # Кулоновское расталкивание вытесняет протоны наружу, нейтроны формируют остов
    types = [] # +1 для p, -1 для n
    if Z == N:
        # Для N=Z идеальное шахматное чередование p-n пар (конгруэнтность)
        types = [+1 if i % 2 == 0 else -1 for i in range(A)]
    else:
        # При N > Z избыточные нейтроны занимают внешнюю оболочку (шуба)
        radii_sq = [np.dot(p, p) for p in coords]
        sorted_indices = np.argsort(radii_sq)
        types = [-1] * A
        # Протоны размещаются во внутренне-симметричных узлах
        step = max(1, A // Z) if Z > 0 else 1
        p_count = 0
        for idx in sorted_indices:
            if p_count < Z:
                types[idx] = +1
                p_count += 1

    # 3. Ориентация сплюснутых осей трилистников (C_3 ось вдоль радиального луча)
    orientations = []
    for i, p in enumerate(coords):
        r_norm = np.linalg.norm(p)
        if r_norm > 1e-5:
            n_vec = p / r_norm
        else:
            n_vec = np.array([0.0, 0.0, 1.0])
        orientations.append(n_vec)

    # 4. Дискретный расчет парных взаимодействий
    E_nuclear_total = 0.0
    E_coulomb_total = 0.0
    
    for i in range(A):
        for j in range(i + 1, A):
            r_vec = coords[i] - coords[j]
            r_dist = np.linalg.norm(r_vec)
            
            # Проверяем, являются ли нуклоны ближайшими соседями на решетке
            if r_dist <= D_0 * 1.08:
                # Базовый ядерный квант связи в зависимости от пары
                t_i, t_j = types[i], types[j]
                if t_i * t_j == -1: # пара p-n
                    e_bond_base = E_BOND_PN
                elif t_i == 1 and t_j == 1: # пара p-p
                    e_bond_base = E_BOND_PP
                else: # пара n-n
                    e_bond_base = E_BOND_NN
                
                # Анизотропный фактор ориентации сплюснутых граней (только для протонов)
                u_ij = r_vec / r_dist
                align_factor = 1.0
                if t_i == +1 and t_j == +1:
                    cos_i = np.abs(np.dot(orientations[i], u_ij))
                    cos_j = np.abs(np.dot(orientations[j], u_ij))
                    align_factor += ETA_ALIGN * (cos_i * cos_j)
                elif (t_i == +1 or t_j == +1):
                    cos_val = np.abs(np.dot(orientations[i if t_i==1 else j], u_ij))
                    align_factor += 0.5 * ETA_ALIGN * cos_val
                
                E_nuclear_total += e_bond_base * align_factor

            # Точный кулоновский расчет между протонами на любых расстояниях
            if types[i] == +1 and types[j] == +1:
                e_coul = (ALPHA * HBAR_C) / r_dist # e^2 / r в МэВ
                E_coulomb_total += e_coul

    # 5. Энергия нулевых колебаний решетки (ZPE)
    E_zpe = 0.5 * E_0 * (A**(-1.0/3.0)) if A > 1 else 0.0

    # Полная энергия связи
    B_total = E_nuclear_total - E_coulomb_total - E_zpe
    if B_total < 0:
        B_total = 0.0
        
    return B_total, B_total / A, E_coulomb_total

def main():
    print("=" * 80)
    print("    ТОПОЛОГИЧЕСКАЯ ЭЛАСТОДИНАМИКА ВАКУУМА: ДИСКРЕТНЫЙ РЕШАТЕЛЬ ИЗОТОПОВ    ")
    print("=" * 80)
    print(f"Базис: ГЦК-решетка нуклонов (d_0 = {D_0} фм), сплюснутый протон C_3, сфера-нейтрон\n")

    # Сверка по ключевым изотопам
    benchmarks = [
        (1, 1, "H-2 (d)",   1.1123),
        (1, 2, "H-3 (t)",   2.8273),
        (2, 1, "He-3",      2.5727),
        (2, 2, "He-4 (a)",  7.0739),
        (3, 3, "Li-6",      5.3323),
        (6, 6, "C-12",      7.6801),
        (8, 8, "O-16",      7.9762),
        (20, 20, "Ca-40",   8.5513),
        (26, 30, "Fe-56",   8.7904),
        (28, 30, "Ni-58",   8.7321),
        (50, 70, "Sn-120",  8.5045),
        (82, 126, "Pb-208", 7.8675)
    ]

    print(f"{'Изотоп':<10} | {'Z':<3} | {'N':<3} | {'Эксперимент':<15} | {'ТЭВ (Дискретная)':<18} | {'Невязка Delta'}")
    print("-" * 80)
    
    for z, n, name, exp_val in benchmarks:
        b_tot, b_per_a, e_coul = build_discrete_isotope(z, n)
        delta = b_per_a - exp_val
        pct = (delta / exp_val) * 100.0
        print(f"{name:<10} | {z:<3} | {n:<3} | {exp_val:11.4f} МэВ/н | {b_per_a:11.4f} МэВ/н    | {delta:+8.4f} МэВ ({pct:+5.2f}%)")

    print("-" * 80)
    print("[+] Расчет завершен. Каждое ядро рассчитано как индивидуальный 3D-полиэдр.")
    print("=" * 80 + "\n")

if __name__ == "__main__":
    main()
