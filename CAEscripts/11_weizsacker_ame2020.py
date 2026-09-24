#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
ТОПОЛОГИЧЕСКАЯ ЭЛАСТОДИНАМИКА ВАКУУМА (ТЭВ)
Модуль 10: Каноническая верификация ядерного функционала по базе AME2020
================================================================================
Все коэффициенты являются точными проекциями девиатора напряжений континуума
на торе Клиффорда через фундаментальный масштаб E_0 = alpha^-1 * m_e.
Ноль подгоночных параметров.
================================================================================
"""

import os
import sys
import urllib.request
import numpy as np

# ==============================================================================
# 1. ФУНДАМЕНТАЛЬНЫЕ КОНСТАНТЫ И АНАЛИТИЧЕСКИЕ ИНВАРИАНТЫ ТЭВ
# ==============================================================================
M_E_MEV   = 0.51099895000     # Масса электрона (CODATA), МэВ
ALPHA_INV = 137.035999177   # Обратная постоянная тонкой структуры
ALPHA     = 1.0 / ALPHA_INV

# Базовый масштаб сдвиговой деформации (квант массы Намбу)
E_0 = ALPHA_INV * M_E_MEV   # ~ 70.025252 МэВ

# Канонические аналитические коэффициенты ТЭВ (0 свободных параметров)
A_V = (2.0 / 9.0) * E_0        # Объемный член (угол Лоде theta_0 = 2/9): 15.5610 МэВ
A_S = (1.0 / 4.0) * E_0        # Поверхностное натяжение (тор Клиффорда):  17.5061 МэВ
A_C = (27.0 / 20.0) * M_E_MEV  # Кулоновское отталкивание (3/5 * 9/4 * m_e): 0.6898 МэВ
A_A = (1.0 / 3.0) * E_0        # Объемная асимметрия (3 главные оси):     23.3415 МэВ
A_P = (1.0 / 6.0) * E_0        # Квант спаривания (1 степень свободы):    11.6708 МэВ

# ==============================================================================
# 2. ЗАГРУЗКА И ПАРСИНГ ОФИЦИАЛЬНОЙ БАЗЫ ДАННЫХ AME2020
# ==============================================================================
AME2020_URL = "https://www-nds.iaea.org/amdc/ame2020/mass_1.mas20"
LOCAL_FILENAME = "mass_1.mas20"

def download_ame2020():
    """Скачивание официальной таблицы масс AME2020 при ее отсутствии."""
    if not os.path.exists(LOCAL_FILENAME):
        print(f"[*] Скачивание официальной базы AME2020 с сервера МАГАТЭ...")
        try:
            urllib.request.urlretrieve(AME2020_URL, LOCAL_FILENAME)
            print(f"[+] База AME2020 успешно загружена: {LOCAL_FILENAME}")
        except Exception as e:
            print(f"[-] Ошибка загрузки базы: {e}")
            print("[!] Убедитесь в наличии интернет-соединения или положите файл mass_1.mas20 в папку.")
            sys.exit(1)

def parse_ame2020(min_a=16):
    """
    Парсинг таблицы AME2020.
    Формат AMDC: извлечение Z, A, N, Символа и экспериментальной B/A (МэВ/нуклон).
    """
    download_ame2020()
    nuclei = []
    
    with open(LOCAL_FILENAME, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()
        
    start_parsing = False
    for line in lines:
        if "1N-Z" in line or ("N-Z" in line and "MASS EXCESS" in line):
            start_parsing = True
            continue
        if not start_parsing or len(line) < 70:
            continue
            
        try:
            n_str = line[5:10].strip()
            z_str = line[10:15].strip()
            a_str = line[15:20].strip()
            sym   = line[20:24].strip()
            be_str = line[54:68].strip().replace("#", "")  # B/A в кэВ
            
            if not (n_str and z_str and a_str and be_str):
                continue
                
            N = int(n_str)
            Z = int(z_str)
            A = int(a_str)
            be_per_a = float(be_str) / 1000.0  # Перевод в МэВ/нуклон
            
            if A >= min_a and be_per_a > 0:
                nuclei.append({
                    "Z": Z, "A": A, "N": N, "sym": sym,
                    "BE_per_A_exp": be_per_a,
                    "BE_total_exp": be_per_a * A
                })
        except ValueError:
            continue
            
    return nuclei

# ==============================================================================
# 3. РАСЧЕТ ЭНЕРГИИ СВЯЗИ В ТЭВ
# ==============================================================================
def get_pairing_delta(A, Z, a_p_val):
    """Квант спаривания девиатора ТЭВ."""
    if A % 2 != 0:
        return 0.0
    elif Z % 2 == 0:  # Четно-четное ядро
        return + a_p_val / np.sqrt(A)
    else:             # Нечетно-нечетное ядро
        return - a_p_val / np.sqrt(A)

def calculate_tev(A, Z):
    """Каноническая 5-членная модель ТЭВ."""
    delta_p = get_pairing_delta(A, Z, A_P)
    b_vol   = A_V * A
    b_surf  = - A_S * (A**(2.0/3.0))
    b_coul  = - A_C * (Z * (Z - 1.0)) / (A**(1.0/3.0))
    b_asym  = - A_A * ((A - 2.0*Z)**2) / A
    b_pair  = delta_p
    
    b_total = b_vol + b_surf + b_coul + b_asym + b_pair
    return b_total, b_total / A

def fit_ols_reference(nuclei):
    """Свободный 5-параметрический МНК-фит для оценки предела регрессии."""
    X, y = [], []
    for n in nuclei:
        A, Z = n["A"], n["Z"]
        par_sign = 1.0 if (A % 2 == 0 and Z % 2 == 0) else (-1.0 if (A % 2 == 0 and Z % 2 != 0) else 0.0)
        row = [
            A,
            - (A**(2.0/3.0)),
            - (Z * (Z - 1.0)) / (A**(1.0/3.0)),
            - ((A - 2.0*Z)**2) / A,
            par_sign / np.sqrt(A)
        ]
        X.append(row)
        y.append(n["BE_total_exp"])
        
    coeffs, _, _, _ = np.linalg.lstsq(np.array(X), np.array(y), rcond=None)
    return coeffs

# ==============================================================================
# 4. ГЛАВНЫЙ ИСПОЛНИТЕЛЬНЫЙ БЛОК
# ==============================================================================
def main():
    print("=" * 80)
    print("      ТОПОЛОГИЧЕСКАЯ ЭЛАСТОДИНАМИКА ВАКУУМА (ТЭВ) — ВЕРИФИКАЦИЯ AME2020      ")
    print("=" * 80)
    
    # 1. Загрузка данных макроскопического домена (A >= 16)
    nuclei = parse_ame2020(min_a=16)
    n_count = len(nuclei)
    print(f"[+] Успешно загружено экспериментальных ядер (A >= 16): {n_count}")
    
    # 2. МНК-фит сравнения
    ols = fit_ols_reference(nuclei)
    
    # 3. Вычисление метрик ТЭВ
    exp_vals = np.array([n["BE_per_A_exp"] for n in nuclei])
    calc_tev = np.array([calculate_tev(n["A"], n["Z"])[1] for n in nuclei])
    
    diff_tev = calc_tev - exp_vals
    rms_tev  = np.sqrt(np.mean(diff_tev**2))
    mae_tev  = np.mean(np.abs(diff_tev))
    r2_tev   = (1.0 - np.sum(diff_tev**2) / np.sum((exp_vals - np.mean(exp_vals))**2)) * 100.0
    
    # Метрики МНК
    calc_ols = np.zeros(n_count)
    for i, n in enumerate(nuclei):
        A, Z = n["A"], n["Z"]
        par_sign = 1.0 if (A % 2 == 0 and Z % 2 == 0) else (-1.0 if (A % 2 == 0 and Z % 2 != 0) else 0.0)
        calc_ols[i] = (ols[0]*A - ols[1]*(A**(2.0/3.0)) - 
                       ols[2]*(Z*(Z-1))/(A**(1.0/3.0)) - 
                       ols[3]*((A-2*Z)**2)/A + ols[4]*par_sign/np.sqrt(A)) / A
    diff_ols = calc_ols - exp_vals
    rms_ols  = np.sqrt(np.mean(diff_ols**2))
    
    # Вывод результатов
    print("\n" + "-" * 80)
    print("1. СРАВНЕНИЕ КОЭФФИЦИЕНТОВ: ТЕОРИЯ ТЭВ vs ЭМПИРИЧЕСКИЙ ФИТ AME2020")
    print("-" * 80)
    print(f"{'Параметр':<20} | {'Формула ТЭВ':<22} | {'ТЭД (Теория)':<12} | {'МНК-фит AME'}")
    print("-" * 80)
    print(f"Объемный a_V         | (2/9) * E_0            | {A_V:11.4f} МэВ | {ols[0]:11.4f} МэВ")
    print(f"Поверхностный a_S    | (1/4) * E_0            | {A_S:11.4f} МэВ | {ols[1]:11.4f} МэВ")
    print(f"Кулоновский a_C      | (27/20) * m_e          | {A_C:11.4f} МэВ | {ols[2]:11.4f} МэВ")
    print(f"Асимметрия a_A       | (1/3) * E_0            | {A_A:11.4f} МэВ | {ols[3]:11.4f} МэВ")
    print(f"Спаривание a_P       | (1/6) * E_0            | {A_P:11.4f} МэВ | {ols[4]:11.4f} МэВ")
    print("-" * 80)
    
    print("\n" + "-" * 80)
    print(f"2. МЕТРИКИ ТОЧНОСТИ НА ВСЕХ ЯДРАХ (A >= 16, выборка = {n_count} ядер)")
    print("-" * 80)
    print(f"Среднеквадратичное отклонение RMS (ТЭВ):      {rms_tev:9.4f} МэВ/нуклон")
    print(f"Средняя абсолютная ошибка MAE (ТЭВ):           {mae_tev:9.4f} МэВ/нуклон")
    print(f"Коэффициент детерминации R^2 (ТЭВ):            {r2_tev:9.3f} %")
    print(f"Предел наилучшего эмпирического фита (RMS):    {rms_ols:9.4f} МэВ/нуклон")
    print("-" * 80)
    
    # Точечная сверка по реперным ядрам
    benchmark_nuclei = [
        (2, 4, "He-4"),
        (6, 12, "C-12"),
        (8, 16, "O-16"),
        (20, 40, "Ca-40"),
        (26, 56, "Fe-56"),
        (28, 58, "Ni-58"),
        (50, 120, "Sn-120"),
        (82, 208, "Pb-208"),
        (92, 238, "U-238")
    ]
    
    all_nuclei = parse_ame2020(min_a=4)
    n_dict = {(n["Z"], n["A"]): n["BE_per_A_exp"] for n in all_nuclei}
    
    print("\n" + "-" * 80)
    print("3. ТОЧЕЧНАЯ СВЕРКА ПО РЕПЕРНЫМ ЯДРАМ (Энергия связи на нуклон, МэВ)")
    print("-" * 80)
    print(f"{'Ядро':<10} | {'Эксперимент':<15} | {'Расчет ТЭВ':<15} | {'Невязка Delta'}")
    print("-" * 80)
    
    for z_b, a_b, name_b in benchmark_nuclei:
        if (z_b, a_b) in n_dict:
            exp_v = n_dict[(z_b, a_b)]
            _, calc_v = calculate_tev(a_b, z_b)
            delta = calc_v - exp_v
            pct = (delta / exp_v) * 100.0
            print(f"{name_b:<10} | {exp_v:11.4f} МэВ   | {calc_v:11.4f} МэВ   | {delta:+8.4f} МэВ ({pct:+5.2f}%)")
            
    print("-" * 80)
    print("[+] Верификация завершена. Ноль свободных параметров.")
    print("=" * 80 + "\n")

if __name__ == "__main__":
    main()
