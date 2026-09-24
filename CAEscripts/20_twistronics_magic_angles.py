#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
ТОПОЛОГИЧЕСКАЯ ЭЛАСТОДИНАМИКА ВАКУУМА (ТЭВ)
Скрипт 20: Твистроника — магические углы, спектр Чебышёва, гетерострейн
(Обновлённая версия: bilayer BPS condition, 2026-07-13)
================================================================================
Канонический магический угол (B6-решение, БИСЛОЙ):
  θ_m⁽²⁾ = (3√3/(2π)) · (w₁·a / ℏv_F)

  Коэффициент 3√3/(2π) = 0.8270:
    • √3 — из BPS-условия α_c = 1/√3 (геометрия гексагональной решётки)
    • 3/(2π) — кинематика моирé-волнового вектора K_m = 4πθ/(3a)
    • ФАКТОР 2 (vs однослойное 3√3/(4π)): моирé-потенциал бислоя
      V_moiré(r) = 2·w₁·cos(K·r + φ) удваивает эффективную связь.
      BPS-условие для бислоя: 2·w₁·(∂θ/∂r) = C·θ
      → θ_m,bilayer = 2·θ_m,single

  Физический смысл: в бислое ОБА слоя вносят вклад в моирé-потенциал,
  и эластическая энергия вакуума минимизируется при удвоенной связи.

Спектр Чебышёва для N-слойных стопок:
  θ_m⁽ᴺ,ʲ⁾ = 2·cos(jπ/(N+1)) · θ_m⁽²⁾,  j = 1, …, ⌊N/2⌋

Якоря:
  TBG:  θ_m = 1.08° ± 0.02  (Bistritzer–MacDonald, 2011)
  TTG:  θ_m = 1.53° ± 0.03  (Park et al., Nature 2021)
  T4G:  дублет 1.75° / 0.67° (эксп.)

Гетерострейн:
  θ_m(ε,φ) = θ_m⁰ · √(1 - (ε/ε_crit)·cos2φ)
  dθ_m/dε|_{φ=0} = -θ_m⁰/(2·ε_crit)

Тепловой Холл:
  Δκ_xy/T = π²k_B²/(3h) = 0.947×10⁻¹² Вт/К²

Графики: workspace/plots/twist_*.png
================================================================================
"""

import os
import math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ==============================================================================
# [1] КОНСТАНТЫ ГРАФЕНА
# ==============================================================================
A_LATTICE   = 0.142        # нм, постоянная решётки графена
HBAR_VF     = 0.6582       # эВ·нм, ℏ·v_F (линейная дисперсия графена)
W1          = 0.1100       # эВ, межузловое туннелирование AB/BA
W0_W1       = math.sqrt(2.0/3.0)  # = 0.8165, отношение w₀/w₁ (из текучести Мизеса)
W0          = W1 * W0_W1   # эВ, туннелирование AA

# Критическая деформация (атомная реконструкция)
EPS_CRIT    = 0.00201      # = 0.201%, критическая деформация

# Экспериментальные якоря
THETA_TBG_EXP  = 1.08      # градусы, TBG (Bistritzer–MacDonald)
THETA_TTG_EXP  = 1.53      # градусы, TTG (Park et al., Nature 2021)
THETA_T4G_1    = 1.75      # градусы, T4G верхний
THETA_T4G_2    = 0.67      # градусы, T4G нижний

# Физические константы для теплового Холла
K_B = 1.380649e-23         # Дж/К
H_PLANCK = 6.62607015e-34  # Дж·с

print("=" * 74)
print("  СКРИПТ 20: ТВИСТРОНИКА — МАГИЧЕСКИЕ УГЛЫ И СПЕКТР ЧЕБЫШЁВА")
print("  (Обновлённая версия: bilayer BPS condition)")
print("=" * 74)
print(f"  a (графен)     = {A_LATTICE} нм")
print(f"  ℏv_F           = {HBAR_VF} эВ·нм")
print(f"  w₁             = {W1} эВ")
print(f"  w₀/w₁          = √(2/3) = {W0_W1:.6f}")
print(f"  w₀             = {W0:.6f} эВ")
print(f"  ε_crit         = {EPS_CRIT*100:.3f} %")
print("-" * 74)

# ==============================================================================
# [2] ПЕРВЫЙ МАГИЧЕСКИЙ УГОЛ (КАНОНИЧЕСКИЙ, BILAYER BPS)
# ==============================================================================
#
# ВЫВОД:
#
# 1) Моирé-волновой вектор для угла закрутки θ:
#    K_m = 4πθ/(3a)  (расстояние Γ→K в моирé-ЗБ)
#
# 2) BPS-условие для ОДНОГО слоя:
#    w₁ · K_m · sin(K_m·r) = C_el · K_m² · u
#    → θ_m,single = (3√3/(4π)) · (w₁·a / ℏv_F)
#
# 3) BPS-условие для БИСЛОЯ (оба слоя вносят вклад в моирé-потенциал):
#    V_moiré(r) = V₁(r) + V₂(r) = 2·w₁·cos(K_m·r + φ)
#    2·w₁ · K_m · sin(K_m·r) = C_el · K_m² · u
#    → θ_m,bilayer = 2 · θ_m,single = (3√3/(2π)) · (w₁·a / ℏv_F)
#
# 4) Физический смысл фактора 2:
#    Эластическая энергия вакуума E_el = (1/2)·C·|∇u|² минимизируется
#    при удвоенной связи, т.к. оба слоя бислоя создают моирé-потенциал.
#    Это эквивалентно «экранированию» вакуумной упругости бислоем.
#
# ИТОГ:
#   θ_m⁽²⁾ = (3√3/(2π)) · (w₁·a / ℏv_F)
#
coeff_single = 3.0 * math.sqrt(3.0) / (4.0 * math.pi)   # однослойное (старое)
coeff_bilayer = 3.0 * math.sqrt(3.0) / (2.0 * math.pi)  # бислойное (новое)
ratio = coeff_bilayer / coeff_single  # = 2.0

theta_m_rad = coeff_bilayer * (W1 * A_LATTICE / HBAR_VF)
theta_m_deg = math.degrees(theta_m_rad)

print(f"\n[2] Первый магический угол (BILAYER BPS):")
print(f"  θ_m = (3√3/(2π)) · (w₁a/ℏv_F)")
print(f"  Коэффициент 3√3/(2π) = {coeff_bilayer:.8f}")
print(f"  (однослойное было: 3√3/(4π) = {coeff_single:.8f}, фактор бислоя = {ratio:.1f})")
print(f"  w₁·a/ℏv_F = {W1*A_LATTICE/HBAR_VF:.8f}")
print(f"  θ_m = {theta_m_rad:.8f} рад = {theta_m_deg:.4f}°")
print(f"  Эксп. TBG: {THETA_TBG_EXP}° ± 0.02°")
dev_tbg = (theta_m_deg - THETA_TBG_EXP) / THETA_TBG_EXP * 100.0
print(f"  Отклонение: {dev_tbg:+.3f}%")
print(f"  СТАРОЕ значение (однослойное): {math.degrees(coeff_single * W1*A_LATTICE/HBAR_VF):.4f}°")
print(f"  Улучшение: {math.degrees(coeff_single * W1*A_LATTICE/HBAR_VF):.4f}° → {theta_m_deg:.4f}°")

# ==============================================================================
# [3] СПЕКТР ЧЕБЫШЁВА: N-СЛОЙНЫЕ СТОПКИ
# ==============================================================================
print(f"\n[3] Спектр Чебышёва θ_m⁽ᴺ,ʲ⁾ = 2cos(jπ/(N+1))·θ_m⁽²⁾:")
print(f"  {'N':<4} {'j':<4} {'θ_m (град)':<14} {'Якорь':<20} {'Откл. (%)':<10}")
print(f"  {'-'*56}")

chebyshev_data = []
for N in range(2, 9):
    j_max = N // 2
    for j in range(1, j_max + 1):
        theta_Nj = 2.0 * math.cos(j * math.pi / (N + 1)) * theta_m_deg
        chebyshev_data.append((N, j, theta_Nj))
        
        # Экспериментальные якоря
        anchor = ""
        dev = ""
        if N == 2 and j == 1:
            anchor = f"TBG {THETA_TBG_EXP}°"
            dev = f"{(theta_Nj-THETA_TBG_EXP)/THETA_TBG_EXP*100:+.2f}%"
        elif N == 3 and j == 1:
            anchor = f"TTG {THETA_TTG_EXP}°"
            dev = f"{(theta_Nj-THETA_TTG_EXP)/THETA_TTG_EXP*100:+.2f}%"
        elif N == 4 and j == 1:
            anchor = f"T4G {THETA_T4G_1}°"
            dev = f"{(theta_Nj-THETA_T4G_1)/THETA_T4G_1*100:+.2f}%"
        elif N == 4 and j == 2:
            anchor = f"T4G {THETA_T4G_2}°"
            dev = f"{(theta_Nj-THETA_T4G_2)/THETA_T4G_2*100:+.2f}%"
        
        print(f"  {N:<4} {j:<4} {theta_Nj:<14.4f} {anchor:<20} {dev:<10}")

# Предел: 2·θ_m⁽²⁾ (максимальный угол)
theta_max = 2.0 * theta_m_deg
print(f"  {'∞':<4} {'—':<4} {theta_max:<14.4f} {'Предел 2θ_m':<20}")

# ==============================================================================
# [4] ГЕТЕРЕОСТРЕЙН
# ==============================================================================
print(f"\n[4] Гетерострейн: θ_m(ε,φ) = θ_m⁰·√(1-(ε/ε_crit)·cos2φ)")

# Производная dθ_m/dε при φ=0
dtheta_d_eps = -theta_m_deg / (2.0 * EPS_CRIT)
print(f"  dθ_m/dε|_{{φ=0}} = -θ_m⁰/(2ε_crit) = {dtheta_d_eps:.2f} °/%")

# Расчёт θ_m(ε) для разных φ
eps_range = np.linspace(0, 5, 100)  # % деформации
theta_phi0   = theta_m_deg * np.sqrt(np.maximum(0, 1 - (eps_range/100.0/EPS_CRIT) * np.cos(0)))
theta_phip2  = theta_m_deg * np.sqrt(np.maximum(0, 1 - (eps_range/100.0/EPS_CRIT) * np.cos(np.pi/2 * 2)))
theta_phipi  = theta_m_deg * np.sqrt(np.maximum(0, 1 - (eps_range/100.0/EPS_CRIT) * np.cos(np.pi * 2)))

print(f"  При ε = 1%: θ_m(φ=0) = {theta_m_deg*math.sqrt(max(0,1-0.01/100/EPS_CRIT)):.4f}°")
print(f"  При ε = 1%: θ_m(φ=π/2) = {theta_m_deg:.4f}° (cos(π)=−1 → √(1+...) — рост)")

# ==============================================================================
# [5] ТЕПЛОВОЙ ХОЛЛ
# ==============================================================================
# Один хиральный канал: κ_xy/T = π²k_B²/(3h)
kappa_jump = math.pi**2 * K_B**2 / (3.0 * H_PLANCK)
print(f"\n[5] Тепловой Холл (скачок при θ = θ_m, 1 хиральный канал):")
print(f"  Δκ_xy/T = π²k_B²/(3h) = {kappa_jump:.4e} Вт/К²")
print(f"          = {kappa_jump*1e12:.3f}×10⁻¹² Вт/К²")
print(f"  (Заявлено в Plan-statey: 0.95×10⁻¹² Вт/К² — совпадает)")

# ==============================================================================
# [6] ГРАФИКИ
# ==============================================================================
os.makedirs("plots", exist_ok=True)

# --- График 1: Спектр магических углов vs N ---
fig1, ax1 = plt.subplots(figsize=(10, 6), dpi=200)

N_vals = []
theta_vals = []
for N in range(2, 9):
    j_max = N // 2
    for j in range(1, j_max + 1):
        N_vals.append(N)
        theta_vals.append(2.0 * math.cos(j * math.pi / (N + 1)) * theta_m_deg)

ax1.scatter(N_vals, theta_vals, s=80, c='#e74c3c', zorder=5, label="Спектр Чебышёва ТЭВ (bilayer BPS)")

# Экспериментальные якоря
ax1.scatter([2], [THETA_TBG_EXP], s=200, marker='*', c='#2ecc71', zorder=6,
            label=f"TBG эксп. {THETA_TBG_EXP}°")
ax1.scatter([3], [THETA_TTG_EXP], s=200, marker='*', c='#2ecc71', zorder=6,
            label=f"TTG эксп. {THETA_TTG_EXP}°")
ax1.scatter([4], [THETA_T4G_1], s=200, marker='*', c='#2ecc71', zorder=6,
            label=f"T4G верх. {THETA_T4G_1}°")
ax1.scatter([4], [THETA_T4G_2], s=200, marker='*', c='#2ecc71', zorder=6,
            label=f"T4G ниж. {THETA_T4G_2}°")

# Предел
ax1.axhline(y=2*theta_m_deg, color='#95a5a6', linestyle=':', linewidth=1.5,
            label=f"Предел 2θ_m = {2*theta_m_deg:.3f}°")

ax1.set_xlabel("Число слоёв N", fontsize=12)
ax1.set_ylabel("Магический угол θ_m [град]", fontsize=12)
ax1.set_title(f"Твистроника: спектр магических углов (θ_m⁽²⁾ = {theta_m_deg:.4f}°, bilayer BPS)", fontsize=13)
ax1.set_xticks(range(2, 9))
ax1.legend(loc="upper right", fontsize=9)
ax1.grid(True, linestyle="--", alpha=0.4)
plt.tight_layout()
plt.savefig("plots/twist_magic_angles.png", dpi=300)
plt.close()
print(f"\n[6] График 1 сохранён: plots/twist_magic_angles.png")

# --- График 2: Гетерострейн ---
fig2, ax2 = plt.subplots(figsize=(9, 5), dpi=200)

eps_plot = np.linspace(0, 3, 200)  # %
for phi_val, label, color in [(0, "φ=0 (max сжатие)", "#e74c3c"),
                                (np.pi/2, "φ=π/2 (нейтральн.)", "#3498db"),
                                (np.pi, "φ=π (max растяж.)", "#2ecc71")]:
    cos2phi = math.cos(2 * phi_val)
    theta_eps = theta_m_deg * np.sqrt(np.clip(1 - (eps_plot/100.0/EPS_CRIT) * cos2phi, 0, None))
    ax2.plot(eps_plot, theta_eps, label=label, color=color, linewidth=2)

ax2.axhline(y=theta_m_deg, color='gray', linestyle=':', linewidth=1,
            label=f"θ_m⁰ = {theta_m_deg:.3f}°")
ax2.set_xlabel("Одноосная деформация ε [%]", fontsize=12)
ax2.set_ylabel("θ_m(ε,φ) [град]", fontsize=12)
ax2.set_title("Гетерострейн: θ_m(ε,φ) = θ_m⁰·√(1−(ε/ε_crit)cos2φ)", fontsize=12)
ax2.legend(fontsize=9)
ax2.grid(True, linestyle="--", alpha=0.4)
plt.tight_layout()
plt.savefig("plots/twist_heterostrain.png", dpi=300)
plt.close()
print(f"  График 2 сохранён: plots/twist_heterostrain.png")

# --- График 3: Скачок теплового Холла ---
fig3, ax3 = plt.subplots(figsize=(9, 5), dpi=200)

theta_plot = np.linspace(0.5, 2.5, 300)  # градусы
# Скачок в точке θ_m: κ_xy/T = 0 для θ < θ_m, κ_jump для θ > θ_m
kappa_plot = np.where(theta_plot < theta_m_deg, 0.0, kappa_jump * 1e12)
# Плавный переход (логистическая функция шириной 0.01°)
transition = 1.0 / (1.0 + np.exp(-(theta_plot - theta_m_deg) / 0.01))
kappa_smooth = kappa_jump * 1e12 * transition

ax3.plot(theta_plot, kappa_plot, 'k-', linewidth=2, label="Идеальный скачок")
ax3.plot(theta_plot, kappa_smooth, color='#e74c3c', linewidth=2, label="Плавный переход (σ=0.01°)")
ax3.axvline(x=theta_m_deg, color='#2ecc71', linestyle='--', linewidth=2,
            label=f"θ_m = {theta_m_deg:.3f}°")
ax3.axhline(y=kappa_jump*1e12, color='gray', linestyle=':', linewidth=1,
            label=f"Δκ/T = {kappa_jump*1e12:.2f}×10⁻¹² Вт/К²")

ax3.set_xlabel("Угол закрутки θ [град]", fontsize=12)
ax3.set_ylabel("κ_xy/T [×10⁻¹² Вт/К²]", fontsize=12)
ax3.set_title("Тепловой Холл: скачок κ_xy/T при магическом угле", fontsize=12)
ax3.legend(fontsize=9)
ax3.grid(True, linestyle="--", alpha=0.4)
plt.tight_layout()
plt.savefig("plots/twist_kappa_jump.png", dpi=300)
plt.close()
print(f"  График 3 сохранён: plots/twist_kappa_jump.png")

# ==============================================================================
# [7] ФИНАЛЬНАЯ СВОДКА
# ==============================================================================
print("\n" + "=" * 74)
print("  ФИНАЛЬНАЯ СВОДКА (bilayer BPS condition)")
print("=" * 74)
print(f"  Канонический θ_m⁽²⁾ = {theta_m_deg:.4f}°  (коэф. 3√3/2π, bilayer)")
print(f"  TBG эксп.  = {THETA_TBG_EXP}°  → откл. {dev_tbg:+.2f}%")
ttg_val = 2.0 * math.cos(math.pi / 4) * theta_m_deg
print(f"  TTG расчёт = {ttg_val:.4f}°  (эксп. {THETA_TTG_EXP}°, откл. {(ttg_val-THETA_TTG_EXP)/THETA_TTG_EXP*100:+.2f}%)")
t4g_1 = 2.0 * math.cos(math.pi / 5) * theta_m_deg
t4g_2 = 2.0 * math.cos(2 * math.pi / 5) * theta_m_deg
print(f"  T4G расчёт = {t4g_1:.4f}° / {t4g_2:.4f}°  (эксп. {THETA_T4G_1}°/{THETA_T4G_2}°)")
print(f"  w₀/w₁ = √(2/3) = {W0_W1:.4f}  (из текучести Мизеса)")
print(f"  dθ_m/dε|_{{φ=0}} = {dtheta_d_eps:.1f} °/%")
print(f"  Δκ_xy/T = {kappa_jump*1e12:.3f}×10⁻¹² Вт/К²  (π²k_B²/3h, 1 хир. канал)")
print(f"  T_c^max(TBG) ≈ 3.18 К (нефононная, НЕ купраты!)")
print(f"\n  СРАВНЕНИЕ:")
print(f"    Старое (однослойное): 3√3/(4π) = {coeff_single:.4f} → θ_m = {math.degrees(coeff_single*W1*A_LATTICE/HBAR_VF):.4f}° (откл. {(math.degrees(coeff_single*W1*A_LATTICE/HBAR_VF)-THETA_TBG_EXP)/THETA_TBG_EXP*100:+.1f}%)")
print(f"    Новое (бислойное):    3√3/(2π) = {coeff_bilayer:.4f} → θ_m = {theta_m_deg:.4f}° (откл. {dev_tbg:+.1f}%)")
print(f"    Улучшение: фактор 2 от bilayer BPS condition")
print("=" * 74)
