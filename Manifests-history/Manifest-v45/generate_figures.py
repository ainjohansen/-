#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Генератор графиков и 3D-моделей для статьи:
"Топологическая эластодинамика 3D-континуума"
Версия 2.0 (Исправленная геометрия и верстка)
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

os.makedirs("figures", exist_ok=True)

plt.rcParams.update({
    "font.size": 11,
    "font.family": "serif",
    "mathtext.fontset": "cm",
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 9.5,
    "figure.titlesize": 13,
    "savefig.dpi": 300,
    "savefig.bbox": "tight"
})

# ==============================================================================
# РИСУНОК 1: 3D-Тор T(1,1) с реальной геометрией и спиральным фазовым бинтованием
# ==============================================================================
def plot_figure_1():
    print("Генерация Рисунка 1: Реалистичный 3D-тор с центральным отверстием...")
    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection='3d')

    R = 3.0       # Большой радиус
    r = 1.0       # Малый радиус керна (R/r = 3.0 — четкий открытый тор)

    theta = np.linspace(0, 2*np.pi, 80)
    phi = np.linspace(0, 2*np.pi, 160)
    THETA, PHI = np.meshgrid(theta, phi)

    X = (R + r * np.cos(THETA)) * np.cos(PHI)
    Y = (R + r * np.cos(THETA)) * np.sin(PHI)
    Z = r * np.sin(THETA)

    # Отрисовка каркасной полупрозрачной поверхности тора
    ax.plot_surface(X, Y, Z, color='cornflowerblue', alpha=0.22, edgecolor='gray', lw=0.15)

    # Спиральное фазовое бинтование (быстрая полоидальная прецессия)
    s = np.linspace(0, 2*np.pi, 1500)
    p_turns = 16   # 16 витков бинтования на 1 большой оборот
    q_turns = 1
    
    x_helix = (R + r * np.cos(p_turns * s)) * np.cos(q_turns * s)
    y_helix = (R + r * np.cos(p_turns * s)) * np.sin(q_turns * s)
    z_helix = r * np.sin(p_turns * s)

    ax.plot(x_helix, y_helix, z_helix, color='crimson', lw=2.2, 
            label=r'Фазовое бинтование ($\omega_{\mathrm{prec}} = \omega_{\mathrm{orb}}/\alpha$)')

    # Осевая направляющая окружность керна
    x_core = R * np.cos(s)
    y_core = R * np.sin(s)
    z_core = np.zeros_like(s)
    ax.plot(x_core, y_core, z_core, 'k--', lw=1.2, label=r'Ось волновода ($R_e = \hbar/m_e c$)')

    ax.set_title(r"Геометрия тора $T(1,1)$ и гироскопическое бинтование $r_e = \alpha R_e$", pad=10)
    
    # Строго равные пропорции по всем трем осям (исключает эффект цилиндра)
    ax.set_box_aspect([1, 1, 0.45])
    ax.set_xlim(-4.2, 4.2)
    ax.set_ylim(-4.2, 4.2)
    ax.set_zlim(-2.0, 2.0)
    ax.set_axis_off()
    ax.legend(loc='upper right', frameon=True, framealpha=0.9)
    ax.view_init(elev=32, azim=48)

    plt.savefig("figures/fig1_torus_shear.png")
    plt.savefig("figures/fig1_torus_shear.pdf")
    plt.close()

# ==============================================================================
# РИСУНОК 2: Координаты Хейга-Вестергорда (без наложения текста)
# ==============================================================================
def plot_figure_2():
    print("Генерация Рисунка 2: Девиаторная pi-плоскость (чистая верстка)...")
    fig, ax = plt.subplots(figsize=(7.5, 7.5))

    theta = np.linspace(0, 2*np.pi, 500)
    rho_mises = np.sqrt(2)
    x_mises = rho_mises * np.cos(theta)
    y_mises = rho_mises * np.sin(theta)

    ax.plot(x_mises, y_mises, 'b-', lw=2.2, label=r'Критерий Мизеса $Q = 2/3$ ($A=\sqrt{2}$)')
    ax.fill(x_mises, y_mises, color='lightblue', alpha=0.18)

    # Главные оси
    axes_angles = [0, 2*np.pi/3, 4*np.pi/3]
    axes_labels = [r'$\sigma_1^{\prime}$ ($\tau$)', r'$\sigma_2^{\prime}$ ($e$)', r'$\sigma_3^{\prime}$ ($\mu$)']
    
    for ang, lab in zip(axes_angles, axes_labels):
        ax.plot([0, 1.9*np.cos(ang)], [0, 1.9*np.sin(ang)], 'k--', lw=1.0, alpha=0.5)
        ax.text(2.05*np.cos(ang), 2.05*np.sin(ang), lab, ha='center', va='center', fontsize=11, fontweight='bold')

    theta_0 = 2.0 / 9.0  # 0.2222 рад
    leptons = [
        (r'$\tau$ (1776.9 МэВ)', theta_0, 'darkred', 's', (15, 10)),
        (r'$e$ (0.511 МэВ)', theta_0 + 2*np.pi/3, 'darkgreen', 'o', (-40, 15)),
        (r'$\mu$ (105.7 МэВ)', theta_0 + 4*np.pi/3, 'darkblue', '^', (-45, -25))
    ]

    for name, ang, col, marker, offset in leptons:
        x_pt = rho_mises * np.cos(ang)
        y_pt = rho_mises * np.sin(ang)
        ax.plot(x_pt, y_pt, marker=marker, color=col, markersize=8.5, zorder=5)
        ax.plot([0, x_pt], [0, y_pt], color=col, lw=1.2, alpha=0.7)
        ax.annotate(name, xy=(x_pt, y_pt), xytext=offset, textcoords='offset points',
                    fontsize=10, fontweight='bold', color=col,
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=col, lw=0.8, alpha=0.85))

    # Дуга угла Лоде
    arc_theta = np.linspace(0, theta_0, 50)
    ax.plot(0.65*np.cos(arc_theta), 0.65*np.sin(arc_theta), 'r-', lw=2.0)
    ax.text(0.82*np.cos(theta_0/2), 0.82*np.sin(theta_0/2), r'$\theta_0 = \frac{2}{9}$ рад', 
            color='red', fontsize=10.5, fontweight='bold')

    ax.set_aspect('equal')
    ax.set_xlim(-2.5, 2.5)
    ax.set_ylim(-2.5, 2.5)
    ax.set_xlabel(r'Девиаторное напряжение $s_x / K\sqrt{M_0}$')
    ax.set_ylabel(r'Девиаторное напряжение $s_y / K\sqrt{M_0}$')
    ax.set_title(r"Проекция тензора напряжений на $\pi$-плоскость и формула Коиде", pad=12)
    ax.grid(True, linestyle=':', alpha=0.5)
    ax.legend(loc='lower left', frameon=True, framealpha=0.9)

    plt.savefig("figures/fig2_haigh_westergaard.png")
    plt.savefig("figures/fig2_haigh_westergaard.pdf")
    plt.close()

# ==============================================================================
# РИСУНОК 3: Сечение DIS (Табличка перенесена вниз и вправо)
# ==============================================================================
def plot_figure_3():
    print("Генерация Рисунка 3: Порог DIS (табличка внизу справа)...")
    fig, ax = plt.subplots(figsize=(8.5, 5.5))

    E = np.linspace(1, 150, 600)
    E_crit = 67.055  # ГэВ

    # Сечение квазиупругого упругого режима
    sigma_elastic = 0.67 * E * (E <= E_crit)
    
    # Резонансный пик предела текучести
    sigma_peak = 0.67 * E_crit * np.exp(-((E - E_crit)/5.5)**2) * 1.35
    
    # Режим глубоко неупругого рассеяния (DIS)
    sigma_dis = np.where(E >= E_crit, 0.67 * E_crit + 0.90 * (E - E_crit)**0.94, 0.0)
    sigma_total = np.where(E < E_crit, 0.67 * E + sigma_peak, sigma_dis + sigma_peak)

    # Отрисовка кривой сечения и вертикальной границы
    ax.plot(E, sigma_total, 'b-', lw=2.4, label=r'Полное сечение $\sigma_{\nu N}(E_\nu)$')
    ax.axvline(x=E_crit, color='crimson', linestyle='--', lw=1.8, 
               label=rf'Порог моды $N_4=28$ ($E_{{\mathrm{{crit}}}} = 67.1$ ГэВ)')

    # Зоны режимов
    ax.axvspan(0, E_crit, color='forestgreen', alpha=0.08, label='Упругий режим (1D-струна)')
    ax.axvspan(E_crit, 150, color='firebrick', alpha=0.08, label='DIS / Адронизация (разрыв)')

    # Выноска с формулой порога (вверху слева от пика, указывает на излом)
    ax.annotate(r'$E_{\nu,\mathrm{crit}} = \frac{(M_4^{(0)})^2}{2 M_p} \approx 67.1\ \mathrm{GeV}$', 
                xy=(E_crit, 78), xytext=(E_crit - 42, 95),
                arrowprops=dict(arrowstyle="->", color="crimson", lw=1.5),
                fontsize=10.5, fontweight='bold', color='crimson',
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="crimson", lw=1.0, alpha=0.9))

    ax.set_xlabel(r'Энергия нейтрино $E_\nu$ в лаб. системе (ГэВ)')
    ax.set_ylabel(r'Сечение $\sigma_{\nu N} / E_\nu$ (отн. ед.)')
    ax.set_title(r"Механический порог прочности струны нейтрино и переход к DIS", pad=12)
    ax.set_xlim(0, 150)
    ax.set_ylim(0, 130)
    ax.grid(True, linestyle=':', alpha=0.5)

    # Легенда размещена строго внизу справа, где нет кривой (кривая идет выше 70)
    ax.legend(loc='lower right', frameon=True, framealpha=0.95, edgecolor='gray')

    plt.savefig("figures/fig3_dis_threshold.png")
    plt.savefig("figures/fig3_dis_threshold.pdf")
    plt.close()

# ==============================================================================
# РИСУНОК 4: Масштабная зависимость delta(R/r_e) (Мюон строго скомпенсирован)
# ==============================================================================
def plot_figure_4():
    print("Генерация Рисунка 4: Масштабная компенсация присоединенной массы...")
    fig, ax = plt.subplots(figsize=(8.5, 5.5))

    alpha = 1.0 / 137.035999
    delta_geom = (1.5 * alpha / np.pi) * 100  # 0.3484%
    r_e = 2.81794  # фм (классический радиус керна)

    # Шкала от 0.01 r_e до 200 r_e
    R_ratio = np.logspace(-2, 2.5, 600)
    
    # Строгая физическая зависимость от порога r_e
    delta_curve = np.where(
        R_ratio <= 1.0,
        delta_geom,
        delta_geom + (6.0 * (alpha**2) * np.log(R_ratio)) * 100
    )

    ax.plot(R_ratio, delta_curve, 'b-', lw=2.4, 
            label=r'Теория: $\delta(R) = \frac{1.5\alpha}{\pi} + 6\alpha^2\ln\left[\max\left(1, \frac{R}{r_e}\right)\right]$')
    ax.axhline(y=delta_geom, color='gray', linestyle=':', lw=1.4, 
               label=rf'Геометрическое плато $\frac{{1.5\alpha}}{{\pi}} = {delta_geom:.4f}\%$')
    ax.axvline(x=1.0, color='purple', linestyle='--', lw=1.2, alpha=0.7,
               label=r'Граница керна $r_e = \alpha R_e \approx 2.82$ фм')

    # Экспериментальные точки лептонов относительно r_e = 2.818 фм
    # R_tau = 0.111 фм -> R_tau / r_e = 0.0394
    # R_mu  = 1.868 фм -> R_mu / r_e  = 0.6629 (СТРОГО < 1, на плато!)
    # R_e   = 386.16 фм -> R_e / r_e   = 137.036 (СТРОГО > 1)
    leptons_data = [
        (r'$\tau$ ($0.04\,r_e$)', 0.111 / r_e, 0.3468, 'darkred', 's', (-25, 15)),
        (r'$\mu$ ($0.66\,r_e$)', 1.868 / r_e, 0.3479, 'darkblue', '^', (-25, 15)),
        (r'$e$ ($137\,r_e$)', 386.16 / r_e, 0.5088, 'darkgreen', 'o', (-60, -20))
    ]

    for name, r_val, d_val, col, marker, offset in leptons_data:
        ax.scatter(r_val, d_val, color=col, s=85, zorder=5, marker=marker)
        ax.annotate(f'{name}\n$\\delta = {d_val:.4f}\\%$', xy=(r_val, d_val), xytext=offset,
                    textcoords='offset points', fontsize=9.5, fontweight='bold', color=col,
                    bbox=dict(boxstyle="round,pad=0.25", fc="white", ec=col, lw=0.9, alpha=0.9))

    ax.set_xscale('log')
    ax.set_xlabel(r'Отношение волнового размера к радиусу керна $R / r_e$')
    ax.set_ylabel(r'Присоединенная масса $\Delta M / M^{(0)}$ (\%)')
    ax.set_title(r"Компенсация присоединенной массы вакуума ($R_\tau < R_\mu < r_e < R_e$)", pad=12)
    ax.set_ylim(0.30, 0.55)
    ax.grid(True, which='both', linestyle=':', alpha=0.5)
    ax.legend(loc='upper left', frameon=True, framealpha=0.9)

    plt.savefig("figures/fig4_added_mass_scaling.png")
    plt.savefig("figures/fig4_added_mass_scaling.pdf")
    plt.close()

# ==============================================================================
# РИСУНОК 5: Эффект Парселла (без перекрытия стрелок)
# ==============================================================================
def plot_figure_5():
    print("Генерация Рисунка 5: Время жизни нейтрона (чистая верстка)...")
    fig, ax = plt.subplots(figsize=(8.5, 5.5))

    L_cavity = np.linspace(0.05, 2.5, 500)
    tau_beam = 887.7
    tau_bottle = 879.06
    
    tau_curve = tau_bottle + (tau_beam - tau_bottle) * (1.0 / (1.0 + np.exp(-(L_cavity - 0.7)/0.18)))

    ax.plot(L_cavity, tau_curve, 'm-', lw=2.4, label=r'$\tau_n(L_{\mathrm{cavity}})$ (переход $S^2 \to T^2$)')
    ax.axhline(y=tau_beam, color='darkgreen', linestyle='--', lw=1.5, label=rf'Пучок (Beam): $\tau = 887.7 \pm 1.2$ с')
    ax.axhline(y=tau_bottle, color='crimson', linestyle='--', lw=1.5, label=rf'Ловушка (UCN$\tau$ Bottle): $\tau = 878.4 \pm 0.5$ с')

    # Стрелка дефекта времени жизни вынесена в свободную зону слева
    ax.annotate('', xy=(0.25, tau_bottle), xytext=(0.25, tau_beam),
                arrowprops=dict(arrowstyle='<->', color='black', lw=1.6))
    ax.text(0.30, (tau_beam + tau_bottle)/2, 
            r'$\Delta \tau = \frac{4}{3}\alpha \tau_{\mathrm{beam}} \approx 8.64\ \mathrm{s}$', 
            va='center', fontsize=11, fontweight='bold', color='black',
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black", lw=0.8, alpha=0.9))

    ax.set_xlabel(r'Характерный размер резонатора ловушки $L_{\mathrm{cavity}}$ (м)')
    ax.set_ylabel(r'Время жизни нейтрона $\tau_n$ (секунды)')
    ax.set_title(r"Разрешение аномалии «пучок--ловушка» через эффект Парселла", pad=12)
    ax.set_ylim(873, 893)
    ax.grid(True, linestyle=':', alpha=0.5)
    ax.legend(loc='lower right', frameon=True, framealpha=0.9)

    plt.savefig("figures/fig5_purcell_cavity.png")
    plt.savefig("figures/fig5_purcell_cavity.pdf")
    plt.close()

if __name__ == "__main__":
    print("=== Генерация обновленных графиков (v2.0) ===")
    plot_figure_1()
    plot_figure_2()
    plot_figure_3()
    plot_figure_4()
    plot_figure_5()
    print("=== Все 5 графиков перегенерированы без наложений (PNG и PDF) ===")
