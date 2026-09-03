"""
======================================================================
  3D ТОПОЛОГИЧЕСКАЯ МОДЕЛЬ ЛЕПТОНОВ (ЭЛЕКТРОН, МЮОН, ТАУ)
  Адаптивный хопфион с азимутальной модуляцией фазы n = 1, 6, 15
======================================================================
  Входные параметры (точность/скорость) находятся в блоке "НАСТРОЙКИ".
  Для запуска: python leptons_3d_optimized.py
  Результаты сохраняются в CSV и графики в PNG.
"""

import torch
import torch.multiprocessing as mp
import numpy as np
import math
import time
import pandas as pd
import matplotlib.pyplot as plt
import warnings
from functools import partial

warnings.filterwarnings("ignore")

# =====================================================================
#                         НАСТРОЙКИ (МЕНЯТЬ ЗДЕСЬ)
# =====================================================================
# Размер сетки (N^3). Для точных результатов рекомендуется 80 или 100.
# При недостатке памяти уменьшить до 64.
GRID_SIZE = 200

# Полуразмер области. Влияет на граничные условия.
# Больше L → меньше влияние границ, но нужно больше памяти/времени.
L = 10.0

# Физические константы
ALPHA = 0.0072973525693      # постоянная тонкой структуры (1/137)
J_FIXED = 0.5                # фиксированный спин (безразмерный)
SKYRME_COEF = 10.9           # коэффициент при члене Скирма

# Числа навивки фазы для лептонов
N_VALUES = [1, 6, 15]

# Параметры оптимизации
STEPS_ADAM = 1200            # шагов Adam (быстрый спуск)
STEPS_LBFGS = 800            # шагов LBFGS (точная доводка)
LR_ADAM = 0.005              # начальный learning rate для Adam
LR_GAMMA = 0.5               # коэффициент затухания LR (StepLR каждые 400 шагов)

# Сохранение промежуточных полей (True/False)
SAVE_FIELDS = True

# =====================================================================
#                    ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# =====================================================================

def init_hopfion(device, n=1, scale=1.2):
    """
    Создаёт начальное поле хопфиона Q=1 с азимутальной модуляцией exp(i n φ).
    """
    x = torch.linspace(-L, L, GRID_SIZE, dtype=torch.float64, device=device)
    y = torch.linspace(-L, L, GRID_SIZE, dtype=torch.float64, device=device)
    z = torch.linspace(-L, L, GRID_SIZE, dtype=torch.float64, device=device)
    Z, Y, X = torch.meshgrid(z, y, x, indexing='ij')
    X, Y, Z = X/scale, Y/scale, Z/scale

    u_re, u_im = X, Y
    v_re, v_im = Z, (X**2 + Y**2 + Z**2 - 1) / 2

    # Азимутальная модуляция
    phi = torch.atan2(Y, X)
    exp_real = torch.cos(n * phi)
    exp_imag = torch.sin(n * phi)

    u_re_mod = u_re * exp_real - u_im * exp_imag
    u_im_mod = u_re * exp_imag + u_im * exp_real

    denom = u_re_mod**2 + u_im_mod**2 + v_re**2 + v_im**2 + 1e-12
    n1 = 2 * (u_re_mod * v_re + u_im_mod * v_im) / denom
    n2 = 2 * (u_im_mod * v_re - u_re_mod * v_im) / denom
    n3 = (u_re_mod**2 + u_im_mod**2 - v_re**2 - v_im**2) / denom
    return torch.stack([n1, n2, n3], dim=0)


def compute_energy(w, alpha, J_fixed, skyrme_coef, K2, dx, dV):
    """
    Вычисляет полную энергию и момент инерции для текущего поля w.
    """
    n = w / torch.norm(w, dim=0, keepdim=True).clamp(min=1e-12)
    grad_z, grad_y, grad_x = torch.gradient(n, spacing=dx, dim=(1, 2, 3), edge_order=1)

    # Кинетическая энергия (I2)
    E_dir = 0.5 * torch.sum(grad_x**2 + grad_y**2 + grad_z**2) * dV

    # Энергия Скирма (I4)
    cross_yz = torch.cross(grad_y, grad_z, dim=0)
    cross_zx = torch.cross(grad_z, grad_x, dim=0)
    cross_xy = torch.cross(grad_x, grad_y, dim=0)
    E_sk = skyrme_coef * torch.sum(cross_xy**2 + cross_yz**2 + cross_zx**2) * dV

    # Потенциальная энергия (массовый член)
    E_pot = 3 * alpha * torch.sum(1 + n[2]) * dV

    # Электромагнитная энергия (отрицательная)
    B_x = torch.sum(n * cross_yz, dim=0)
    B_y = torch.sum(n * cross_zx, dim=0)
    B_z = torch.sum(n * cross_xy, dim=0)
    B_mag = torch.sqrt(B_x**2 + B_y**2 + B_z**2 + 1e-12)
    rho = math.sqrt(4 * math.pi * alpha) * (B_mag / (8 * math.pi))
    rho_fft = torch.fft.fftn(rho)
    Phi_fft = rho_fft / K2
    Phi_fft[0,0,0] = 0.0
    Phi = torch.fft.ifftn(Phi_fft).real
    E_em = -0.5 * torch.sum(rho * Phi) * dV

    # Центробежная энергия (фиксированный спин)
    I_rot = torch.sum(n[0]**2 + n[1]**2) * dV
    E_spin = 0.5 * (J_fixed**2) / (I_rot + 1e-12)

    E_tot = E_dir + E_sk + E_pot + E_em + E_spin
    return E_tot, I_rot


def optimize_on_device(device_id, n):
    """
    Основная функция оптимизации для заданного n на конкретном GPU.
    Возвращает словарь с результатами.
    """
    device = torch.device(f"cuda:{device_id}")
    print(f"[n={n}] Старт на {device}")

    dx = 2 * L / GRID_SIZE
    dV = dx**3

    # Подготовка K-пространства для уравнения Пуассона
    kx = torch.fft.fftfreq(GRID_SIZE, d=dx, device=device) * 2 * math.pi
    Kz, Ky, Kx = torch.meshgrid(kx, kx, kx, indexing='ij')
    K2 = Kx**2 + Ky**2 + Kz**2
    K2[0,0,0] = 1.0

    # Инициализация поля
    w = init_hopfion(device, n).clone().requires_grad_(True)

    # Этап 1: Adam
    optimizer = torch.optim.Adam([w], lr=LR_ADAM)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=400, gamma=LR_GAMMA)
    for step in range(STEPS_ADAM):
        optimizer.zero_grad()
        E, _ = compute_energy(w, ALPHA, J_FIXED, SKYRME_COEF, K2, dx, dV)
        E.backward()
        with torch.no_grad():
            # Заморозка границ
            w.grad[:, 0, :, :] = 0; w.grad[:, -1, :, :] = 0
            w.grad[:, :, 0, :] = 0; w.grad[:, :, -1, :] = 0
            w.grad[:, :, :, 0] = 0; w.grad[:, :, :, -1] = 0
        torch.nn.utils.clip_grad_norm_([w], max_norm=1.0)
        optimizer.step()
        scheduler.step()
        if step % 200 == 0:
            print(f"[n={n}] Adam {step}: E={E.item():.2f}")

    # Этап 2: LBFGS
    optimizer = torch.optim.LBFGS([w], max_iter=STEPS_LBFGS, tolerance_grad=1e-9,
                                  line_search_fn='strong_wolfe')
    def closure():
        optimizer.zero_grad()
        E, _ = compute_energy(w, ALPHA, J_FIXED, SKYRME_COEF, K2, dx, dV)
        E.backward()
        with torch.no_grad():
            w.grad[:, 0, :, :] = 0; w.grad[:, -1, :, :] = 0
            w.grad[:, :, 0, :] = 0; w.grad[:, :, -1, :] = 0
            w.grad[:, :, :, 0] = 0; w.grad[:, :, :, -1] = 0
        return E
    optimizer.step(closure)

    # Финальный расчёт
    with torch.no_grad():
        E_final, I_final = compute_energy(w, ALPHA, J_FIXED, SKYRME_COEF, K2, dx, dV)
        n_final = w / torch.norm(w, dim=0, keepdim=True).clamp(min=1e-12)

    print(f"[n={n}] Завершено: E={E_final.item():.6f}, I={I_final.item():.3f}")

    result = {'n': n, 'E': E_final.item(), 'I': I_final.item()}

    if SAVE_FIELDS:
        # Сохраняем поле на CPU
        result['field'] = n_final.cpu().numpy()

    return result


# =====================================================================
#                         ГЛАВНЫЙ БЛОК
# =====================================================================
if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    num_gpus = torch.cuda.device_count()
    if num_gpus == 0:
        raise RuntimeError("Не найдено GPU. Для расчётов требуется CUDA.")

    print(f"Доступно GPU: {num_gpus}")
    print(f"Сетка: {GRID_SIZE}^3, L={L}")
    print(f"Шаги: Adam={STEPS_ADAM}, LBFGS={STEPS_LBFGS}")

    # Распределение n по GPU
    args = [(i % num_gpus, n) for i, n in enumerate(N_VALUES)]

    with mp.Pool(processes=num_gpus) as pool:
        results = pool.starmap(optimize_on_device, args)

    # Сортировка по n
    results.sort(key=lambda r: r['n'])

    # Создание DataFrame
    df = pd.DataFrame(results)
    if SAVE_FIELDS:
        fields = df.pop('field')
        # Сохраняем поля отдельно
        for i, n in enumerate(df['n']):
            np.save(f"hopfion_field_n{n}.npy", fields[i])

    df['I_normalized'] = df['I'] / df[df['n']==1]['I'].values[0]
    df['E_normalized'] = df['E'] / df[df['n']==1]['E'].values[0]

    print("\n" + "="*70)
    print("РЕЗУЛЬТАТЫ 3D РЕЛАКСАЦИИ")
    print("="*70)
    print(df[['n', 'E', 'I', 'E_normalized', 'I_normalized']].to_string(index=False))
    print("="*70)

    df.to_csv('leptons_3d_results.csv', index=False)
    print("Таблица сохранена в 'leptons_3d_results.csv'")

    # Построение графиков
    plt.style.use('dark_background')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12,5))

    ax1.plot(df['n'], df['E'], 'o-', color='cyan', markersize=8)
    ax1.set_xlabel('Число навивки n')
    ax1.set_ylabel('Энергия E')
    ax1.set_title('Энергия лептонов')
    ax1.grid(alpha=0.3)

    ax2.plot(df['n'], df['I'], 'o-', color='magenta', markersize=8)
    ax2.set_xlabel('Число навивки n')
    ax2.set_ylabel('Момент инерции I')
    ax2.set_title('Размер (момент инерции)')
    ax2.grid(alpha=0.3)

    plt.suptitle(f'Топологическая модель лептонов (сетка {GRID_SIZE}^3)')
    plt.tight_layout()
    plt.savefig('leptons_3d_plots.png', dpi=150)
    plt.show()

    print("\nГотово. Графики сохранены в 'leptons_3d_plots.png'.")
