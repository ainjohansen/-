import torch
import torch.multiprocessing as mp
import numpy as np
import math
import time
import pandas as pd
import warnings

warnings.filterwarnings("ignore")

# ==========================================
# ПАРАМЕТРЫ СЕТКИ И ФИЗИКИ
# ==========================================
N = 64               # размер сетки (можно увеличить до 80, если хватит памяти)
L = 8.0
dx = 2 * L / N
dV = dx**3
alpha_target = 0.0072973525693
J_fixed = 0.5
skyrme_coef = 10.9

# Числа навивки для лептонов
n_values = [1, 6, 15]

# Параметры оптимизации
STEPS_ADAM = 800
STEPS_LBFGS = 500
LR = 0.005

# ==========================================
# ИНИЦИАЛИЗАЦИЯ ХОПФИОНА С АЗИМУТАЛЬНОЙ МОДУЛЯЦИЕЙ n
# ==========================================
def init_hopfion(device, n=1):
    x = torch.linspace(-L, L, N, dtype=torch.float64, device=device)
    y = torch.linspace(-L, L, N, dtype=torch.float64, device=device)
    z = torch.linspace(-L, L, N, dtype=torch.float64, device=device)
    Z, Y, X = torch.meshgrid(z, y, x, indexing='ij')
    scale = 1.2
    X, Y, Z = X/scale, Y/scale, Z/scale
    u_re, u_im = X, Y
    v_re, v_im = Z, (X**2 + Y**2 + Z**2 - 1) / 2

    # Азимутальная модуляция фазы: exp(i n φ)
    phi = torch.atan2(Y, X)
    exp_n_real = torch.cos(n * phi)
    exp_n_imag = torch.sin(n * phi)

    u_re_mod = u_re * exp_n_real - u_im * exp_n_imag
    u_im_mod = u_re * exp_n_imag + u_im * exp_n_real

    denom = u_re_mod**2 + u_im_mod**2 + v_re**2 + v_im**2 + 1e-12
    n1 = 2 * (u_re_mod * v_re + u_im_mod * v_im) / denom
    n2 = 2 * (u_im_mod * v_re - u_re_mod * v_im) / denom
    n3 = (u_re_mod**2 + u_im_mod**2 - v_re**2 - v_im**2) / denom
    return torch.stack([n1, n2, n3], dim=0)

# ==========================================
# ВЫЧИСЛЕНИЕ ЭНЕРГИИ И МОМЕНТА ИНЕРЦИИ
# ==========================================
def compute_energy(w, alpha, J_fixed, skyrme_coef, K2):
    n = w / torch.norm(w, dim=0, keepdim=True).clamp(min=1e-12)
    grad_z, grad_y, grad_x = torch.gradient(n, spacing=dx, dim=(1, 2, 3), edge_order=1)

    E_dir = 0.5 * torch.sum(grad_x**2 + grad_y**2 + grad_z**2) * dV

    cross_yz = torch.cross(grad_y, grad_z, dim=0)
    cross_zx = torch.cross(grad_z, grad_x, dim=0)
    cross_xy = torch.cross(grad_x, grad_y, dim=0)
    E_sk = skyrme_coef * torch.sum(cross_xy**2 + cross_yz**2 + cross_zx**2) * dV

    E_pot = 3 * alpha * torch.sum(1 + n[2]) * dV

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

    I_rot = torch.sum(n[0]**2 + n[1]**2) * dV
    E_spin = 0.5 * (J_fixed**2) / (I_rot + 1e-12)

    E_tot = E_dir + E_sk + E_pot + E_em + E_spin
    return E_tot, I_rot

# ==========================================
# ОПТИМИЗАЦИЯ ДЛЯ ОДНОГО n (на одном GPU)
# ==========================================
def optimize_for_n(n, device_id):
    device = torch.device(f"cuda:{device_id}")
    print(f"[n={n}] Старт на {device}")

    # K-пространство для Пуассона
    kx = torch.fft.fftfreq(N, d=dx, device=device) * 2 * math.pi
    Kz, Ky, Kx = torch.meshgrid(kx, kx, kx, indexing='ij')
    K2 = Kx**2 + Ky**2 + Kz**2
    K2[0,0,0] = 1.0

    w = init_hopfion(device, n).clone().requires_grad_(True)

    # Этап 1: Adam
    optimizer = torch.optim.Adam([w], lr=LR)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=400, gamma=0.5)
    for step in range(STEPS_ADAM):
        optimizer.zero_grad()
        E, _ = compute_energy(w, alpha_target, J_fixed, skyrme_coef, K2)
        E.backward()
        with torch.no_grad():
            w.grad[:, 0, :, :] = 0; w.grad[:, -1, :, :] = 0
            w.grad[:, :, 0, :] = 0; w.grad[:, :, -1, :] = 0
            w.grad[:, :, :, 0] = 0; w.grad[:, :, :, -1] = 0
        torch.nn.utils.clip_grad_norm_([w], max_norm=1.0)
        optimizer.step()
        scheduler.step()
        if step % 200 == 0:
            print(f"[n={n}] Adam step {step}: E={E.item():.4f}")

    # Этап 2: LBFGS
    optimizer = torch.optim.LBFGS([w], max_iter=STEPS_LBFGS, tolerance_grad=1e-9,
                                  line_search_fn='strong_wolfe')
    def closure():
        optimizer.zero_grad()
        E, _ = compute_energy(w, alpha_target, J_fixed, skyrme_coef, K2)
        E.backward()
        with torch.no_grad():
            w.grad[:, 0, :, :] = 0; w.grad[:, -1, :, :] = 0
            w.grad[:, :, 0, :] = 0; w.grad[:, :, -1, :] = 0
            w.grad[:, :, :, 0] = 0; w.grad[:, :, :, -1] = 0
        return E
    optimizer.step(closure)

    with torch.no_grad():
        E_final, I_final = compute_energy(w, alpha_target, J_fixed, skyrme_coef, K2)
        n_final = w / torch.norm(w, dim=0, keepdim=True).clamp(min=1e-12)

    print(f"[n={n}] Завершено: E={E_final.item():.6f}, I={I_final.item():.3f}")
    return {'n': n, 'E': E_final.item(), 'I': I_final.item()}

# ==========================================
# ПАРАЛЛЕЛЬНЫЙ ЗАПУСК НА ВСЕХ GPU
# ==========================================
if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    num_gpus = torch.cuda.device_count()
    print(f"Доступно GPU: {num_gpus}")

    # Распределение n по GPU
    args = [(n_values[i], i % num_gpus) for i in range(len(n_values))]

    with mp.Pool(processes=num_gpus) as pool:
        results = pool.starmap(optimize_for_n, args)

    # Создаём DataFrame
    df = pd.DataFrame(results)
    df = df.sort_values('n')
    E1 = df[df['n']==1]['E'].values[0]
    df['E_ratio'] = df['E'] / E1
    df['n3'] = df['n']**3
    df['ratio_pred'] = df['n3'] / 1
    df['error'] = abs(df['E_ratio'] - df['ratio_pred']) / df['ratio_pred'] * 100

    print("\n" + "="*70)
    print("РЕЗУЛЬТАТЫ 3D РЕЛАКСАЦИИ ДЛЯ ЛЕПТОНОВ")
    print("="*70)
    print(df[['n', 'E', 'E_ratio', 'n3', 'error']].to_string(index=False))
    print("="*70)
    df.to_csv('leptons_3d_results.csv', index=False)
    print("Результаты сохранены в 'leptons_3d_results.csv'")
