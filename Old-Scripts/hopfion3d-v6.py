import torch
import torch.multiprocessing as mp
import numpy as np
import matplotlib.pyplot as plt
import time
import warnings
from functools import partial

warnings.filterwarnings("ignore")

N = 80
L = 8.0
dx = 2 * L / N
dV = dx**3
STEPS = 1200         # больше шагов для точной сходимости
J_TARGET = 0.5
SKYRME_COEF = 10.9

def init_hopfion(device):
    x = torch.linspace(-L, L, N, dtype=torch.float64, device=device)
    y = torch.linspace(-L, L, N, dtype=torch.float64, device=device)
    z = torch.linspace(-L, L, N, dtype=torch.float64, device=device)
    Z, Y, X = torch.meshgrid(z, y, x, indexing='ij')
    scale = 1.2
    X, Y, Z = X/scale, Y/scale, Z/scale
    u_re, u_im = X, Y
    v_re, v_im = Z, (X**2 + Y**2 + Z**2 - 1) / 2
    denom = u_re**2 + u_im**2 + v_re**2 + v_im**2 + 1e-12
    n1 = 2 * (u_re * v_re + u_im * v_im) / denom
    n2 = 2 * (u_im * v_re - u_re * v_im) / denom
    n3 = (u_re**2 + u_im**2 - v_re**2 - v_im**2) / denom
    return torch.stack([n1, n2, n3], dim=0)

def relax_fixed_spin(alpha, J_fixed, device_id):
    device = torch.device(f"cuda:{device_id}")
    worker_name = f"GPU-{device_id}"

    kx = torch.fft.fftfreq(N, d=dx, device=device) * 2 * np.pi
    Kz, Ky, Kx = torch.meshgrid(kx, kx, kx, indexing='ij')
    K2 = Kx**2 + Ky**2 + Kz**2
    K2[0, 0, 0] = 1.0

    w = init_hopfion(device).clone().requires_grad_(True)
    optimizer = torch.optim.Adam([w], lr=0.005)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=400, gamma=0.5)

    best_E = float('inf')
    best_I = None

    for step in range(STEPS):
        optimizer.zero_grad()
        n = w / torch.norm(w, dim=0, keepdim=True).clamp(min=1e-12)
        grad_z, grad_y, grad_x = torch.gradient(n, spacing=dx, dim=(1, 2, 3), edge_order=1)

        E_dir = 0.5 * torch.sum(grad_x**2 + grad_y**2 + grad_z**2) * dV
        cross_yz = torch.cross(grad_y, grad_z, dim=0)
        cross_zx = torch.cross(grad_z, grad_x, dim=0)
        cross_xy = torch.cross(grad_x, grad_y, dim=0)
        E_sk = SKYRME_COEF * torch.sum(cross_xy**2 + cross_yz**2 + cross_zx**2) * dV
        E_pot = 3 * alpha * torch.sum(1 + n[2]) * dV

        B_x = torch.sum(n * cross_yz, dim=0)
        B_y = torch.sum(n * cross_zx, dim=0)
        B_z = torch.sum(n * cross_xy, dim=0)
        B_mag = torch.sqrt(B_x**2 + B_y**2 + B_z**2 + 1e-12)
        rho = np.sqrt(4 * np.pi * alpha) * (B_mag / (8 * np.pi))
        rho_fft = torch.fft.fftn(rho)
        Phi_fft = rho_fft / K2
        Phi_fft[0,0,0] = 0.0
        Phi = torch.fft.ifftn(Phi_fft).real
        E_em = -0.5 * torch.sum(rho * Phi) * dV

        I_rot = torch.sum(n[0]**2 + n[1]**2) * dV
        E_spin = 0.5 * (J_fixed**2) / (I_rot + 1e-12)

        E_tot = E_dir + E_sk + E_pot + E_em + E_spin
        E_tot.backward()

        with torch.no_grad():
            w.grad[:, 0, :, :] = 0; w.grad[:, -1, :, :] = 0
            w.grad[:, :, 0, :] = 0; w.grad[:, :, -1, :] = 0
            w.grad[:, :, :, 0] = 0; w.grad[:, :, :, -1] = 0

        torch.nn.utils.clip_grad_norm_([w], max_norm=1.0)
        optimizer.step()
        scheduler.step()

        with torch.no_grad():
            if E_tot.item() < best_E:
                best_E = E_tot.item()
                best_I = I_rot.item()

        if step % 200 == 0 and device_id == 0:
            print(f"   [{worker_name}] α={alpha:.5f} Step {step:4d} | E={E_tot.item():.2f} | I={I_rot.item():.1f}")

    J_actual = np.sqrt(2 * E_spin.item() * best_I) if best_I else 0
    print(f"[{worker_name}] α={alpha:.5f} | E_min={best_E:.2f} | I_rot={best_I:.1f} | J={J_actual:.3f}")
    return alpha, best_E, best_I, J_actual

def worker_wrapper(alpha_device_tuple):
    alpha, device_id = alpha_device_tuple
    return relax_fixed_spin(alpha, J_TARGET, device_id)

if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    num_gpus = torch.cuda.device_count()
    print(f"Доступно GPU: {num_gpus}")
    print(f"Фиксированный спин J = {J_TARGET}")
    print(f"Коэффициент Скирма = {SKYRME_COEF}")

    # Точная сетка вблизи предполагаемого минимума
    alphas = np.linspace(0.0070, 0.0085, 8)

    args = [(alphas[i], i % num_gpus) for i in range(len(alphas))]

    with mp.Pool(processes=num_gpus) as pool:
        results = pool.map(worker_wrapper, args)

    results.sort(key=lambda x: x[0])
    alphas_res = [r[0] for r in results]
    energies = [r[1] for r in results]

    print("\n" + "="*70)
    print("РЕЗУЛЬТАТЫ ТОЧНОГО СКАНИРОВАНИЯ")
    print("="*70)
    for a, e in zip(alphas_res, energies):
        print(f"α={a:.6f} E={e:.4f}")

    min_idx = np.argmin(energies)
    best_alpha = alphas_res[min_idx]
    print(f"\nМинимум энергии при α = {best_alpha:.6f}")
    print(f"Экспериментальное α = 0.007297")
    print(f"Отклонение: {abs(best_alpha - 0.007297)/0.007297*100:.2f}%")

    plt.figure(figsize=(8,5))
    plt.plot(alphas_res, energies, 'o-', label=f'E(α) J={J_TARGET}')
    plt.axvline(1/137.036, color='r', linestyle='--', label='1/137')
    plt.xlabel('α'); plt.ylabel('Энергия')
    plt.legend(); plt.grid(alpha=0.3)
    plt.title('Точное сканирование вблизи минимума')
    plt.savefig('fine_scan.png', dpi=150)
    plt.show()
