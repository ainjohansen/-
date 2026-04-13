import torch
import torch.multiprocessing as mp
import numpy as np
import math
import time
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import warnings
import os

warnings.filterwarnings("ignore")

# =====================================================================
#                         НАСТРОЙКИ
# =====================================================================
GRID_SIZE = 80          # 80^3 обычно влезает в 11 ГБ
L = 10.0
ALPHA = 0.0072973525693
J_FIXED = 0.5
SKYRME_COEF = 10.9
N_VALUES = [1, 6, 15]
STEPS_ADAM = 800        # уменьшим для скорости
STEPS_LBFGS = 400
LR_ADAM = 0.005
SAVE_FIELDS = True
VISUALIZE = True

# =====================================================================
#                    ФУНКЦИИ
# =====================================================================

def init_hopfion(device, n=1, scale=1.2):
    x = torch.linspace(-L, L, GRID_SIZE, dtype=torch.float32, device=device)
    y = torch.linspace(-L, L, GRID_SIZE, dtype=torch.float32, device=device)
    z = torch.linspace(-L, L, GRID_SIZE, dtype=torch.float32, device=device)
    Z, Y, X = torch.meshgrid(z, y, x, indexing='ij')
    X, Y, Z = X/scale, Y/scale, Z/scale

    u_re, u_im = X, Y
    v_re, v_im = Z, (X**2 + Y**2 + Z**2 - 1) / 2

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


def optimize_on_device(device_id, n):
    device = torch.device(f"cuda:{device_id}")
    print(f"[n={n}] Старт на {device}")

    dx = 2 * L / GRID_SIZE
    dV = dx**3

    kx = torch.fft.fftfreq(GRID_SIZE, d=dx, device=device) * 2 * math.pi
    Kz, Ky, Kx = torch.meshgrid(kx, kx, kx, indexing='ij')
    K2 = Kx**2 + Ky**2 + Kz**2
    K2[0,0,0] = 1.0

    w = init_hopfion(device, n).clone().requires_grad_(True)

    # Adam
    optimizer = torch.optim.Adam([w], lr=LR_ADAM)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=300, gamma=0.5)
    for step in range(STEPS_ADAM):
        optimizer.zero_grad()
        E, _ = compute_energy(w, ALPHA, J_FIXED, SKYRME_COEF, K2, dx, dV)
        E.backward()
        with torch.no_grad():
            w.grad[:, 0, :, :] = 0; w.grad[:, -1, :, :] = 0
            w.grad[:, :, 0, :] = 0; w.grad[:, :, -1, :] = 0
            w.grad[:, :, :, 0] = 0; w.grad[:, :, :, -1] = 0
        torch.nn.utils.clip_grad_norm_([w], max_norm=1.0)
        optimizer.step()
        scheduler.step()
        if step % 200 == 0:
            print(f"[n={n}] Adam {step}: E={E.item():.2f}")

    # LBFGS
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

    with torch.no_grad():
        E_final, I_final = compute_energy(w, ALPHA, J_FIXED, SKYRME_COEF, K2, dx, dV)
        n_final = w / torch.norm(w, dim=0, keepdim=True).clamp(min=1e-12)

    print(f"[n={n}] Завершено: E={E_final.item():.6f}, I={I_final.item():.3f}")

    result = {'n': n, 'E': E_final.item(), 'I': I_final.item()}
    if SAVE_FIELDS:
        result['field'] = n_final.cpu().numpy()
    return result


def visualize_field(field, n, save_prefix="hopfion"):
    """Создаёт срезы и изоповерхность для поля n3."""
    n3 = field[2]  # третья компонента
    mid = GRID_SIZE // 2

    # Срезы по осям
    fig, axes = plt.subplots(1, 3, figsize=(12,4))
    axes[0].imshow(n3[mid, :, :], origin='lower', extent=[-L, L, -L, L], cmap='RdBu')
    axes[0].set_title(f'n={n}: сечение Z=0')
    axes[0].set_xlabel('X'); axes[0].set_ylabel('Y')
    axes[1].imshow(n3[:, mid, :], origin='lower', extent=[-L, L, -L, L], cmap='RdBu')
    axes[1].set_title(f'сечение Y=0')
    axes[1].set_xlabel('X'); axes[1].set_ylabel('Z')
    axes[2].imshow(n3[:, :, mid], origin='lower', extent=[-L, L, -L, L], cmap='RdBu')
    axes[2].set_title(f'сечение X=0')
    axes[2].set_xlabel('Y'); axes[2].set_ylabel('Z')
    plt.tight_layout()
    plt.savefig(f"{save_prefix}_slices_n{n}.png", dpi=150)
    plt.close()

    # 3D изоповерхность через plotly (сохраняем как HTML)
    try:
        import plotly.graph_objects as go
        # Для экономии памяти берём каждую 2-ю точку
        X, Y, Z = np.mgrid[-L:L:GRID_SIZE*1j, -L:L:GRID_SIZE*1j, -L:L:GRID_SIZE*1j][:,::2,::2,::2]
        values = n3[::2,::2,::2]
        fig = go.Figure(data=go.Isosurface(
            x=X.flatten(), y=Y.flatten(), z=Z.flatten(),
            value=values.flatten(),
            isomin=-0.5, isomax=0.5,
            surface_count=3,
            colorscale='RdBu',
            caps=dict(x_show=False, y_show=False, z_show=False)
        ))
        fig.write_html(f"{save_prefix}_isosurface_n{n}.html")
    except ImportError:
        print("Plotly not installed, skipping 3D isosurface.")


if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    num_gpus = torch.cuda.device_count()
    if num_gpus == 0:
        raise RuntimeError("Требуется CUDA")

    print(f"GPU: {num_gpus}, сетка {GRID_SIZE}^3")
    args = [(i % num_gpus, n) for i, n in enumerate(N_VALUES)]

    with mp.Pool(processes=num_gpus) as pool:
        results = pool.starmap(optimize_on_device, args)

    results.sort(key=lambda r: r['n'])
    df = pd.DataFrame(results)
    fields = None
    if SAVE_FIELDS:
        fields = [r.pop('field') for r in results]

    df['I_norm'] = df['I'] / df[df['n']==1]['I'].values[0]
    df['E_norm'] = df['E'] / df[df['n']==1]['E'].values[0]

    print("\n" + "="*70)
    print(df[['n', 'E', 'I', 'E_norm', 'I_norm']].to_string(index=False))
    print("="*70)
    df.to_csv('leptons_3d_results.csv', index=False)

    if VISUALIZE and fields:
        for i, n in enumerate(df['n']):
            visualize_field(fields[i], n)

    print("Готово.")
