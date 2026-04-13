import torch
import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline

# Загрузка эталонного профиля
data = torch.load('electron_hopfion.pt')
n_star = data['n'].to(torch.float64)  # [3, N, N, N]
alpha0 = data['alpha']
J_fixed = data['J']
skyrme_coef = data['skyrme_coef']
L = data['L']
N = data['N']

dx = 2 * L / N
dV = dx**3

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
n_star = n_star.to(device)
n_star.requires_grad = False

# K-пространство для Пуассона
kx = torch.fft.fftfreq(N, d=dx, device=device) * 2 * math.pi
Kz, Ky, Kx = torch.meshgrid(kx, kx, kx, indexing='ij')
K2 = Kx**2 + Ky**2 + Kz**2
K2[0,0,0] = 1.0

def energy_for_alpha(alpha):
    n = n_star  # фиксировано!
    grad_z, grad_y, grad_x = torch.gradient(n, spacing=dx, dim=(1,2,3), edge_order=1)
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

    return (E_dir + E_sk + E_pot + E_em + E_spin).item()

# Сканирование α
alphas = np.linspace(0.003, 0.012, 50)
energies = []
for a in alphas:
    e = energy_for_alpha(a)
    energies.append(e)
    print(f"α={a:.6f} E={e:.4f}")

# Поиск минимума
min_idx = np.argmin(energies)
best_alpha = alphas[min_idx]
print(f"\nМинимум при α = {best_alpha:.6f} (отклонение {abs(best_alpha-0.007297)/0.007297*100:.2f}%)")

# График
plt.figure(figsize=(8,5))
plt.plot(alphas, energies, 'o-')
plt.axvline(1/137.036, color='r', linestyle='--', label='1/137')
plt.axvline(best_alpha, color='g', linestyle='--', label=f'min={best_alpha:.5f}')
plt.xlabel('α'); plt.ylabel('E'); plt.legend(); plt.grid()
plt.title('Энергия фиксированного хопфиона в разных вселенных')
plt.savefig('frozen_hopfion_scan.png', dpi=150)
plt.show()
