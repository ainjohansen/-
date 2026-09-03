import torch
import numpy as np
import math
import time

# ==========================================
# ПАРАМЕТРЫ СЕТКИ И ФИЗИКИ
# ==========================================
N = 80
L = 8.0
dx = 2 * L / N
dV = dx**3
alpha_target = 0.0072973525693
J_fixed = 0.5
skyrme_coef = 10.9

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Устройство: {device}")

# ==========================================
# ИНИЦИАЛИЗАЦИЯ ХОПФИОНА Q=1
# ==========================================
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

# ==========================================
# ВЫЧИСЛЕНИЕ ЭНЕРГИИ И ГРАДИЕНТА
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
# ПОДГОТОВКА K-ПРОСТРАНСТВА ДЛЯ ПУАССОНА
# ==========================================
kx = torch.fft.fftfreq(N, d=dx, device=device) * 2 * math.pi
Kz, Ky, Kx = torch.meshgrid(kx, kx, kx, indexing='ij')
K2 = Kx**2 + Ky**2 + Kz**2
K2[0,0,0] = 1.0

# ==========================================
# ОСНОВНАЯ ОПТИМИЗАЦИЯ
# ==========================================
w = init_hopfion(device).clone().requires_grad_(True)

# 1. Предварительный спуск Adam
print("Этап 1: Adam")
optimizer = torch.optim.Adam([w], lr=0.005)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=400, gamma=0.5)

for step in range(800):
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
    if step % 100 == 0:
        print(f"  Step {step:3d}: E = {E.item():.4f}")

# 2. Точная доводка LBFGS
print("Этап 2: LBFGS")
optimizer = torch.optim.LBFGS([w], max_iter=500, tolerance_grad=1e-9,
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

# Финальная энергия и момент инерции
with torch.no_grad():
    E_final, I_final = compute_energy(w, alpha_target, J_fixed, skyrme_coef, K2)
    n_final = w / torch.norm(w, dim=0, keepdim=True).clamp(min=1e-12)

print(f"\nФинальная энергия: {E_final.item():.6f}")
print(f"Момент инерции: {I_final.item():.6f}")
print(f"Проверка спина: {torch.sqrt(2 * (J_fixed**2 / (2*I_final)) * I_final).item():.6f}")

# ==========================================
# СОХРАНЕНИЕ ПРОФИЛЯ
# ==========================================
torch.save({'n': n_final.cpu(), 'alpha': alpha_target, 'J': J_fixed,
            'skyrme_coef': skyrme_coef, 'L': L, 'N': N}, 'electron_hopfion.pt')
print("Эталонный профиль сохранён в 'electron_hopfion.pt'")
