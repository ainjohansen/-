import torch
import numpy as np
import matplotlib.pyplot as plt
import time

# ==========================================
# ПАРАМЕТРЫ СЕТКИ И ФИЗИКИ
# ==========================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Вычисления выполняются на: {DEVICE}")

N = 64            # Размер 3D сетки (для прототипа 64^3, на кластере ставьте 128-256)
L = 6.0           # Физический размер ящика [-L, L]
dx = 2 * L / N
dV = dx**3

# ==========================================
# 1. ИНИЦИАЛИЗАЦИЯ ПОЛЯ (АНЗАЦ ХОПФА Q=1)
# ==========================================
def init_hopfion():
    """Создает начальное поле n(x,y,z) в виде закрученного тора (Хопфиона Q=1)"""
    x = torch.linspace(-L, L, N, dtype=torch.float64, device=DEVICE)
    y = torch.linspace(-L, L, N, dtype=torch.float64, device=DEVICE)
    z = torch.linspace(-L, L, N, dtype=torch.float64, device=DEVICE)
    Z, Y, X = torch.meshgrid(z, y, x, indexing='ij')
    
    R2 = X**2 + Y**2 + Z**2
    
    # Стереографическая проекция R^3 на сферу S^3, затем расслоение Хопфа на S^2
    Z0_re = 2 * X / (R2 + 1)
    Z0_im = 2 * Y / (R2 + 1)
    Z1_re = 2 * Z / (R2 + 1)
    Z1_im = (R2 - 1) / (R2 + 1)
    
    # Комплексное деление W = Z0 / Z1
    denom = Z1_re**2 + Z1_im**2 + 1e-12
    W_re = (Z0_re * Z1_re + Z0_im * Z1_im) / denom
    W_im = (Z0_im * Z1_re - Z0_re * Z1_im) / denom
    W_mod2 = W_re**2 + W_im**2
    
    # Проекция на 3-вектор поля n (единичная сфера S^2)
    n1 = 2 * W_re / (W_mod2 + 1)
    n2 = 2 * W_im / (W_mod2 + 1)
    n3 = (1 - W_mod2) / (W_mod2 + 1)
    
    # Складываем в тензор [3, N, N, N]
    n = torch.stack([n1, n2, n3], dim=0)
    return n

# ==========================================
# 2. ПОДГОТОВКА K-ПРОСТРАНСТВА ДЛЯ FFT (ПУАССОН)
# ==========================================
# Для решения \nabla^2 \Phi = -B^0 в пространстве Фурье
kx = torch.fft.fftfreq(N, d=dx, device=DEVICE) * 2 * np.pi
ky = torch.fft.fftfreq(N, d=dx, device=DEVICE) * 2 * np.pi
kz = torch.fft.fftfreq(N, d=dx, device=DEVICE) * 2 * np.pi
Kz, Ky, Kx = torch.meshgrid(kz, ky, kx, indexing='ij')
K2 = Kx**2 + Ky**2 + Kz**2
K2[0, 0, 0] = 1.0 # Избегаем деления на 0 для нулевой гармоники (будет занулена позже)

# ==========================================
# 3. ФУНКЦИЯ ВЫЧИСЛЕНИЯ ЭНЕРГИИ И ПОЛЯ
# ==========================================
def compute_physics(n_field, alpha):
    # Нормализуем поле, чтобы строго |n| = 1
    n = n_field / torch.norm(n_field, dim=0, keepdim=True).clamp(min=1e-12)
    
    # Считаем градиенты через конечные разности (сдвиги тензора)
    # n_x = \partial n / \partial x и т.д.
    n_x = (torch.roll(n, shifts=-1, dims=3) - torch.roll(n, shifts=1, dims=3)) / (2 * dx)
    n_y = (torch.roll(n, shifts=-1, dims=2) - torch.roll(n, shifts=1, dims=2)) / (2 * dx)
    n_z = (torch.roll(n, shifts=-1, dims=1) - torch.roll(n, shifts=1, dims=1)) / (2 * dx)
    
    # --- 1. Кинетический член (Dirichlet) ---
    dirichlet_density = 0.5 * (torch.sum(n_x**2 + n_y**2 + n_z**2, dim=0))
    E_dir = torch.sum(dirichlet_density) * dV
    
    # --- 2. Член Скирма ---
    # \partial_i n \times \partial_j n
    cross_xy = torch.cross(n_x, n_y, dim=0)
    cross_yz = torch.cross(n_y, n_z, dim=0)
    cross_zx = torch.cross(n_z, n_x, dim=0)
    skyrme_density = 0.25 * (torch.sum(cross_xy**2 + cross_yz**2 + cross_zx**2, dim=0))
    E_sk = torch.sum(skyrme_density) * dV
    
    # --- 3. Потенциал (массовый член) ---
    # V(n) = 3*alpha * (1 - n3)
    pot_density = 3 * alpha * (1 - n[2])
    E_pot = torch.sum(pot_density) * dV
    
    # --- 4. Топологический заряд и Кулон (Maxwell) ---
    # B^0 = 1/(8*pi) * e_ijk * n \cdot (\partial_i n \times \partial_j n)
    B0_density = (1 / (8 * np.pi)) * (
        torch.sum(n * cross_yz, dim=0) + # Это упрощенно, полная свертка:
        torch.sum(n_x * torch.cross(n_y, n_z, dim=0), dim=0) 
    )
    # Топологический заряд системы (должен быть ~1.0)
    Q = torch.sum(B0_density) * dV
    
    # Решаем уравнение Пуассона через 3D FFT: \Delta \Phi = - B^0
    B0_fft = torch.fft.fftn(B0_density)
    Phi_fft = B0_fft / K2
    Phi_fft[0, 0, 0] = 0.0 # Граничное условие на бесконечности
    Phi = torch.fft.ifftn(Phi_fft).real
    
    # Электромагнитная энергия (условно связываем с альфа)
    # Коэффициент перед кулоновской энергией масштабируется с alpha
    em_density = 0.5 * B0_density * Phi
    E_em = alpha * torch.sum(em_density) * dV * (4 * np.pi) 
    
    E_total = E_dir + E_sk + E_pot + E_em
    return E_total, E_dir, E_sk, E_pot, E_em, Q, n

# ==========================================
# 4. ЦИКЛ ОПТИМИЗАЦИИ (РЕЛАКСАЦИЯ ПОЛЯ)
# ==========================================
def relax_soliton(alpha, steps=250):
    print(f"\n--- Релаксация для alpha = {alpha:.6f} ---")
    
    # Инициализируем ненормализованное поле w (optimizer будет менять его)
    w = init_hopfion().clone().requires_grad_(True)
    
    # Используем мощный оптимизатор Adam
    optimizer = torch.optim.Adam([w], lr=0.05)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)
    
    best_E = float('inf')
    
    for step in range(steps):
        optimizer.zero_grad()
        
        # Считаем физику
        E_tot, E_dir, E_sk, E_pot, E_em, Q, _ = compute_physics(w, alpha)
        
        # Градиентный спуск
        E_tot.backward()
        optimizer.step()
        scheduler.step()
        
        if E_tot.item() < best_E:
            best_E = E_tot.item()
            
        if step % 50 == 0 or step == steps - 1:
            print(f"Шаг {step:3d} | E_tot: {E_tot.item():.5f} | Q: {Q.item():.4f} | E_em: {E_em.item():.5f}")
            
    return best_E

# ==========================================
# 5. ГЛАВНЫЙ БЛОК: СКАНИРОВАНИЕ АЛЬФА
# ==========================================
if __name__ == "__main__":
    start_time = time.time()
    
    # Ищем минимум вокруг реального значения 1/137 ~ 0.00729
    alphas = np.linspace(0.0065, 0.0080, 10)
    energies =[]
    
    for a in alphas:
        # Для каждой альфы "отпускаем" поле и даем ему принять оптимальную форму
        E_min = relax_soliton(a, steps=200)
        energies.append(E_min)
        
    # Ищем точку минимума на графике
    energies = np.array(energies)
    best_idx = np.argmin(energies)
    best_alpha = alphas[best_idx]
    
    print("\n" + "="*50)
    print(f"ЭКСПЕРИМЕНТ ЗАВЕРШЕН ЗА {time.time() - start_time:.1f} сек")
    print(f"Минимум энергии найден при alpha = {best_alpha:.6f}")
    print(f"Реальная константа 1/137      = {1/137.036:.6f}")
    print("="*50)
    
    # Отрисовка
    plt.style.use('dark_background')
    plt.figure(figsize=(8, 5))
    plt.plot(alphas, energies, 'o-', color='cyan', lw=2)
    plt.axvline(best_alpha, color='yellow', linestyle='--', label=f'Model Min: {best_alpha:.6f}')
    plt.axvline(1/137.036, color='red', linestyle=':', label='1/137.036')
    plt.title("Энергия 3D Хопфиона в зависимости от $\\alpha$")
    plt.xlabel("Параметр $\\alpha$")
    plt.ylabel("Полная Энергия (безразмерная)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()
