import torch
import torch.multiprocessing as mp
import numpy as np
import matplotlib.pyplot as plt
import time
import queue
import warnings
import os
from scipy.interpolate import make_interp_spline

warnings.filterwarnings("ignore")

# ==========================================
# ПАРАМЕТРЫ СЕТКИ
# ==========================================
N = 80            # Размер сетки (N^3)
L = 8.0           # Полуразмер области
dx = 2 * L / N
dV = dx**3
STEPS = 400       # Шагов оптимизации для каждой (alpha, omega)

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
# ОСНОВНАЯ ФУНКЦИЯ РЕЛАКСАЦИИ С ВРАЩЕНИЕМ
# ==========================================
def relax_soliton(alpha, omega, device, worker_name):
    # K-пространство для уравнения Пуассона
    kx = torch.fft.fftfreq(N, d=dx, device=device) * 2 * np.pi
    Kz, Ky, Kx = torch.meshgrid(kx, kx, kx, indexing='ij')
    K2 = Kx**2 + Ky**2 + Kz**2
    K2[0, 0, 0] = 1.0
    
    w = init_hopfion(device).clone().requires_grad_(True)
    optimizer = torch.optim.Adam([w], lr=0.005)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=150, gamma=0.5)
    
    best_E = float('inf')
    best_state = None
    LOCAL_STEPS = 300
    
    for step in range(LOCAL_STEPS):
        optimizer.zero_grad()
        
        n = w / torch.norm(w, dim=0, keepdim=True).clamp(min=1e-12)
        grad_z, grad_y, grad_x = torch.gradient(n, spacing=dx, dim=(1, 2, 3), edge_order=1)
        
        # Кинетическая энергия (I2)
        E_dir = 0.5 * torch.sum(grad_x**2 + grad_y**2 + grad_z**2) * dV
        
        # Энергия Скирма (I4)
        cross_yz = torch.cross(grad_y, grad_z, dim=0)
        cross_zx = torch.cross(grad_z, grad_x, dim=0)
        cross_xy = torch.cross(grad_x, grad_y, dim=0)
        E_sk = 0.5 * torch.sum(cross_xy**2 + cross_yz**2 + cross_zx**2) * dV
        
        # Потенциальная энергия (массовый член)
        E_pot = 3 * alpha * torch.sum(1 + n[2]) * dV
        
        # Электромагнитная энергия
        B_x = torch.sum(n * cross_yz, dim=0)
        B_y = torch.sum(n * cross_zx, dim=0)
        B_z = torch.sum(n * cross_xy, dim=0)
        B_mag = torch.sqrt(B_x**2 + B_y**2 + B_z**2 + 1e-12)
        rho = np.sqrt(4 * np.pi * alpha) * (B_mag / (8 * np.pi))
        
        rho_fft = torch.fft.fftn(rho)
        Phi_fft = rho_fft / K2
        Phi_fft[0,0,0] = 0.0
        Phi = torch.fft.ifftn(Phi_fft).real
        E_em = 0.5 * torch.sum(rho * Phi) * dV
        
        # Центробежная энергия вращения (вокруг оси Z)
        I_rot = torch.sum(n[0]**2 + n[1]**2) * dV
        E_rot = -0.5 * omega**2 * I_rot
        
        E_tot = E_dir + E_sk + E_pot + E_em + E_rot
        E_tot.backward()
        
        # Заморозка границ
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
                # Вычисляем угловой момент для лучшего состояния
                J_z = omega * I_rot.item()   # в безразмерных единицах
                best_state = {
                    'E_pot': E_pot.item(),
                    'E_em': E_em.item(),
                    'I_rot': I_rot.item(),
                    'J_z': J_z
                }
                
        if step % 150 == 0 and worker_name == "GPU-0":
            print(f"   [{worker_name}] Step {step:3d} | E: {E_tot.item():.2f} | "
                  f"E_pot: {E_pot.item():.2f} | E_em: {E_em.item():.3f} | J_z: {best_state['J_z']:.3f}")
    
    print(f"[{worker_name}] α={alpha:.5f} ω={omega:.3f} | E={best_E:.2f} | J_z={best_state['J_z']:.3f}")
    return alpha, omega, best_E, best_state

# ==========================================
# ВОРКЕР ДЛЯ ПАРАЛЛЕЛЬНОЙ ОБРАБОТКИ
# ==========================================
def worker_process(worker_id, device_str, task_queue, result_queue):
    device = torch.device(device_str)
    worker_name = f"GPU-{worker_id}"
    while not task_queue.empty():
        try:
            alpha, omega = task_queue.get_nowait()
        except queue.Empty:
            break
        res = relax_soliton(alpha, omega, device, worker_name)
        result_queue.put(res)

# ==========================================
# ЗАПУСК СКАНИРОВАНИЯ ПАРАМЕТРОВ
# ==========================================
if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    
    print("="*75)
    print(" 3D ВРАЩАЮЩИЙСЯ ХОПФИОН: ПОИСК α И СПИНА")
    print("="*75)
    
    num_gpus = torch.cuda.device_count()
    devices = [f"cuda:{i}" for i in range(num_gpus)]
    for i, dev in enumerate(devices):
        print(f"[+] GPU {i}: {torch.cuda.get_device_name(i)}")

    # Сетки параметров
    alphas = np.linspace(0.005, 0.009, 12)
    omegas = np.linspace(0.0, 2.0, 8)   # частота вращения
    
    # Создаём список задач (все комбинации)
    tasks = [(a, w) for a in alphas for w in omegas]
    
    task_queue = mp.Queue()
    result_queue = mp.Queue()
    for t in tasks:
        task_queue.put(t)
        
    print(f"\nЗапуск 3D-Релаксации ({len(tasks)} задач) на {num_gpus} GPU...")
    start_time = time.time()
    
    processes = [mp.Process(target=worker_process, args=(i, dev, task_queue, result_queue))
                 for i, dev in enumerate(devices)]
    for p in processes:
        p.start()
        
    results = []
    completed = 0
    while completed < len(tasks):
        res = result_queue.get()
        results.append(res)
        completed += 1
        
    for p in processes:
        p.join()
        
    print("\n" + "="*75)
    print("РЕЗУЛЬТАТЫ СКАНИРОВАНИЯ")
    print("="*75)
    
    # Анализ: для каждого omega находим alpha с минимальной энергией
    # и соответствующий спин J_z.
    import pandas as pd
    df = pd.DataFrame(results, columns=['alpha', 'omega', 'E', 'state'])
    # Извлекаем J_z из state
    df['J_z'] = df['state'].apply(lambda s: s['J_z'])
    df['I_rot'] = df['state'].apply(lambda s: s['I_rot'])
    
    # Группируем по omega и находим минимум энергии
    grouped = df.groupby('omega').apply(lambda g: g.loc[g['E'].idxmin()])
    best_omega_row = grouped.loc[grouped['E'].idxmin()]
    
    print("\nОптимальные параметры по минимуму энергии:")
    print(grouped[['alpha', 'E', 'J_z']].to_string())
    
    print("\n" + "="*75)
    print(f"Глобальный минимум: alpha = {best_omega_row['alpha']:.6f}, omega = {best_omega_row['omega']:.3f}")
    print(f"Энергия: {best_omega_row['E']:.2f}")
    print(f"Угловой момент J_z: {best_omega_row['J_z']:.3f} (безразм.)")
    print(f"Экспериментальное alpha = 1/137 = 0.007297")
    print(f"Отклонение alpha: {abs(best_omega_row['alpha'] - 0.007297)/0.007297*100:.2f}%")
    print(f"Время расчета: {time.time() - start_time:.1f} сек")
    print("="*75)
    
    # Построение графиков
    plt.style.use('dark_background')
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # График 1: E(alpha) для разных omega
    ax1 = axes[0]
    for w in omegas:
        sub = df[df['omega'] == w]
        ax1.plot(sub['alpha'], sub['E'], 'o-', label=f'ω={w:.2f}')
    ax1.axvline(1/137.036, color='yellow', linestyle=':', label='1/137')
    ax1.set_xlabel('α')
    ax1.set_ylabel('Полная энергия')
    ax1.set_title('Энергия вращающегося хопфиона')
    ax1.legend()
    ax1.grid(alpha=0.3)
    
    # График 2: J_z(omega) для оптимальных alpha
    ax2 = axes[1]
    ax2.plot(grouped['omega'], grouped['J_z'], 'o-', color='cyan', lw=2)
    ax2.axhline(0.5, color='red', linestyle='--', label='Спин 1/2 (целевой)')
    ax2.set_xlabel('Частота вращения ω')
    ax2.set_ylabel('Угловой момент J_z (безразм.)')
    ax2.set_title('Спин как функция частоты')
    ax2.legend()
    ax2.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("rotating_hopfion_scan.png", dpi=150)
    plt.show()
    
    # Сохраняем данные
    df.to_csv("rotating_hopfion_results.csv", index=False)
    print("Результаты сохранены в rotating_hopfion_results.csv")
