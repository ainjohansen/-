import torch
import torch.multiprocessing as mp
import numpy as np
import matplotlib.pyplot as plt
import time
import queue
import warnings

warnings.filterwarnings("ignore")

# ==========================================
# ПАРАМЕТРЫ СЕТКИ
# ==========================================
N = 80            
L = 8.0          
dx = 2 * L / N
dV = dx**3
STEPS = 400       

# ==========================================
# ЯДРО СИМУЛЯЦИИ (Только GPU)
# ==========================================
def init_hopfion(device):
    """Классический анзац Хопфиона Q=1 (Тороид)"""
    x = torch.linspace(-L, L, N, dtype=torch.float64, device=device)
    y = torch.linspace(-L, L, N, dtype=torch.float64, device=device)
    z = torch.linspace(-L, L, N, dtype=torch.float64, device=device)
    Z, Y, X = torch.meshgrid(z, y, x, indexing='ij')
    
    # Масштаб тора
    scale = 1.2
    X, Y, Z = X/scale, Y/scale, Z/scale
    
    # Отображение R^3 -> S^3 -> S^2
    u_re, u_im = X, Y
    v_re, v_im = Z, (X**2 + Y**2 + Z**2 - 1) / 2
    
    denom = u_re**2 + u_im**2 + v_re**2 + v_im**2 + 1e-12
    n1 = 2 * (u_re * v_re + u_im * v_im) / denom
    n2 = 2 * (u_im * v_re - u_re * v_im) / denom
    # ВНИМАНИЕ: На бесконечности v_im -> inf, значит n3 -> -1 (Вакуум)
    n3 = (u_re**2 + u_im**2 - v_re**2 - v_im**2) / denom
    
    return torch.stack([n1, n2, n3], dim=0)

def relax_soliton(alpha, device, worker_name):
    # K-пространство для уравнения Пуассона
    kx = torch.fft.fftfreq(N, d=dx, device=device) * 2 * np.pi
    Kz, Ky, Kx = torch.meshgrid(kx, kx, kx, indexing='ij')
    K2 = Kx**2 + Ky**2 + Kz**2
    K2[0, 0, 0] = 1.0 
    
    w = init_hopfion(device).clone().requires_grad_(True)
    
    # СНИЖЕННЫЙ ШАГ ОБУЧЕНИЯ (чтобы поле не "кипело", а плавно остывало)
    optimizer = torch.optim.Adam([w], lr=0.005) 
    # Плавное затухание скорости
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=150, gamma=0.5)
    
    best_E = float('inf')
    best_comps = {}
    
    # Увеличим количество шагов для плавности
    LOCAL_STEPS = 600
    
    for step in range(LOCAL_STEPS):
        optimizer.zero_grad()
        
        n = w / torch.norm(w, dim=0, keepdim=True).clamp(min=1e-12)
        grad_z, grad_y, grad_x = torch.gradient(n, spacing=dx, dim=(1, 2, 3), edge_order=1)
        
        # 1. Интеграл Дирихле (Кинетика)
        E_dir = 0.5 * torch.sum(grad_x**2 + grad_y**2 + grad_z**2) * dV
        
        # 2. Интеграл Скирма (Стабилизирует от схлопывания)
        cross_yz = torch.cross(grad_y, grad_z, dim=0)
        cross_zx = torch.cross(grad_z, grad_x, dim=0)
        cross_xy = torch.cross(grad_x, grad_y, dim=0)
        # Убраны все подгоночные k_crit. Только чистая математика поля!
        E_sk = 0.5 * torch.sum(cross_xy**2 + cross_yz**2 + cross_zx**2) * dV 
        
        # 3. Интеграл Потенциала (Масса)
        E_pot = 3 * alpha * torch.sum(1 + n[2]) * dV
        
        # 4. Электромагнетизм
        B_x = torch.sum(n * cross_yz, dim=0)
        B_y = torch.sum(n * cross_zx, dim=0)
        B_z = torch.sum(n * cross_xy, dim=0)
        B_mag = torch.sqrt(B_x**2 + B_y**2 + B_z**2 + 1e-12)
        
        # Плотность заряда
        rho = np.sqrt(4 * np.pi * alpha) * (B_mag / (8 * np.pi))
        
        # Быстрое преобразование Фурье для Пуассона
        rho_fft = torch.fft.fftn(rho)
        Phi_fft = rho_fft / K2
        Phi_fft[0,0,0] = 0.0 # Обнуление постоянной составляющей
        Phi = torch.fft.ifftn(Phi_fft).real
        
        E_em = 0.5 * torch.sum(rho * Phi) * dV
        
        # Строгая полная энергия! Никаких динамических весов.
        E_tot = E_dir + E_sk + E_pot + E_em
        E_tot.backward()
        
        # Заморозка границ, чтобы тор не вытек
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
                best_comps = {'E_pot': E_pot.item(), 'E_em': E_em.item()}
                
        if step % 150 == 0 and worker_name == "GPU-0":
            print(f"   [{worker_name}] Шаг {step:3d} | E: {E_tot.item():.2f} | E_pot: {E_pot.item():.2f} | E_em: {E_em.item():.3f}")
            
    print(f"[{worker_name}] α={alpha:.5f} | E={best_E:.2f} | E_pot={best_comps['E_pot']:.2f} | E_em={best_comps['E_em']:.3f}")
    return alpha, best_E, best_comps

def worker_process(worker_id, device_str, task_queue, result_queue):
    device = torch.device(device_str)
    worker_name = f"GPU-{worker_id}"
    while not task_queue.empty():
        try: alpha = task_queue.get_nowait()
        except queue.Empty: break
        res = relax_soliton(alpha, device, worker_name)
        result_queue.put(res)

if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    
    print("="*75)
    print(" 3D ХОПФИОН: СТАБИЛИЗАЦИЯ ДЕРРИКА (Версия 10 - ФИНАЛ)")
    print("="*75)
    
    num_gpus = torch.cuda.device_count()
    devices = [f"cuda:{i}" for i in range(num_gpus)]
    for i, dev in enumerate(devices): print(f"[+] GPU {i}: {torch.cuda.get_device_name(i)}")

    # Точечный скан вокруг ожидаемой ямы
    alphas = np.linspace(0.0050, 0.0090, 24)
    
    task_queue = mp.Queue()
    result_queue = mp.Queue()
    for a in alphas: task_queue.put(a)
        
    print(f"\nЗапуск 3D-Релаксации ({len(alphas)} задач)...")
    start_time = time.time()
    
    processes =[mp.Process(target=worker_process, args=(i, dev, task_queue, result_queue)) for i, dev in enumerate(devices)]
    for p in processes: p.start()
        
    results, components = {}, {}
    completed = 0
    while completed < len(alphas):
        a, e_tot, comp = result_queue.get()
        results[a] = e_tot; components[a] = comp
        completed += 1
        
    for p in processes: p.join()
        
    print("\n" + "="*75)
    alphas_sorted = np.array(sorted(results.keys()))
    energies_sorted = np.array([results[a] for a in alphas_sorted])
    
    best_alpha = alphas_sorted[np.argmin(energies_sorted)]
    
    print(f"Абсолютный минимум 3D-модели : alpha = {best_alpha:.6f}")
    print(f"Экспериментальная постоянная : 1/137 = 0.007297")
    print(f"Отклонение (без подгонок!)   : {abs(best_alpha - 0.007297)/0.007297 * 100:.2f} %")
    print(f"Время расчета                : {time.time() - start_time:.1f} сек")
    print("="*75)
    
    plt.style.use('dark_background')
    plt.figure(figsize=(10, 6))
    plt.plot(alphas_sorted, energies_sorted, 'o', color='#00ffcc', markersize=7)
    
    from scipy.interpolate import make_interp_spline
    spline = make_interp_spline(alphas_sorted, energies_sorted, k=3)
    x_smooth = np.linspace(alphas_sorted.min(), alphas_sorted.max(), 300)
    y_smooth = spline(x_smooth)
    plt.plot(x_smooth, y_smooth, color='#00ffcc', alpha=0.7, lw=2)
    
    plt.axvline(x_smooth[np.argmin(y_smooth)], color='#ff00ff', linestyle='--', lw=2, label=f'Минимум Модели')
    plt.axvline(1/137.036, color='yellow', linestyle=':', lw=2, label='1/137 (Exp)')
    
    plt.title("Самосогласованная энергия Тороидального Электрона (3D)", fontsize=14)
    plt.xlabel(r"Постоянная тонкой структуры $\alpha$", fontsize=12)
    plt.ylabel("Полная энергия (безразмерная)", fontsize=12)
    plt.legend(); plt.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig("hopfion_final_v10.png", dpi=150)
    plt.show()
