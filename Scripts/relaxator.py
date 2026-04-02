import torch
import numpy as np
import time
import sys
import matplotlib.pyplot as plt

print("="*80)
print(" ШАГ 1.5: КВАНТОВЫЙ РЕЛАКСАТОР (GRADIENT DESCENT FOR FIELDS)")
print("="*80)

# =====================================================================
# 1. ПАРАМЕТРЫ СЕТКИ И ИНИЦИАЛИЗАЦИЯ GPU
# =====================================================================
if not torch.cuda.is_available():
    print("[-] ОШИБКА: Необходим GPU для тензорной релаксации.")
    sys.exit()

device = torch.device('cuda:0')
print(f"[Устройство] Вычисления на: {torch.cuda.get_device_name(0)}")

grid_input = input("Размер сетки N (рекомендуется 200, это займет ~2 ГБ VRAM): ")
N = int(grid_input) if grid_input.strip().isdigit() else 200
iters_input = input("Количество шагов релаксации (рекомендуется 100-300): ")
ITERATIONS = int(iters_input) if iters_input.strip().isdigit() else 200

L = 8.0 
dx = 2 * L / N
dV = dx**3
scale_a = 1.5 

C2, C4, C0 = 1.0, 1.0, 0.1 

# =====================================================================
# 2. АНАЛИТИЧЕСКАЯ ГЕНЕРАЦИЯ (НАЧАЛЬНАЯ ДОГАДКА)
# =====================================================================
print("\nГенерация начального состояния (Полый Хопфион)...")
with torch.no_grad():
    x_ts = torch.linspace(-L, L, N, device=device, dtype=torch.float32)
    Y, X, Z = torch.meshgrid(x_ts, x_ts, x_ts, indexing='ij')
    
    xs, ys, zs = X / scale_a, Y / scale_a, Z / scale_a
    R2 = xs**2 + ys**2 + zs**2
    R = torch.sqrt(R2 + 1e-12)
    
    g_R2 = R2**2 # Наш полый профиль
    denom = 1.0 + g_R2
    
    X1 = 2 * xs * R / denom
    X2 = 2 * ys * R / denom
    X3 = 2 * zs * R / denom
    X4 = (1.0 - g_R2) / denom
    
    n1 = 2 * (X1 * X4 + X2 * X3)
    n2 = 2 * (X2 * X4 - X1 * X3)
    n3 = X4**2 + X3**2 - X1**2 - X2**2
    
    # Объединяем в тензор и говорим PyTorch отслеживать его градиенты!
    n_field = torch.stack([n1, n2, n3], dim=0)
    n_field.requires_grad = True

# =====================================================================
# 3. НАСТРОЙКА ОПТИМИЗАТОРА (РЕЛАКСАТОРА)
# =====================================================================
# Оптимизатор Adam будет "толкать" поле вниз по градиенту энергии
optimizer = torch.optim.Adam([n_field], lr=0.03)

# Маска границ (на краях куба должен быть строгий вакуум n = (0,0,1))
boundary_mask = torch.zeros((N, N, N), dtype=torch.bool, device=device)
boundary_mask[0, :, :] = True; boundary_mask[-1, :, :] = True
boundary_mask[:, 0, :] = True; boundary_mask[:, -1, :] = True
boundary_mask[:, :, 0] = True; boundary_mask[:, :, -1] = True

energy_history =[]
print(f"\nЗАПУСК РЕЛАКСАЦИИ НА {ITERATIONS} ШАГОВ:")
print("-" * 60)
print(f"{'Шаг':<6} | {'I2 (Дирихле)':<12} | {'I4 (Скирм)':<12} | {'I0 (Масса)':<10} | {'E_total':<10}")
print("-" * 60)

start_time = time.time()

# =====================================================================
# 4. ЦИКЛ ВАРИАЦИОННОГО СПУСКА
# =====================================================================
for step in range(ITERATIONS):
    optimizer.zero_grad()
    
    # Берем "внутреннюю" часть поля для вычисления производных без выхода за границы
    # Это позволяет считать градиенты мгновенно без padding
    nf_inner = n_field[:, 1:-1, 1:-1, 1:-1]
    
    dn_dx = (n_field[:, 2:, 1:-1, 1:-1] - n_field[:, :-2, 1:-1, 1:-1]) / (2 * dx)
    dn_dy = (n_field[:, 1:-1, 2:, 1:-1] - n_field[:, 1:-1, :-2, 1:-1]) / (2 * dx)
    dn_dz = (n_field[:, 1:-1, 1:-1, 2:] - n_field[:, 1:-1, 1:-1, :-2]) / (2 * dx)
    
    # I2 (Упругость)
    i2_dens = torch.sum(dn_dx**2 + dn_dy**2 + dn_dz**2, dim=0)
    
    # I4 (Кривизна / Защита от коллапса)
    c_xy = torch.cross(dn_dx, dn_dy, dim=0)
    c_yz = torch.cross(dn_dy, dn_dz, dim=0)
    c_zx = torch.cross(dn_dz, dn_dx, dim=0)
    i4_dens = torch.sum(c_xy**2 + c_yz**2 + c_zx**2, dim=0)
    
    # I0 (Потенциал / Масса)
    i0_dens = 1.0 - nf_inner[2]
    
    # Интегрирование (суммируем внутренний объем)
    I2_tot = torch.sum(i2_dens) * dV
    I4_tot = torch.sum(i4_dens) * dV
    I0_tot = torch.sum(i0_dens) * dV
    
    # ПОЛНАЯ ЭНЕРГИЯ ФУНКЦИОНАЛА
    E_tot = C2 * I2_tot + C4 * I4_tot + C0 * I0_tot
    
    # -----------------------------------------------------
    # МАГИЯ АВТОГРАДА: ВЫЧИСЛЕНИЕ ПРОИЗВОДНЫХ ВО ВСЕХ ТОЧКАХ
    # -----------------------------------------------------
    E_tot.backward()
    
    # Шаг градиентного спуска
    optimizer.step()
    
    # -----------------------------------------------------
    # ПРОЕКЦИЯ (Обеспечение условия |n| = 1 и Вакуума на краях)
    # -----------------------------------------------------
    with torch.no_grad():
        # Принудительный вакуум на границах
        n_field[0, boundary_mask] = 0.0
        n_field[1, boundary_mask] = 0.0
        n_field[2, boundary_mask] = 1.0
        
        # Нормировка длины вектора обратно к 1 во всем объеме
        norm = torch.sqrt(torch.sum(n_field**2, dim=0, keepdim=True))
        n_field.div_(norm)
        
    energy_history.append(E_tot.item())
    
    # Вывод прогресса
    if step == 0 or (step + 1) % 10 == 0 or step == ITERATIONS - 1:
        print(f"{step+1:<6} | {I2_tot.item():<12.4f} | {I4_tot.item():<12.4f} | {I0_tot.item():<10.4f} | {E_tot.item():<10.4f}")

calc_time = time.time() - start_time

print("-" * 60)
print(f"Релаксация завершена за {calc_time:.2f} сек.")
print(f"Начальная энергия : {energy_history[0]:.4f}")
print(f"Финальная энергия : {energy_history[-1]:.4f}")
delta = energy_history[0] - energy_history[-1]
print(f"Система 'остыла' на: {delta:.4f} единиц!")

# =====================================================================
# 5. ВИЗУАЛИЗАЦИЯ ИТОГОВОГО ("ОСТЫВШЕГО") ПОЛЯ
# =====================================================================
print("\nРендеринг результатов...")

# Получаем срез Z=0 остывшего поля
with torch.no_grad():
    mid = N // 2
    slice_n3 = n_field[2, :, :, mid].cpu().numpy()
    slice_n1 = n_field[0, ::4, ::4, mid].cpu().numpy()
    slice_n2 = n_field[1, ::4, ::4, mid].cpu().numpy()

plt.style.use('dark_background')
fig = plt.figure(figsize=(16, 6), facecolor='black')
fig.suptitle('Градиентная Релаксация Хопфиона', fontsize=18, fontweight='bold', color='white')

# График 1: Падение энергии
ax1 = fig.add_subplot(1, 2, 1)
ax1.set_facecolor('black')
ax1.plot(energy_history, color='#00ffff', lw=3, label='Полная Энергия $E_{tot}$')
ax1.set_title("Остывание системы (Стремление к оптимуму)", color='white', fontsize=14)
ax1.set_xlabel("Шаг релаксации (Итерация)", color='white')
ax1.set_ylabel("Энергия", color='white')
ax1.grid(True, color='#333333', linestyle=':')
ax1.legend(facecolor='black', edgecolor='white', labelcolor='white')
ax1.tick_params(colors='white')

# График 2: Форма ядра после релаксации
ax2 = fig.add_subplot(1, 2, 2)
ax2.set_facecolor('black')
img = ax2.imshow(slice_n3.T, extent=[-L, L, -L, L], origin='lower', cmap='magma')
ax2.set_title("Поле n3 после релаксации (Срез Z=0)", color='white', fontsize=14)
ax2.set_xlabel("X", color='white')
ax2.set_ylabel("Y", color='white')
fig.colorbar(img, ax=ax2, fraction=0.046, pad=0.04).ax.yaxis.set_tick_params(color='white')

# Добавляем векторы на срез
x_coords = np.linspace(-L, L, N)
X_m, Y_m = np.meshgrid(x_coords, x_coords, indexing='ij')
ax2.quiver(X_m[::4, ::4], Y_m[::4, ::4], slice_n1, slice_n2, color='#00ffff', alpha=0.7)
ax2.tick_params(colors='white')

plt.tight_layout()
plt.show()
