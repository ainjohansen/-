import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.collections import LineCollection

def generate_nucleon_curve(beta_phase, num_points=10000):
    t = np.linspace(0, 2 * np.pi, num_points)
    R_torus, a_torus = 2.0, 0.8
    xp = (R_torus + a_torus * np.cos(3 * t)) * np.cos(2 * t)
    yp = (R_torus + a_torus * np.cos(3 * t)) * np.sin(2 * t)
    zp = a_torus * np.sin(3 * t)

    if beta_phase > 0:
        twist_freq = 15  
        twist_amp = 0.0247371 * beta_phase 
        x = xp + twist_amp * np.cos(twist_freq * t)
        y = yp + twist_amp * np.sin(twist_freq * t)
        z = zp + twist_amp * np.cos(twist_freq * t + np.pi/2)
        return x, y, z
    else:
        return xp, yp, zp

# Настройка стиля
plt.style.use('dark_background')
fig = plt.figure(figsize=(14, 6), facecolor='black')

# Функция для красивой отрисовки с градиентом глубины
def plot_beautiful_knot(ax, beta, title, cmap_name):
    x, y, z = generate_nucleon_curve(beta)
    
    # Создаем сегменты для раскраски линии по Z-координате (глубине)
    # Это полностью убирает визуальные артефакты слипания!
    ax.scatter(x, y, z, c=z, cmap=cmap_name, s=2, alpha=0.8, edgecolors='none')
    
    # Настраиваем идеальный ракурс камеры
    ax.view_init(elev=35, azim=45)
    
    # Убираем оси и фон
    ax.set_axis_off()
    ax.set_facecolor('black')
    ax.xaxis.set_pane_color((0,0,0,0))
    ax.yaxis.set_pane_color((0,0,0,0))
    ax.zaxis.set_pane_color((0,0,0,0))
    
    ax.set_title(title, color='white', fontsize=16, pad=0)
    
    # Фиксируем масштаб, чтобы узлы не сплющивало
    max_range = np.array([x.max()-x.min(), y.max()-y.min(), z.max()-z.min()]).max() / 2.0
    mid_x = (x.max()+x.min()) * 0.5
    mid_y = (y.max()+y.min()) * 0.5
    mid_z = (z.max()+z.min()) * 0.5
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)

# Левый график: Протон
ax1 = fig.add_subplot(1, 2, 1, projection='3d')
plot_beautiful_knot(ax1, 0.0, "Протон (Гладкий Трилистник)", 'cool')

# Правый график: Нейтрон
ax2 = fig.add_subplot(1, 2, 2, projection='3d')
plot_beautiful_knot(ax2, 1.0, "Нейтрон (Скрученный Трилистник, $\\beta=1.0$)", 'spring')

plt.tight_layout()
plt.savefig('beautiful_knots.png', dpi=300, bbox_inches='tight', facecolor='black')
print("График сохранен как beautiful_knots.png")
plt.show()
