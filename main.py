import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.path as mpath
import random
import scipy.ndimage as ndimage
import os
from tqdm import tqdm

# настройки сетки
ROWS = 100
COLS = 100
STEPS = 300
OBSTACLE_SIZE = 10
PROFILE_FILE = "ls417.txt"

DIRECTIONS = np.array([
    [0, 1], [-1, 1], [-1, -1], [0, -1], [1, -1], [1, 1]
])
NEIGHBORS_DELTAS = [(0, 1), (-1, 1), (-1, -1), (0, -1), (1, -1), (1, 1)]

def initialize_grid():
    ni = np.zeros((ROWS, COLS, 6), dtype=np.int8)
    walls = np.zeros((ROWS, COLS), dtype=bool)
    return ni, walls

def add_obstacle(walls, center_r, center_c, size, filename):
    r, c = np.ogrid[:ROWS, :COLS]
    
    # --- старые фигуры, если понадобятся ---
    # круг
    # mask = (r - center_r)**2 + (c - center_c)**2 <= size**2
    # квадрат
    # mask = (np.abs(r - center_r) <= size) & (np.abs(c - center_c) <= size)
    # треугольник вправо
    # mask = ((c - center_c) >= -size) & ((c - center_c) <= size) & (np.abs(r - center_r) <= (size - (c - center_c)) / 2)
    # треугольник влево
    # mask = ((c - center_c) >= -size) & ((c - center_c) <= size) & (np.abs(r - center_r) <= (size + (c - center_c)) / 2)
    
    # читаем профиль из файла
    if not os.path.exists(filename):
        print("нет файла, ставлю круг")
        mask = (r - center_r)**2 + (c - center_c)**2 <= size**2
    else:
        # пропускаем первую строку с названием
        coords = np.loadtxt(filename, skiprows=1)
        x = coords[:, 0]
        y = coords[:, 1]
        
        # маштабируем чтобы влезло в сетку
        scale = 60
        offset_x = COLS // 4
        offset_y = ROWS // 2
        
        gx = (x * scale + offset_x).astype(int)
        gy = (offset_y - y * scale).astype(int)
        
        # заливаем полигон
        pts = np.column_stack((gx, gy))
        path = mpath.Path(pts)
        
        c_grid, r_grid = np.meshgrid(np.arange(COLS), np.arange(ROWS))
        grid_points = np.column_stack((c_grid.ravel(), r_grid.ravel()))
        
        inside = path.contains_points(grid_points)
        mask = inside.reshape(ROWS, COLS)
    
    walls[mask] = True
    return walls

def create_inflow(ni, walls, density=0.5):
    # дуем слева
    for r in range(ROWS):
        if random.random() < density:
            d = random.choice([0, 1, 5])
            ni[r, 0, d] = 1 
            if d == 1:
                ni[max(0, r-1), 1, 1] = 1
            elif d == 5:
                ni[min(ROWS-1, r+1), 1, 5] = 1
    return ni

def update(ni, walls):
    new_ni = np.zeros_like(ni)
    for i in range(6):
        dr, dc = NEIGHBORS_DELTAS[i]
        shifted = np.roll(ni[:,:,i], shift=(dr, dc), axis=(0, 1))
        
        # отскок от стен
        bounce = np.roll(shifted, shift=(-dr, -dc), axis=(0, 1)) * walls
        
        new_ni[:,:,i] = np.where(walls, bounce, shifted)
        new_ni[:,:,(i+3)%6] += np.where(walls, bounce, 0)
    
    new_ni = np.clip(new_ni, 0, 1)
    
    # чистим правый край чтобы не забивался
    new_ni[:, -1, :] = 0 
    return new_ni

def calculate_macroscopic(ni):
    rho = np.sum(ni, axis=2).astype(float)
    v_row = np.sum(ni * DIRECTIONS[:, 0], axis=2).astype(float)
    v_col = np.sum(ni * DIRECTIONS[:, 1], axis=2).astype(float)
    
    # делим на плотность где она не ноль
    with np.errstate(divide='ignore', invalid='ignore'):
        v_row_n = np.divide(v_row, rho, out=np.zeros_like(v_row), where=rho!=0)
        v_col_n = np.divide(v_col, rho, out=np.zeros_like(v_col), where=rho!=0)
        
    v_mag = np.sqrt(v_row_n**2 + v_col_n**2)
    
    # блюр чтобы убрать шум
    v_smooth = ndimage.gaussian_filter(v_mag, sigma=2)
    rho_smooth = ndimage.gaussian_filter(rho, sigma=2)
    
    return rho_smooth, v_row_n, v_col_n, v_smooth

def main():
    ni, walls = initialize_grid()
    cr, cc = ROWS//2, COLS//2
    
    walls = add_obstacle(walls, cr, cc, OBSTACLE_SIZE, PROFILE_FILE)
    
    ni = create_inflow(ni, walls, density=0.8)
    for r in range(ROWS):
        for c in range(5):
            if not walls[r, c]:
                ni[r, c, 0] = 1

    frames_v = []
    frames_rho = []
    
    print("погнали...")
    for step in tqdm(range(STEPS)):
        ni = create_inflow(ni, walls, density=0.3)
        ni = update(ni, walls)
        
        if step % 5 == 0:
            rho, _, _, v_mag = calculate_macroscopic(ni)
            frames_v.append(v_mag.copy())
            frames_rho.append(rho.copy())

    # графики
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    im1 = axes[0].imshow(frames_rho[-1], cmap='Blues', origin='upper')
    fig.colorbar(im1, ax=axes[0], label='Плотность')
    axes[0].set_title('Плотность')
    axes[0].contour(walls, levels=[0.5], colors='purple', linewidths=3)

    im2 = axes[1].imshow(frames_v[-1], cmap='inferno', origin='upper', vmin=0, vmax=1.5)
    fig.colorbar(im2, ax=axes[1], label='Скорость')
    axes[1].set_title('Скорость')
    axes[1].contour(walls, levels=[0.5], colors='purple', linewidths=3)
    
    plt.tight_layout()
    plt.savefig('flow_results.png', dpi=300)
    plt.show()

    # видео
    fig, ax = plt.subplots(figsize=(8, 8))
    im = ax.imshow(frames_v[0], cmap='inferno', origin='upper', vmin=0, vmax=1.5)
    ax.set_title('Обтекание')
    fig.colorbar(im, ax=ax, label='Скорость')
    
    ax.contour(walls, levels=[0.5], colors='purple', linewidths=3)
    
    def animate(i):
        im.set_array(frames_v[i])
        ax.set_title(f'шаг {i*5}')
        return [im]

    anim = animation.FuncAnimation(fig, animate, frames=len(frames_v), interval=50)
    anim.save("flow_video.mp4", fps=20, extra_args=['-vcodec', 'libx264'])
    print("сохранил графики и видео")
    plt.close(fig)

if __name__ == "__main__":
    main()