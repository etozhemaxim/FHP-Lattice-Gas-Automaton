import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.animation as animation
from matplotlib.animation import PillowWriter
import matplotlib.path as mpath
import random
from tqdm import tqdm
import scipy.ndimage as ndimage
import os

# ==========================================
# НАСТРОЙКИ
# ==========================================
ROWS = 100
COLS = 100
STEPS = 300
OBSTACLE_SIZE = 10
PROFILE_FILE = r"C:\LABA_MODEL\FHP-Lattice-Gas-Automaton\ls417.txt"

# Выберите форму: 'circle', 'square', 'triangle_right', 'triangle_left', 'airfoil'
SHAPE = 'circle' 

DIRECTIONS = np.array([
    [0, 1], [-1, 1], [-1, -1], [0, -1], [1, -1], [1, 1]
])
NEIGHBORS_DELTAS = [(0, 1), (-1, 1), (-1, -1), (0, -1), (1, -1), (1, 1)]

def initialize_grid():
    ni = np.zeros((ROWS, COLS, 6), dtype=np.int8)
    walls = np.zeros((ROWS, COLS), dtype=bool)
    return ni, walls

# ==========================================
# ЛОГИКА ФОРМ
# ==========================================
def add_obstacle(walls, center_r, center_c, size, shape_type):
    r, c = np.ogrid[:ROWS, :COLS]
    
    if shape_type == 'circle':
        mask = (r - center_r)**2 + (c - center_c)**2 <= size**2
    elif shape_type == 'square':
        mask = (np.abs(r - center_r) <= size) & (np.abs(c - center_c) <= size)
    elif shape_type == 'triangle_right':
        dr = r - center_r
        dc = c - center_c
        mask = (dc >= -size) & (dc <= size) & (dr >= -size) & (dr <= size) & (np.abs(dr) <= (size - dc) / 2)
    elif shape_type == 'triangle_left':
        dr = r - center_r
        dc = c - center_c
        mask = (dc >= -size) & (dc <= size) & (dr >= -size) & (dr <= size) & (np.abs(dr) <= (size + dc) / 2)
    elif shape_type == 'airfoil':
        if not os.path.exists(PROFILE_FILE):
            print(f"Файл {PROFILE_FILE} не найден, использую круг.")
            mask = (r - center_r)**2 + (c - center_c)**2 <= size**2
        else:
            coords = np.loadtxt(PROFILE_FILE, skiprows=1)
            x = coords[:, 0]
            y = coords[:, 1]
            scale = 60
            offset_x = COLS // 4
            offset_y = ROWS // 2
            gx = (x * scale + offset_x).astype(int)
            gy = (offset_y - y * scale).astype(int)
            pts = np.column_stack((gx, gy))
            path = mpath.Path(pts)
            c_grid, r_grid = np.meshgrid(np.arange(COLS), np.arange(ROWS))
            grid_points = np.column_stack((c_grid.ravel(), r_grid.ravel()))
            inside = path.contains_points(grid_points)
            mask = inside.reshape(ROWS, COLS)
    else:
        mask = (r - center_r)**2 + (c - center_c)**2 <= size**2

    walls[mask] = True
    return walls

def get_patch(shape_type, center_r, center_c, size):
    """Возвращает объект фигуры для отрисовки поверх графиков"""
    if shape_type == 'circle':
        return patches.Circle((center_c, center_r), radius=size, facecolor='none', edgecolor='purple', linewidth=3)
    elif shape_type == 'square':
        return patches.Rectangle((center_c - size, center_r - size), 2*size, 2*size, facecolor='none', edgecolor='purple', linewidth=3)
    elif shape_type == 'triangle_right':
        return patches.Polygon([(center_c - size, center_r - size), (center_c - size, center_r + size), (center_c + size, center_r)], closed=True, facecolor='none', edgecolor='purple', linewidth=3)
    elif shape_type == 'triangle_left':
        return patches.Polygon([(center_c + size, center_r - size), (center_c + size, center_r + size), (center_c - size, center_r)], closed=True, facecolor='none', edgecolor='purple', linewidth=3)
    else:
        # Для профиля крыла мы используем contour, поэтому патч не нужен
        return None

# ==========================================
# ФИЗИКА
# ==========================================
def create_inflow(ni, walls, density=0.5):
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
        shifted_ni = np.roll(ni[:,:,i], shift=(dr, dc), axis=(0, 1))
        bounce_ni = np.roll(shifted_ni, shift=(-dr, -dc), axis=(0, 1)) * walls
        new_ni[:,:,i] = np.where(walls, bounce_ni, shifted_ni)
        new_ni[:,:,(i+3)%6] += np.where(walls, bounce_ni, 0)
    new_ni = np.clip(new_ni, 0, 1)
    new_ni[:, -1, :] = 0 
    return new_ni

def calculate_macroscopic(ni):
    rho = np.sum(ni, axis=2).astype(float)
    v_row = np.sum(ni * DIRECTIONS[:, 0], axis=2).astype(float)
    v_col = np.sum(ni * DIRECTIONS[:, 1], axis=2).astype(float)
    with np.errstate(divide='ignore', invalid='ignore'):
        v_row_norm = np.divide(v_row, rho, out=np.zeros_like(v_row), where=rho!=0)
        v_col_norm = np.divide(v_col, rho, out=np.zeros_like(v_col), where=rho!=0)
    v_mag = np.sqrt(v_row_norm**2 + v_col_norm**2)
    return ndimage.gaussian_filter(rho, sigma=2), ndimage.gaussian_filter(v_mag, sigma=2)

# ==========================================
# ЗАПУСК И СОХРАНЕНИЕ
# ==========================================
def main(shape):
    print(f"\n--- Запуск симуляции для формы: {shape} ---")
    ni, walls = initialize_grid()
    center_r, center_c = ROWS//2, COLS//2
    walls = add_obstacle(walls, center_r, center_c, OBSTACLE_SIZE, shape)
    
    ni = create_inflow(ni, walls, density=0.8)
    for r in range(ROWS):
        for c in range(5):
            if not walls[r, c]:
                ni[r, c, 0] = 1

    frames_v_mag = []
    frames_rho = []
    
    for step in tqdm(range(STEPS)):
        ni = create_inflow(ni, walls, density=0.3)
        ni = update(ni, walls)
        if step % 5 == 0:
            rho, v_mag = calculate_macroscopic(ni)
            frames_v_mag.append(v_mag.copy())
            frames_rho.append(rho.copy())

    # --- СТАТИЧНЫЕ ГРАФИКИ ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    im1 = axes[0].imshow(frames_rho[-1], cmap='Blues', origin='upper')
    fig.colorbar(im1, ax=axes[0], label='Плотность')
    axes[0].set_title(f'Плотность ({shape})')
    
    im2 = axes[1].imshow(frames_v_mag[-1], cmap='inferno', origin='upper', vmin=0, vmax=1.5)
    fig.colorbar(im2, ax=axes[1], label='Скорость')
    axes[1].set_title(f'Скорость ({shape})')
    
    patch_obj = get_patch(shape, center_r, center_c, OBSTACLE_SIZE)
    if patch_obj:
        axes[0].add_patch(patch_obj)
        axes[1].add_patch(patches.Circle((center_c, center_r), radius=OBSTACLE_SIZE, facecolor='none', edgecolor='purple', linewidth=3) if shape != 'airfoil' else None)
        # Для профиля используем контур
        if shape == 'airfoil':
            axes[0].contour(walls, levels=[0.5], colors='purple', linewidths=3)
            axes[1].contour(walls, levels=[0.5], colors='purple', linewidths=3)
    
    plt.tight_layout()
    plt.savefig(f'flow_results_{shape}.png', dpi=300)
    plt.close(fig)

    # --- MP4 ---
    fig_mp4, ax_mp4 = plt.subplots(figsize=(8, 8))
    im_mp4 = ax_mp4.imshow(frames_v_mag[0], cmap='inferno', origin='upper', vmin=0, vmax=1.5)
    ax_mp4.set_title(f'Обтекание ({shape})')
    fig_mp4.colorbar(im_mp4, ax=ax_mp4, label='Скорость')
    
    patch_obj_mp4 = get_patch(shape, center_r, center_c, OBSTACLE_SIZE)
    if patch_obj_mp4:
        ax_mp4.add_patch(patch_obj_mp4)
    if shape == 'airfoil':
        ax_mp4.contour(walls, levels=[0.5], colors='purple', linewidths=3)
    
    def animate_func(i):
        im_mp4.set_array(frames_v_mag[i])
        ax_mp4.set_title(f'Обтекание (шаг {i*5})')
        return [im_mp4]

    anim = animation.FuncAnimation(fig_mp4, animate_func, frames=len(frames_v_mag), interval=50)
    anim.save(f"flow_video_{shape}.mp4", fps=20, extra_args=['-vcodec', 'libx264'])
    plt.close(fig_mp4)

    # --- GIF ---
    fig_gif, ax_gif = plt.subplots(figsize=(8, 8))
    im_gif = ax_gif.imshow(frames_v_mag[0], cmap='inferno', origin='upper', vmin=0, vmax=1.5, interpolation='bicubic')
    ax_gif.set_title(f'Обтекание ({shape})')
    fig_gif.colorbar(im_gif, ax=ax_gif, label='Скорость')
    
    patch_obj_gif = get_patch(shape, center_r, center_c, OBSTACLE_SIZE)
    if patch_obj_gif:
        ax_gif.add_patch(patch_obj_gif)
    if shape == 'airfoil':
        ax_gif.contour(walls, levels=[0.5], colors='purple', linewidths=3)
    
    def animate_gif(i):
        im_gif.set_array(frames_v_mag[i])
        ax_gif.set_title(f'Обтекание (шаг {i*5})')
        return [im_gif]

    anim_gif = animation.FuncAnimation(fig_gif, animate_gif, frames=len(frames_v_mag), interval=50)
    writer = PillowWriter(fps=15)
    anim_gif.save(f"flow_animation_{shape}.gif", writer=writer, dpi=120)
    plt.close(fig_gif)
    
    print(f"Сохранено для {shape}: png, mp4, gif")

if __name__ == "__main__":
    # Здесь можно перечислить формы, для которых нужно запустить симуляцию
    shapes_to_run = ['airfoil']
    
    for shape in shapes_to_run:
        main(shape)