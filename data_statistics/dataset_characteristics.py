"""
数据集特征分别绘图脚本
功能：分别输出每个子图（PNG 含文字 + SVG 无文字），共 11 个子图。

子图清单：
  (a)  各数据集良/恶性数量柱状图（对数 y 轴）          ×1
  (b)  各数据集病灶位置 2D KDE                          ×5
  (c)  各数据集病灶相对尺寸 1D KDE                       ×5
  合计 11 个子图，每个输出 .png 和 .svg

输出目录：脚本同级 panels/ 文件夹
"""

import os
import glob
import numpy as np
from PIL import Image
from scipy import ndimage
from scipy.stats import gaussian_kde
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.ticker import MaxNLocator
from tqdm import tqdm


# ======================== 用户输入区域 ========================
# 图中展示的数据集名称（需同时有类别计数和掩码数据）
dataset_names = ["TN3K", "TN5K", "ThyroidXL", "DDTI", "ZJH-8K"]

# 各数据集良/恶性数量 [benign, malignant]
class_counts = {
    "TN3K":      [2709, 2638],
    "TN5K":      [1426, 3574],
    "ThyroidXL": [8172, 3459],
    "DDTI":      [304, 45],
    "ZJH-8K":    [3202, 4756],
}

# 掩码文件夹路径
mask_dirs = {
    "TN3K":      "/mnt/wangbd8/workspace/DataSets/ThyroidAgent/train_val_test/TN3K/train/masks",
    "TN5K":      "/mnt/wangbd8/workspace/DataSets/ThyroidAgent/train_val_test/TN5K/train/masks",
    "ThyroidXL": "/mnt/wangbd8/workspace/DataSets/ThyroidAgent/train_val_test/ThyroidXL/train/masks",
    "DDTI":      "/mnt/wangbd8/workspace/DataSets/ThyroidAgent/train_val_test/DDTI/train/masks",
    "ZJH-8K":    "/mnt/wangbd8/workspace/DataSets/ThyroidAgent/train_val_test/ZJH-8K/masks",
    "RJH-7K":    "/mnt/wangbd8/workspace/DataSets/ThyroidAgent/augtrain_PNG/mask",
}

# 类别名称
class_names = ["Benign", "Malignant"]

# 输出目录
output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "panels")
os.makedirs(output_dir, exist_ok=True)

# ======================== 配置区域 ========================
# 字号
FONTSIZE_BAR_TICK = 9
FONTSIZE_BAR_LABEL = 10
FONTSIZE_BAR_ANNOT = 7
FONTSIZE_BAR_TITLE = 11
FONTSIZE_PANEL_TITLE = 9
FONTSIZE_PANEL_TICK = 7
FONTSIZE_PANEL_LABEL = 8

# 配色
bar_colors = ["#1B4F72", "#C6975B"]  # Benign, Malignant

position_cmap = LinearSegmentedColormap.from_list(
    'position_kde',
    ['#FFFFFF', '#C6DBEF', '#6BAED6', '#2171B5', '#08306B', '#041E42'],
    N=256,
)
size_color = '#D4726A'
size_edge_color = '#B5443B'

# 柱状图参数
bar_width = 0.35

# KDE 参数
kde_grid_resolution = 100
kde_size_points = 500
shared_size_margin = 0.2
position_contour_levels = 15

# 各子图尺寸（英寸）
FIGSIZE_BAR = (8.0, 4.5)
FIGSIZE_POSITION = (4.5, 4.5)
FIGSIZE_SIZE = (4.5, 3.0)


# ======================== 掩码统计 ========================
def collect_mask_stats(mask_dir):
    """
    遍历掩码文件夹，统计每个掩码的质心位置和相对尺寸。

    返回:
        pos_x:     归一化质心 x 坐标 (0~1)
        pos_y:     归一化质心 y 坐标 (0~1)
        rel_sizes: 相对尺寸 (掩码面积 / 图像总面积)
    """
    extensions = ['*.png', '*.jpg', '*.jpeg', '*.bmp', '*.tif', '*.tiff']
    mask_files = []
    for ext in extensions:
        mask_files.extend(glob.glob(os.path.join(mask_dir, ext)))
        mask_files.extend(glob.glob(os.path.join(mask_dir, ext.upper())))
    mask_files = sorted(list(set(mask_files)))

    if len(mask_files) == 0:
        raise FileNotFoundError(f"在 {mask_dir} 中未找到任何掩码文件")

    pos_x_list, pos_y_list, rel_size_list = [], [], []

    print(f"  正在处理 {len(mask_files)} 个掩码文件...")
    for mask_path in tqdm(mask_files, desc="  掩码", leave=False):
        mask = np.array(Image.open(mask_path).convert('L'))
        h, w = mask.shape
        binary_mask = (mask > 128).astype(np.uint8)
        mask_area = binary_mask.sum()
        if mask_area == 0:
            continue
        cy, cx = ndimage.center_of_mass(binary_mask)
        pos_x_list.append(cx / w)
        pos_y_list.append(cy / h)
        rel_size_list.append(mask_area / (h * w))

    return np.array(pos_x_list), np.array(pos_y_list), np.array(rel_size_list)


def compute_position_density(pos_x, pos_y):
    """计算位置分布的二维 KDE。"""
    xy = np.vstack([pos_x, pos_y])
    kde = gaussian_kde(xy)
    x_grid = np.linspace(0, 1, kde_grid_resolution)
    y_grid = np.linspace(0, 1, kde_grid_resolution)
    X, Y = np.meshgrid(x_grid, y_grid)
    Z = kde(np.vstack([X.ravel(), Y.ravel()])).reshape(X.shape)
    return X, Y, Z


def compute_shared_size_xlim(all_rel_sizes, margin=shared_size_margin):
    """基于全部数据集计算统一的尺寸横轴上限。"""
    valid_maxima = [float(np.max(s)) for s in all_rel_sizes if len(s) > 0]
    if not valid_maxima:
        return 1.0
    return min(max(valid_maxima) * (1 + margin), 1.0)


def compute_shared_position_scale(density_maps, n_levels=position_contour_levels):
    """基于全部数据集的 KDE 密度图生成统一的等高线分级和颜色范围。"""
    vmax = max(float(np.max(z)) for _, _, z in density_maps)
    if vmax <= 0:
        vmax = 1.0
    levels = np.linspace(0, vmax, n_levels)
    return levels, vmax


# ======================== 通用保存工具 ========================
def _hide_all_text(fig):
    """隐藏 figure 中所有文字（保留布局和图形元素）。"""
    for ax in fig.get_axes():
        ax.set_title('')
        ax.set_xlabel('')
        ax.set_ylabel('')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.tick_params(axis='both', which='both', length=3, labelsize=0)
        for text in ax.texts:
            text.set_visible(False)
        leg = ax.get_legend()
        if leg:
            for t in leg.get_texts():
                t.set_color('none')


def save_panel(fig, panel_name, save_png=True, save_svg=True):
    """保存单个子图：PNG（含文字）+ SVG（无文字）。"""
    fig.tight_layout()

    if save_png:
        png_path = os.path.join(output_dir, f"{panel_name}.png")
        fig.savefig(png_path, format='png', bbox_inches='tight', dpi=200)
        print(f"  PNG: {png_path}")

    if save_svg:
        _hide_all_text(fig)
        svg_path = os.path.join(output_dir, f"{panel_name}.svg")
        fig.savefig(svg_path, format='svg', bbox_inches='tight', dpi=150)
        print(f"  SVG: {svg_path}")

    plt.close(fig)


# ======================== 子图绘图函数 ========================
def plot_bar_panel(save_png=True, save_svg=True):
    """(a) 各数据集良/恶性数量柱状图。"""
    n = len(dataset_names)
    x = np.arange(n)
    benign = [class_counts[name][0] for name in dataset_names]
    malignant = [class_counts[name][1] for name in dataset_names]

    fig, ax = plt.subplots(figsize=FIGSIZE_BAR)

    bars1 = ax.bar(x - bar_width / 2, benign, bar_width,
                   color=bar_colors[0], edgecolor='none', label=class_names[0])
    bars2 = ax.bar(x + bar_width / 2, malignant, bar_width,
                   color=bar_colors[1], edgecolor='none', label=class_names[1])

    ax.set_yscale('log')
    ax.set_xticks(x)
    ax.set_xticklabels(dataset_names, fontsize=FONTSIZE_BAR_TICK)
    ax.set_ylabel('Number of images', fontsize=FONTSIZE_BAR_LABEL)
    ax.set_title('Class distributions across thyroid ultrasound datasets',
                 fontsize=FONTSIZE_BAR_TITLE, fontweight='bold', pad=8)

    all_vals = benign + malignant
    ax.set_ylim(min(all_vals) * 0.5, max(all_vals) * 3)

    for bars in [bars1, bars2]:
        for bar in bars:
            h_bar = bar.get_height()
            ax.annotate(f'{h_bar:,.0f}',
                        xy=(bar.get_x() + bar.get_width() / 2, h_bar),
                        xytext=(0, 2), textcoords="offset points",
                        ha='center', va='bottom', fontsize=FONTSIZE_BAR_ANNOT)

    ax.legend(loc='upper left', fontsize=FONTSIZE_BAR_TICK, framealpha=0.9)
    ax.tick_params(axis='y', labelsize=FONTSIZE_BAR_TICK)

    save_panel(fig, "panel_a_class_distribution", save_png, save_svg)


def plot_position_panel(stats, shared_levels, shared_vmax,
                        save_png=True, save_svg=True):
    """(b) 单个数据集的病灶位置 2D KDE。"""
    name = stats['dataset_name']
    n_samples = stats['n_samples']

    fig, ax = plt.subplots(figsize=FIGSIZE_POSITION)

    if stats['position_grid'] is not None and shared_levels is not None:
        X, Y, Z = stats['position_grid']
        ax.contourf(X, Y, Z, levels=shared_levels, cmap=position_cmap,
                    norm=Normalize(vmin=0, vmax=shared_vmax))

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect('equal')
    # 超声图像 y 轴自上而下，翻转 y 轴使分布图方向与图像一致
    ax.invert_yaxis()

    title = f"{name}" if n_samples == 0 else f"{name} (N={n_samples:,})"
    ax.set_title(title, fontsize=FONTSIZE_PANEL_TITLE, pad=3)

    ax.set_xticks(np.arange(0, 1.1, 0.5))
    ax.set_yticks(np.arange(0, 1.1, 0.5))
    ax.tick_params(axis='both', labelsize=FONTSIZE_PANEL_TICK)
    ax.set_ylabel('pos_y', fontsize=FONTSIZE_PANEL_LABEL)
    ax.set_xlabel('pos_x', fontsize=FONTSIZE_PANEL_LABEL)

    save_panel(fig, f"panel_b_position_{name}", save_png, save_svg)


def plot_size_panel(stats, shared_size_xlim, save_png=True, save_svg=True):
    """(c) 单个数据集的病灶相对尺寸 1D KDE。"""
    name = stats['dataset_name']
    n_samples = stats['n_samples']
    rel_sizes = stats['rel_sizes']

    fig, ax = plt.subplots(figsize=FIGSIZE_SIZE)

    if n_samples < 2:
        ax.text(0.5, 0.5, 'N/A', ha='center', va='center',
                transform=ax.transAxes, fontsize=FONTSIZE_PANEL_TICK)
        ax.set_xlim(0, shared_size_xlim)
        ax.set_ylim(0, 1)
    else:
        kde = gaussian_kde(rel_sizes)
        x_range = np.linspace(0, shared_size_xlim, kde_size_points)
        density = kde(x_range)

        ax.plot(x_range, density, color=size_edge_color, linewidth=1.2)
        ax.fill_between(x_range, density, alpha=0.4, color=size_color)
        ax.set_xlim(0, shared_size_xlim)
        ax.set_ylim(0, None)

    ax.tick_params(axis='both', labelsize=FONTSIZE_PANEL_TICK)
    ax.set_xlabel('relative size', fontsize=FONTSIZE_PANEL_LABEL)
    # 使用 MaxNLocator 自动生成简洁的 y 轴刻度
    ax.yaxis.set_major_locator(MaxNLocator(nbins=4, integer=False))
    ax.set_ylabel('density', fontsize=FONTSIZE_PANEL_LABEL)

    title = f"{name}" if n_samples == 0 else f"{name} (N={n_samples:,})"
    ax.set_title(title, fontsize=FONTSIZE_PANEL_TITLE, pad=3)

    save_panel(fig, f"panel_c_size_{name}", save_png, save_svg)


# ======================== 主函数 ========================
def main():
    print(f"共 {len(dataset_names)} 个数据集: {dataset_names}")
    print(f"输出目录: {output_dir}")
    print("=" * 60)

    # 收集掩码统计
    dataset_stats = []
    for name in dataset_names:
        mask_dir = mask_dirs.get(name)
        print(f"\n[{name}]")
        print(f"  掩码目录: {mask_dir}")

        if mask_dir is None:
            print("  跳过: 未配置掩码目录")
            dataset_stats.append({
                'dataset_name': name,
                'pos_x': np.array([]),
                'pos_y': np.array([]),
                'rel_sizes': np.array([]),
                'position_grid': None,
                'n_samples': 0,
            })
            continue

        try:
            pos_x, pos_y, rel_sizes = collect_mask_stats(mask_dir)
        except FileNotFoundError as e:
            print(f"  跳过: {e}")
            dataset_stats.append({
                'dataset_name': name,
                'pos_x': np.array([]),
                'pos_y': np.array([]),
                'rel_sizes': np.array([]),
                'position_grid': None,
                'n_samples': 0,
            })
            continue

        n_samples = len(pos_x)
        print(f"  有效掩码: {n_samples}")
        if n_samples > 0:
            print(f"  pos_x: mean={pos_x.mean():.3f}, std={pos_x.std():.3f}")
            print(f"  pos_y: mean={pos_y.mean():.3f}, std={pos_y.std():.3f}")
            print(f"  rel_size: mean={rel_sizes.mean():.4f}, std={rel_sizes.std():.4f}")

        dataset_stats.append({
            'dataset_name': name,
            'pos_x': pos_x,
            'pos_y': pos_y,
            'rel_sizes': rel_sizes,
            'position_grid': None,
            'n_samples': n_samples,
        })

    # 计算位置密度
    for stats in dataset_stats:
        if stats['n_samples'] > 0:
            stats['position_grid'] = compute_position_density(
                stats['pos_x'], stats['pos_y']
            )

    # 计算共享尺度
    valid_density = [s['position_grid'] for s in dataset_stats
                     if s['position_grid'] is not None]
    valid_sizes = [s['rel_sizes'] for s in dataset_stats
                   if len(s['rel_sizes']) > 0]

    if valid_density:
        shared_levels, shared_vmax = compute_shared_position_scale(valid_density)
    else:
        shared_levels, shared_vmax = None, None

    shared_size_xlim = compute_shared_size_xlim(valid_sizes) if valid_sizes else 1.0

    print("\n" + "=" * 60)
    print(f"共享尺寸横轴上限: {shared_size_xlim:.4f}")
    if shared_vmax:
        print(f"共享位置密度上限: {shared_vmax:.6f}")
    print("=" * 60)

    # ---- 分别绘制 11 个子图 ----
    print("\n正在绘制各子图...")

    # (a) 柱状图 ×1
    print("\n[panel a] 良/恶性数量柱状图")
    plot_bar_panel()

    # (b) 位置 KDE ×5
    print("\n[panel b] 病灶位置 2D KDE")
    for stats in dataset_stats:
        plot_position_panel(stats, shared_levels, shared_vmax)

    # (c) 尺寸 KDE ×5
    print("\n[panel c] 病灶相对尺寸 1D KDE")
    for stats in dataset_stats:
        plot_size_panel(stats, shared_size_xlim)

    print("\n" + "=" * 60)
    print(f"完成！共输出 11 个子图（每个含 .png 和 .svg），保存于：")
    print(f"  {output_dir}")


if __name__ == "__main__":
    main()
