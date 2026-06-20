"""
掩码分布统计绘图脚本
功能：根据输入的数据集名称和掩码路径，绘制掩码的位置分布（2D KDE）和尺寸分布（1D KDE）图。
输出：
  1. position.svg — 位置分布图，无任何文字（方便后续手动添加文字）
  2. position.png — 位置分布图，含完整文字
  3. size.svg — 尺寸分布图，无任何文字
  4. size.png — 尺寸分布图，含完整文字
所有文件保存在以数据集名称命名的目录中。
"""

import os
import glob
import numpy as np
from PIL import Image
from scipy import ndimage
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from scipy.stats import gaussian_kde
from tqdm import tqdm


# ======================== 用户输入区域 ========================
# 数据集名称列表（与 mask_dirs 一一对应）
dataset_names = [
    "Augtrain",
    "FinalData",
    "TN3K",
    "TN5K",
    "PKTN",
    "ThyroidXL",
    "DDTI",
]

# 掩码文件夹路径列表（支持 png, jpg, bmp, tif 等常见格式）
mask_dirs = [
    "/mnt/wangbd8/workspace/DataSets/ThyroidAgent/augtrain_PNG/mask",
    "/mnt/wangbd8/workspace/DataSets/ThyroidAgent/train_val_test/finall_data/mask",
    "/mnt/wangbd8/workspace/DataSets/ThyroidAgent/train_val_test/TN3K/train/masks",
    "/mnt/wangbd8/workspace/DataSets/ThyroidAgent/train_val_test/TN5K/train/masks",
    "/mnt/wangbd8/workspace/DataSets/ThyroidAgent/train_val_test/PKTN/train/masks",
    "/mnt/wangbd8/workspace/DataSets/ThyroidAgent/train_val_test/ThyroidXL/train/masks",
    "/mnt/wangbd8/workspace/DataSets/ThyroidAgent/train_val_test/DDTI/train/masks",
]

# 输出根目录（默认为当前脚本所在目录）
output_root = os.path.dirname(os.path.abspath(__file__))

# ======================== 配置区域 ========================
# 字号统一设置
TICK_LABEL_FONTSIZE = 18       # XY轴刻度数值字号
AXIS_LABEL_FONTSIZE = 11       # 轴标签字号
TITLE_FONTSIZE = 12            # 标题字号

# 图像尺寸
figsize_position = (5, 5)
figsize_size = (5, 4.5)

# KDE 位置图的网格分辨率
kde_grid_resolution = 100

# KDE 尺寸图的点数
kde_size_points = 500

# 位置图颜色映射：白色 -> 浅蓝 -> 深蓝
position_colors = ['#FFFFFF', '#C6DBEF', '#6BAED6', '#2171B5', '#08306B', '#041E42']
position_cmap = LinearSegmentedColormap.from_list('position_kde', position_colors, N=256)

# 尺寸图颜色
size_color = '#D4726A'  # 玫瑰红/粉红色
size_edge_color = '#B5443B'  # 边缘线颜色

# 跨数据集共享绘图参数
shared_size_margin = 0.2
position_contour_levels = 15


def collect_mask_stats(mask_dir):
    """
    遍历掩码文件夹，统计每个掩码的质心位置和相对尺寸。
    
    返回:
        pos_x: 归一化的质心 x 坐标列表 (0~1)
        pos_y: 归一化的质心 y 坐标列表 (0~1)
        rel_sizes: 相对尺寸列表 (掩码面积 / 图像总面积)
    """
    # 支持的图像格式
    extensions = ['*.png', '*.jpg', '*.jpeg', '*.bmp', '*.tif', '*.tiff']
    mask_files = []
    for ext in extensions:
        mask_files.extend(glob.glob(os.path.join(mask_dir, ext)))
        mask_files.extend(glob.glob(os.path.join(mask_dir, ext.upper())))
    
    # 去重并排序
    mask_files = sorted(list(set(mask_files)))
    
    if len(mask_files) == 0:
        raise FileNotFoundError(f"在 {mask_dir} 中未找到任何掩码文件")
    
    pos_x_list = []
    pos_y_list = []
    rel_size_list = []
    
    print(f"正在处理 {len(mask_files)} 个掩码文件...")
    
    for mask_path in tqdm(mask_files, desc="统计掩码"):
        # 读取掩码（灰度）
        mask = np.array(Image.open(mask_path).convert('L'))
        h, w = mask.shape
        total_pixels = h * w
        
        # 二值化（阈值 128）
        binary_mask = (mask > 128).astype(np.uint8)
        
        # 计算掩码面积
        mask_area = binary_mask.sum()
        
        if mask_area == 0:
            # 跳过空掩码
            continue
        
        # 计算质心
        cy, cx = ndimage.center_of_mass(binary_mask)
        
        # 归一化质心坐标到 [0, 1]
        norm_cx = cx / w
        norm_cy = cy / h
        
        # 相对尺寸
        rel_size = mask_area / total_pixels
        
        pos_x_list.append(norm_cx)
        pos_y_list.append(norm_cy)
        rel_size_list.append(rel_size)
    
    print(f"有效掩码数量: {len(pos_x_list)}")
    
    return np.array(pos_x_list), np.array(pos_y_list), np.array(rel_size_list)


def compute_position_density(pos_x, pos_y):
    """
    计算位置分布的二维 KDE 密度图，供所有数据集复用统一网格。
    """
    xy = np.vstack([pos_x, pos_y])
    kde = gaussian_kde(xy)

    x_grid = np.linspace(0, 1, kde_grid_resolution)
    y_grid = np.linspace(0, 1, kde_grid_resolution)
    X, Y = np.meshgrid(x_grid, y_grid)
    positions = np.vstack([X.ravel(), Y.ravel()])
    Z = kde(positions).reshape(X.shape)
    return X, Y, Z


def compute_shared_size_xlim(dataset_stats, margin=shared_size_margin):
    """
    基于全部数据集的相对尺寸最大值，计算统一的横轴上限。
    """
    valid_maxima = [
        float(np.max(stats["rel_sizes"]))
        for stats in dataset_stats
        if len(stats["rel_sizes"]) > 0
    ]
    if not valid_maxima:
        raise ValueError("没有可用于计算共享尺寸横轴的数据")

    return min(max(valid_maxima) * (1 + margin), 1.0)


def compute_shared_position_scale(density_maps, n_levels=position_contour_levels):
    """
    基于全部数据集的 KDE 密度图，生成统一的等高线分级和颜色范围。
    """
    if not density_maps:
        raise ValueError("没有可用于计算共享位置色阶的密度图")

    vmax = max(float(np.max(density_map)) for density_map in density_maps)
    if vmax <= 0:
        vmax = 1.0

    levels = np.linspace(0, vmax, n_levels)
    return levels, vmax


def plot_position(pos_x, pos_y, dataset_name, n_samples, output_dir, save_svg=True, save_png=True,
                  position_grid=None, shared_levels=None, shared_vmax=None):
    """
    绘制位置分布 2D KDE 图。
    """
    X, Y, Z = position_grid if position_grid is not None else compute_position_density(pos_x, pos_y)

    contour_levels = shared_levels if shared_levels is not None else position_contour_levels
    contour_kwargs = {"levels": contour_levels, "cmap": position_cmap}
    if shared_vmax is not None:
        contour_kwargs["norm"] = Normalize(vmin=0, vmax=shared_vmax)

    # ==================== SVG（无文字）====================
    if save_svg:
        fig, ax = plt.subplots(figsize=figsize_position)

        # 绘制填充等高线图
        ax.contourf(X, Y, Z, **contour_kwargs)

        # 设置范围
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect('equal')

        # 隐藏所有文字
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_xlabel('')
        ax.set_ylabel('')
        ax.set_title('')

        # 保留刻度线但隐藏标签
        ax.tick_params(axis='both', which='both', length=4, labelsize=0)

        plt.tight_layout()
        svg_path = os.path.join(output_dir, "position.svg")
        fig.savefig(svg_path, format='svg', bbox_inches='tight', dpi=150)
        plt.close(fig)
        print(f"位置分布 SVG (无文字) 已保存: {svg_path}")

    # ==================== SVG（仅刻度值）====================
    if save_svg:
        fig, ax = plt.subplots(figsize=figsize_position)
        fig.patch.set_alpha(0)  # 图形背景透明
        ax.set_facecolor('none')  # 绘图区背景透明

        # 绘制填充等高线图
        ax.contourf(X, Y, Z, **contour_kwargs)

        # 设置范围
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect('equal')

        # 设置刻度值（只保留刻度数字，不加标题和轴标签）
        ax.set_xticks(np.arange(0, 1.1, 0.2))
        ax.set_yticks(np.arange(0, 1.1, 0.2))
        ax.set_xlabel('')
        ax.set_ylabel('')
        ax.set_title('')
        ax.tick_params(axis='both', labelsize=TICK_LABEL_FONTSIZE)

        plt.tight_layout()
        svg_path = os.path.join(output_dir, "position_with_ticks.svg")
        fig.savefig(svg_path, format='svg', bbox_inches='tight', dpi=150, transparent=True)
        plt.close(fig)
        print(f"位置分布 SVG (仅刻度值) 已保存: {svg_path}")

    # ==================== PNG（完整文字）====================
    if save_png:
        fig, ax = plt.subplots(figsize=figsize_position)

        # 绘制填充等高线图
        ax.contourf(X, Y, Z, **contour_kwargs)

        # 设置范围
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect('equal')

        # 添加标签
        ax.set_xlabel('pos_x', fontsize=AXIS_LABEL_FONTSIZE)
        ax.set_ylabel('pos_y', fontsize=AXIS_LABEL_FONTSIZE)
        ax.set_title(f'{dataset_name} - Position (N={n_samples})', fontsize=TITLE_FONTSIZE)

        # 设置刻度
        ax.set_xticks(np.arange(0, 1.1, 0.2))
        ax.set_yticks(np.arange(0, 1.1, 0.2))
        ax.tick_params(axis='both', labelsize=TICK_LABEL_FONTSIZE)

        plt.tight_layout()
        png_path = os.path.join(output_dir, "position.png")
        fig.savefig(png_path, format='png', bbox_inches='tight', dpi=150)
        plt.close(fig)
        print(f"位置分布 PNG (完整) 已保存: {png_path}")


def plot_size(rel_sizes, dataset_name, n_samples, output_dir, save_svg=True, save_png=True,
              shared_xlim=None):
    """
    绘制尺寸分布 1D KDE 图。
    """
    # 计算 1D KDE
    kde = gaussian_kde(rel_sizes)
    x_max = shared_xlim if shared_xlim is not None else min(rel_sizes.max() * (1 + shared_size_margin), 1.0)
    x_range = np.linspace(0, x_max, kde_size_points)
    density = kde(x_range)

    # ==================== SVG（无文字）====================
    if save_svg:
        fig, ax = plt.subplots(figsize=figsize_size)

        # 绘制 KDE 曲线和填充
        ax.plot(x_range, density, color=size_edge_color, linewidth=1.5)
        ax.fill_between(x_range, density, alpha=0.4, color=size_color)

        # 设置范围
        ax.set_xlim(0, x_max)
        ax.set_ylim(0, None)

        # 隐藏所有文字
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_xlabel('')
        ax.set_ylabel('')
        ax.set_title('')

        # 保留刻度线但隐藏标签
        ax.tick_params(axis='both', which='both', length=4, labelsize=0)

        plt.tight_layout()
        svg_path = os.path.join(output_dir, "size.svg")
        fig.savefig(svg_path, format='svg', bbox_inches='tight', dpi=150)
        plt.close(fig)
        print(f"尺寸分布 SVG (无文字) 已保存: {svg_path}")

    # ==================== SVG（仅刻度值）====================
    if save_svg:
        fig, ax = plt.subplots(figsize=figsize_size)
        fig.patch.set_alpha(0)  # 图形背景透明
        ax.set_facecolor('none')  # 绘图区背景透明

        # 绘制 KDE 曲线和填充
        ax.plot(x_range, density, color=size_edge_color, linewidth=1.5)
        ax.fill_between(x_range, density, alpha=0.4, color=size_color)

        # 设置范围
        ax.set_xlim(0, x_max)
        ax.set_ylim(0, None)

        # 设置刻度值（只保留刻度数字，不加标题和轴标签）
        ax.set_xlabel('')
        ax.set_ylabel('')
        ax.set_title('')
        ax.tick_params(axis='both', labelsize=TICK_LABEL_FONTSIZE)

        plt.tight_layout()
        svg_path = os.path.join(output_dir, "size_with_ticks.svg")
        fig.savefig(svg_path, format='svg', bbox_inches='tight', dpi=150, transparent=True)
        plt.close(fig)
        print(f"尺寸分布 SVG (仅刻度值) 已保存: {svg_path}")

    # ==================== PNG（完整文字）====================
    if save_png:
        fig, ax = plt.subplots(figsize=figsize_size)

        # 绘制 KDE 曲线和填充
        ax.plot(x_range, density, color=size_edge_color, linewidth=1.5)
        ax.fill_between(x_range, density, alpha=0.4, color=size_color)

        # 设置范围
        ax.set_xlim(0, x_max)
        ax.set_ylim(0, None)

        # 添加标签
        ax.set_xlabel('relative size', fontsize=AXIS_LABEL_FONTSIZE)
        ax.set_ylabel('density', fontsize=AXIS_LABEL_FONTSIZE)
        ax.set_title(f'{dataset_name} - Size (N={n_samples})', fontsize=TITLE_FONTSIZE)

        # 设置刻度
        ax.tick_params(axis='both', labelsize=TICK_LABEL_FONTSIZE)

        plt.tight_layout()
        png_path = os.path.join(output_dir, "size.png")
        fig.savefig(png_path, format='png', bbox_inches='tight', dpi=150)
        plt.close(fig)
        print(f"尺寸分布 PNG (完整) 已保存: {png_path}")


def main():
    """主函数"""
    # 检查数据集名称和路径数量是否一致
    assert len(dataset_names) == len(mask_dirs), \
        f"dataset_names ({len(dataset_names)}) 和 mask_dirs ({len(mask_dirs)}) 数量不一致！"

    print(f"共 {len(dataset_names)} 个数据集待处理")
    print("=" * 60)

    dataset_stats = []

    for idx, (dataset_name, mask_dir) in enumerate(zip(dataset_names, mask_dirs)):
        print(f"\n[{idx + 1}/{len(dataset_names)}] 处理数据集: {dataset_name}")
        print(f"掩码目录: {mask_dir}")

        # 创建输出目录
        output_dir = os.path.join(output_root, dataset_name)
        os.makedirs(output_dir, exist_ok=True)
        print(f"输出目录: {output_dir}")
        print("-" * 50)

        # 统计掩码信息
        try:
            pos_x, pos_y, rel_sizes = collect_mask_stats(mask_dir)
        except FileNotFoundError as e:
            print(f"跳过: {e}")
            continue

        n_samples = len(pos_x)

        if n_samples == 0:
            print("警告：没有有效的掩码数据，跳过该数据集！")
            continue

        dataset_stats.append({
            "dataset_name": dataset_name,
            "output_dir": output_dir,
            "pos_x": pos_x,
            "pos_y": pos_y,
            "rel_sizes": rel_sizes,
            "n_samples": n_samples,
        })

    if not dataset_stats:
        print("\n没有可绘制的有效数据集，程序结束。")
        return

    for stats in dataset_stats:
        stats["position_grid"] = compute_position_density(stats["pos_x"], stats["pos_y"])

    shared_size_xlim = compute_shared_size_xlim(dataset_stats)
    shared_position_levels, shared_position_vmax = compute_shared_position_scale(
        [stats["position_grid"][2] for stats in dataset_stats]
    )

    print("\n共享绘图参数：")
    print(f"尺寸分布统一横轴上限: {shared_size_xlim:.4f}")
    print(f"位置分布统一密度上限: {shared_position_vmax:.6f}")
    print("=" * 60)

    for idx, stats in enumerate(dataset_stats, start=1):
        dataset_name = stats["dataset_name"]
        output_dir = stats["output_dir"]
        pos_x = stats["pos_x"]
        pos_y = stats["pos_y"]
        rel_sizes = stats["rel_sizes"]
        n_samples = stats["n_samples"]

        print(f"\n[{idx}/{len(dataset_stats)}] 输出数据集: {dataset_name}")
        print(f"统计完成，共 {n_samples} 个有效掩码")
        print(f"位置 X: mean={pos_x.mean():.3f}, std={pos_x.std():.3f}")
        print(f"位置 Y: mean={pos_y.mean():.3f}, std={pos_y.std():.3f}")
        print(f"相对尺寸: mean={rel_sizes.mean():.4f}, std={rel_sizes.std():.4f}, "
              f"min={rel_sizes.min():.4f}, max={rel_sizes.max():.4f}")
        print("-" * 50)

        # 绘制位置分布图
        print("\n绘制位置分布图...")
        plot_position(
            pos_x,
            pos_y,
            dataset_name,
            n_samples,
            output_dir,
            position_grid=stats["position_grid"],
            shared_levels=shared_position_levels,
            shared_vmax=shared_position_vmax,
        )

        # 绘制尺寸分布图
        print("\n绘制尺寸分布图...")
        plot_size(
            rel_sizes,
            dataset_name,
            n_samples,
            output_dir,
            shared_xlim=shared_size_xlim,
        )

        print(f"\n数据集 [{dataset_name}] 完成！输出文件：")
        print(f"  - {os.path.join(output_dir, 'position.svg')} (无文字)")
        print(f"  - {os.path.join(output_dir, 'position_with_ticks.svg')} (仅刻度值)")
        print(f"  - {os.path.join(output_dir, 'position.png')} (完整)")
        print(f"  - {os.path.join(output_dir, 'size.svg')} (无文字)")
        print(f"  - {os.path.join(output_dir, 'size_with_ticks.svg')} (仅刻度值)")
        print(f"  - {os.path.join(output_dir, 'size.png')} (完整)")

    print("\n" + "=" * 60)
    print(f"全部 {len(dataset_stats)} 个有效数据集处理完成！")


if __name__ == "__main__":
    main()
