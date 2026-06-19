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
from matplotlib.colors import LinearSegmentedColormap
from scipy.stats import gaussian_kde
from tqdm import tqdm


# ======================== 用户输入区域 ========================
# 数据集名称
dataset_name = "Augtrain"

# 掩码文件夹路径（支持 png, jpg, bmp, tif 等常见格式）
mask_dir = "/mnt/wangbd8/workspace/DataSets/ThyroidAgent/augtrain_PNG/mask"

# 输出根目录（默认为当前脚本所在目录）
output_root = os.path.dirname(os.path.abspath(__file__))

# ======================== 配置区域 ========================
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


def plot_position(pos_x, pos_y, dataset_name, n_samples, output_dir, save_svg=True, save_png=True):
    """
    绘制位置分布 2D KDE 图。
    """
    # 计算 2D KDE
    xy = np.vstack([pos_x, pos_y])
    kde = gaussian_kde(xy)
    
    # 创建网格
    x_grid = np.linspace(0, 1, kde_grid_resolution)
    y_grid = np.linspace(0, 1, kde_grid_resolution)
    X, Y = np.meshgrid(x_grid, y_grid)
    positions = np.vstack([X.ravel(), Y.ravel()])
    Z = kde(positions).reshape(X.shape)
    
    # ==================== SVG（无文字）====================
    if save_svg:
        fig, ax = plt.subplots(figsize=figsize_position)
        
        # 绘制填充等高线图
        ax.contourf(X, Y, Z, levels=15, cmap=position_cmap)
        
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
    
    # ==================== PNG（完整文字）====================
    if save_png:
        fig, ax = plt.subplots(figsize=figsize_position)
        
        # 绘制填充等高线图
        ax.contourf(X, Y, Z, levels=15, cmap=position_cmap)
        
        # 设置范围
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect('equal')
        
        # 添加标签
        ax.set_xlabel('pos_x', fontsize=11)
        ax.set_ylabel('pos_y', fontsize=11)
        ax.set_title(f'{dataset_name} - Position (N={n_samples})', fontsize=12)
        
        # 设置刻度
        ax.set_xticks(np.arange(0, 1.1, 0.2))
        ax.set_yticks(np.arange(0, 1.1, 0.2))
        ax.tick_params(axis='both', labelsize=10)
        
        plt.tight_layout()
        png_path = os.path.join(output_dir, "position.png")
        fig.savefig(png_path, format='png', bbox_inches='tight', dpi=150)
        plt.close(fig)
        print(f"位置分布 PNG (完整) 已保存: {png_path}")


def plot_size(rel_sizes, dataset_name, n_samples, output_dir, save_svg=True, save_png=True):
    """
    绘制尺寸分布 1D KDE 图。
    """
    # 计算 1D KDE
    kde = gaussian_kde(rel_sizes)
    x_range = np.linspace(0, min(rel_sizes.max() * 1.2, 1.0), kde_size_points)
    density = kde(x_range)
    
    # ==================== SVG（无文字）====================
    if save_svg:
        fig, ax = plt.subplots(figsize=figsize_size)
        
        # 绘制 KDE 曲线和填充
        ax.plot(x_range, density, color=size_edge_color, linewidth=1.5)
        ax.fill_between(x_range, density, alpha=0.4, color=size_color)
        
        # 设置范围
        ax.set_xlim(0, x_range[-1])
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
    
    # ==================== PNG（完整文字）====================
    if save_png:
        fig, ax = plt.subplots(figsize=figsize_size)
        
        # 绘制 KDE 曲线和填充
        ax.plot(x_range, density, color=size_edge_color, linewidth=1.5)
        ax.fill_between(x_range, density, alpha=0.4, color=size_color)
        
        # 设置范围
        ax.set_xlim(0, x_range[-1])
        ax.set_ylim(0, None)
        
        # 添加标签
        ax.set_xlabel('relative size', fontsize=11)
        ax.set_ylabel('density', fontsize=11)
        ax.set_title(f'{dataset_name} - Size (N={n_samples})', fontsize=12)
        
        # 设置刻度
        ax.tick_params(axis='both', labelsize=10)
        
        plt.tight_layout()
        png_path = os.path.join(output_dir, "size.png")
        fig.savefig(png_path, format='png', bbox_inches='tight', dpi=150)
        plt.close(fig)
        print(f"尺寸分布 PNG (完整) 已保存: {png_path}")


def main():
    """主函数"""
    # 创建输出目录
    output_dir = os.path.join(output_root, dataset_name)
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"数据集: {dataset_name}")
    print(f"掩码目录: {mask_dir}")
    print(f"输出目录: {output_dir}")
    print("-" * 50)
    
    # 统计掩码信息
    pos_x, pos_y, rel_sizes = collect_mask_stats(mask_dir)
    n_samples = len(pos_x)
    
    if n_samples == 0:
        print("错误：没有有效的掩码数据！")
        return
    
    print(f"\n统计完成，共 {n_samples} 个有效掩码")
    print(f"位置 X: mean={pos_x.mean():.3f}, std={pos_x.std():.3f}")
    print(f"位置 Y: mean={pos_y.mean():.3f}, std={pos_y.std():.3f}")
    print(f"相对尺寸: mean={rel_sizes.mean():.4f}, std={rel_sizes.std():.4f}, "
          f"min={rel_sizes.min():.4f}, max={rel_sizes.max():.4f}")
    print("-" * 50)
    
    # 绘制位置分布图
    print("\n绘制位置分布图...")
    plot_position(pos_x, pos_y, dataset_name, n_samples, output_dir)
    
    # 绘制尺寸分布图
    print("\n绘制尺寸分布图...")
    plot_size(rel_sizes, dataset_name, n_samples, output_dir)
    
    print("\n全部完成！")
    print(f"输出文件：")
    print(f"  - {os.path.join(output_dir, 'position.svg')} (无文字)")
    print(f"  - {os.path.join(output_dir, 'position.png')} (完整)")
    print(f"  - {os.path.join(output_dir, 'size.svg')} (无文字)")
    print(f"  - {os.path.join(output_dir, 'size.png')} (完整)")


if __name__ == "__main__":
    main()
