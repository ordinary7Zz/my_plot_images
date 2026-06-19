"""
二分类数据集柱状图绘制脚本
功能：根据输入的数据集名称、二分类名称和数量，输出：
  1. 一个不含文字（标题、轴标签、刻度标签、数值标注、图例文字）的 SVG 图
  2. 一个包含完整内容的 PNG 图
"""

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import os

# ======================== 用户输入区域 ========================
# 数据集名称
datasets = ["TN3K", "TN5K", "ThyroidXL", "DDTI", "FinalData"]

# 二分类类别名称
class_names = ["Benign", "Malignant"]

# 每个数据集对应的两类数量 [类别1数量, 类别2数量]
counts = [
    [2709, 2638],      # TN3K
    [1301, 3199],      # TN5K
    [8172, 3459],      # ThyroidXL
    [304, 45],     # DDTI
    [426, 1248],      # FinalData
]

# 输出文件名（不含扩展名）
output_name = "binary_classification_bar"

# 图表标题
title = "Benign vs Malignant Case Counts by Dataset"

# Y轴标签
ylabel = "Number of Cases"

# X轴标签
xlabel = "Dataset"

# ======================== 配置区域 ========================
# 配色：深蓝色和土黄色
colors = ["#1B4F72", "#C6975B"]

# 柱子宽度
bar_width = 0.35

# 输出目录
output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dataset_count")
os.makedirs(output_dir, exist_ok=True)


def plot_chart(save_svg_without_text=True, save_png_with_text=True):
    """绘制二分类柱状图"""

    x = np.arange(len(datasets))

    # 提取两类数据
    class1_counts = [c[0] for c in counts]
    class2_counts = [c[1] for c in counts]

    # ==================== 第一步：输出无文字的 SVG ====================
    if save_svg_without_text:
        fig, ax = plt.subplots(figsize=(10, 6))

        bars1 = ax.bar(x - bar_width / 2, class1_counts, bar_width,
                       color=colors[0], edgecolor='none')
        bars2 = ax.bar(x + bar_width / 2, class2_counts, bar_width,
                       color=colors[1], edgecolor='none')

        # 设置对数刻度
        ax.set_yscale('log')

        # 使用透明色文字占位，预留标题、轴标签、刻度标签的空间
        ax.set_xlabel(xlabel, fontsize=12, color='none')
        ax.set_ylabel(ylabel, fontsize=12, color='none')
        ax.set_title(title, fontsize=14, fontweight='bold', color='none')

        # 设置x轴刻度（透明文字占位）
        ax.set_xticks(x)
        ax.set_xticklabels(datasets, fontsize=11, color='none')

        # y轴刻度标签透明占位
        ax.tick_params(axis='y', which='major', labelcolor='none')

        # 设置y轴范围与完整图一致
        all_counts = class1_counts + class2_counts
        y_min = min(all_counts) * 0.7
        y_max = max(all_counts) * 1.3
        ax.set_ylim(y_min, y_max)

        # 在柱子顶部添加透明数值标注占位
        def add_invisible_labels(bars):
            for bar in bars:
                height = bar.get_height()
                ax.annotate(f'{height:,.0f}',
                            xy=(bar.get_x() + bar.get_width() / 2, height),
                            xytext=(0, 3),
                            textcoords="offset points",
                            ha='center', va='bottom', fontsize=10,
                            color='none')

        add_invisible_labels(bars1)
        add_invisible_labels(bars2)

        # 添加无文字的图例（仅色块，无外边框，用透明文字占位）
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor=colors[0]),
                           Patch(facecolor=colors[1])]
        leg = ax.legend(handles=legend_elements, labels=[class_names[0], class_names[1]],
                        loc='upper right', frameon=False, fontsize=11)
        for text in leg.get_texts():
            text.set_color('none')

        plt.tight_layout()
        svg_path = os.path.join(output_dir, f"{output_name}.svg")
        fig.savefig(svg_path, format='svg', bbox_inches='tight', dpi=150)
        plt.close(fig)
        print(f"SVG (无文字) 已保存: {svg_path}")

    # ==================== 第二步：输出完整的 PNG ====================
    if save_png_with_text:
        fig, ax = plt.subplots(figsize=(10, 6))

        bars1 = ax.bar(x - bar_width / 2, class1_counts, bar_width,
                       color=colors[0], edgecolor='none', label=class_names[0])
        bars2 = ax.bar(x + bar_width / 2, class2_counts, bar_width,
                       color=colors[1], edgecolor='none', label=class_names[1])

        # 设置对数刻度
        ax.set_yscale('log')

        # 设置轴标签和标题
        ax.set_xlabel(xlabel, fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')

        # 设置x轴刻度
        ax.set_xticks(x)
        ax.set_xticklabels(datasets, fontsize=11)

        # 设置y轴范围
        all_counts = class1_counts + class2_counts
        y_min = min(all_counts) * 0.7
        y_max = max(all_counts) * 1.3
        ax.set_ylim(y_min, y_max)

        # 在柱子顶部添加数值标注（带千位分隔符）
        def add_value_labels(bars):
            for bar in bars:
                height = bar.get_height()
                ax.annotate(f'{height:,.0f}',
                            xy=(bar.get_x() + bar.get_width() / 2, height),
                            xytext=(0, 3),  # 3 points vertical offset
                            textcoords="offset points",
                            ha='center', va='bottom', fontsize=10)

        add_value_labels(bars1)
        add_value_labels(bars2)

        # 添加图例
        ax.legend(loc='upper right', fontsize=11, framealpha=0.9)

        plt.tight_layout()
        png_path = os.path.join(output_dir, f"{output_name}.png")
        fig.savefig(png_path, format='png', bbox_inches='tight', dpi=150)
        plt.close(fig)
        print(f"PNG (完整) 已保存: {png_path}")


if __name__ == "__main__":
    plot_chart()
