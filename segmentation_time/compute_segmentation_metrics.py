#!/usr/bin/env python3
"""
================================================================================
 分割性能对比图 — 人工标注 vs AI辅助标注的分割质量评估
================================================================================

 背景与目的:
   在医学图像分割任务中, 需要量化评估 AI 辅助标注是否能达到与纯人工标注
   相当的精度。本脚本通过对比同一病例在"纯人工"和"AI辅助"两种模式下的
   标注结果与金标准 (GT) 之间的 Dice 相似系数, 生成直观的可视化对比图。

 核心流程:
   1. 读取实验日志 CSV (由 annotator.py 输出), 提取已完成任务
   2. 将每条日志按 (标注者, 图像名) 配对, 得到 manual / assisted 两种模式
   3. 加载对应模式的 corrected mask 和 GT mask
   4. 逐对计算 Dice 相似系数: Dice = 2 * |pred ∩ gt| / (|pred| + |gt|)
   5. 生成两张独立图形: 箱线图 (Dice Distribution) 和配对散点图 (Paired Comparison)
   6. 使用 Wilcoxon signed-rank 检验评估两种模式的差异显著性
   7. 输出 SVG (不含文字, 仅图形元素) 和 PNG (完整内容) 两种格式

 输出图形内容 (两张独立图, 均含底部统计栏):
   图 1 — "Dice Distribution" ({prefix}_boxplot):
     - 橙色/蓝色箱线图: Manual / AI-Assisted 的 Dice 分布概况
     - 橙色/蓝色散点: 每个病例的原始 Dice 值, 水平随机抖动避免重叠
     - 灰色配对连线: 同一病例在两种模式下的 Dice 变化方向
     - Y 轴范围 0.5–1.0, 聚焦高质量分割区间

   图 2 — "Paired Comparison" ({prefix}_scatter):
     - 散点图: X=Manual Dice, Y=AI-Assisted Dice
     - 红色虚线 y=x: 对角线, 点在上方 = AI 优于手动
     - 坐标轴等比例, 直观对比两种模式的相对表现

   底部统计栏 (两张图均有):
     - 样本量 n, 均值 μ, 标准差 σ
     - Wilcoxon signed-rank 检验 p 值及显著性标注
     - AI-Assisted 不低于 Manual 的病例占比

 使用方式:
   python compute_segmentation_metrics.py \
       --log experiment_single/experiment_log.csv \
       --mask-dir experiment_single/masks \
       --gt-dir datasets/gt \
       --output-dir ./figures

 依赖: pip install opencv-python numpy scipy matplotlib
"""

import os
import csv
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

# ---- matplotlib ----
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

try:
    from scipy import stats as scipy_stats
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

# 中文字体
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# ============================================================
# 可调视觉参数 — 圆点大小与透明度
# ============================================================
# 左面板箱线图上的 jitter 散点
SCATTER_ALPHA_BOXPLOT = 0.8   # 散点透明度 (0=全透明, 1=不透明)
SCATTER_SIZE_BOXPLOT  = 35     # 散点大小 (单位: 磅^2)

# 右面板配对散点图
SCATTER_ALPHA_PAIRED  = 0.8    # 散点透明度
SCATTER_SIZE_PAIRED   = 45     # 散点大小

# jitter 随机偏移与配对连线
JITTER_STD            = 0.04   # 水平随机抖动的标准差
PAIR_LINE_ALPHA       = 0.2    # 配对连线透明度
PAIR_LINE_WIDTH       = 0.6    # 配对连线宽度

# 箱线图线条与填充
BOX_ALPHA             = 0.7    # 箱体填充透明度
BOX_LINE_WIDTH        = 1.0    # 箱体边框 / 须线宽度
MEDIAN_LINE_WIDTH     = 1.5    # 中位数线宽度
GRID_ALPHA            = 0.3    # 网格线透明度


# ============================================================
# Mask 加载 & 指标
# ============================================================

def load_mask(path: str, target_shape: Optional[Tuple[int, int]] = None) -> np.ndarray:
    mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise ValueError(f"Cannot read: {path}")
    if target_shape and mask.shape[:2] != target_shape:
        mask = cv2.resize(mask, (target_shape[1], target_shape[0]))
    return (mask > 128)


def compute_dice(pred: np.ndarray, gt: np.ndarray) -> float:
    inter = np.logical_and(pred, gt).sum()
    total = pred.sum() + gt.sum()
    return (2.0 * inter) / total if total > 0 else 0.0


# ============================================================
# 路径查找
# ============================================================

def find_corrected_mask(mask_dir: Path, image_name: str, annotator: str, mode: str) -> Optional[str]:
    stem = Path(image_name).stem
    name = f"{stem}_{annotator}_{mode}_corrected.png"
    path = mask_dir / name
    return str(path) if path.exists() else None


def find_gt_mask(gt_dir: Path, image_name: str) -> Optional[str]:
    stem = Path(image_name).stem
    for ext in ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'):
        p = gt_dir / f"{stem}{ext}"
        if p.exists():
            return str(p)
    return None


# ============================================================
# 批量计算
# ============================================================

def load_log(log_path: str) -> List[Dict]:
    rows = []
    with open(log_path, "r") as f:
        for row in csv.DictReader(f):
            rows.append(row)
    return rows


def compute_paired_dice(log_rows: List[Dict], mask_dir: str, gt_dir: str
                        ) -> List[Dict]:
    """返回配对的手动/辅助 Dice 列表"""
    mask_dir = Path(mask_dir)
    gt_dir = Path(gt_dir)

    pairs: Dict[Tuple[str, str], Dict[str, dict]] = {}
    for row in log_rows:
        if row.get("finished") != "True":
            continue
        key = (row["annotator"], row["image_name"])
        if key not in pairs:
            pairs[key] = {}
        pairs[key][row["mode"]] = row

    results = []
    for (annotator, image_name), modes in sorted(pairs.items()):
        if "manual" not in modes or "assisted" not in modes:
            continue
        gt_path = find_gt_mask(gt_dir, image_name)
        if gt_path is None:
            continue
        gt = load_mask(gt_path)

        item = {"annotator": annotator, "image_name": image_name}
        for mode in ("manual", "assisted"):
            mp = find_corrected_mask(mask_dir, image_name, annotator, mode)
            if mp is None:
                item[f"{mode}_dice"] = None
            else:
                pred = load_mask(mp, target_shape=gt.shape)
                item[f"{mode}_dice"] = compute_dice(pred, gt)

        if item.get("manual_dice") is not None and item.get("assisted_dice") is not None:
            results.append(item)
    return results


# ============================================================
# 绘图
# ============================================================

def save_svg_then_png(fig: plt.Figure, output_dir: Path, stem: str, png_dpi: int = 200) -> None:
    """保存 SVG (不含所有文字) 和 PNG (完整内容)"""
    import matplotlib.text as mtext

    output_dir.mkdir(parents=True, exist_ok=True)

    # 收集图中所有文字元素
    text_artists = set()
    for ax in fig.axes:
        text_artists.update(ax.texts)
        text_artists.update(ax.get_xticklabels())
        text_artists.update(ax.get_yticklabels())
        if ax.xaxis.label:
            text_artists.add(ax.xaxis.label)
        if ax.yaxis.label:
            text_artists.add(ax.yaxis.label)
        if ax.title:
            text_artists.add(ax.title)
        legend = ax.get_legend()
        if legend:
            text_artists.update(legend.get_texts())
    # fig.text 元素
    for child in fig.get_children():
        if isinstance(child, mtext.Text) and child not in text_artists:
            text_artists.add(child)

    # 隐藏文字 -> 保存 SVG
    for t in text_artists:
        t.set_visible(False)
    fig.savefig(output_dir / f"{stem}.svg", bbox_inches="tight")

    # 恢复文字 -> 保存 PNG
    for t in text_artists:
        t.set_visible(True)
    fig.savefig(output_dir / f"{stem}.png", dpi=png_dpi, bbox_inches="tight")

    print(f"[PLOT] {output_dir / f'{stem}.svg'}")
    print(f"[PLOT] {output_dir / f'{stem}.png'}")


def plot_segmentation_quality(results: List[Dict], output_dir: str,
                               prefix: str = "segmentation_quality"):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    manual = np.array([r["manual_dice"] for r in results])
    assisted = np.array([r["assisted_dice"] for r in results])
    n = len(manual)

    if n == 0:
        print("[ERROR] No paired data found.")
        return

    # ---- 颜色 ----
    color_manual = "#E69F00"   # 橙色
    color_assist = "#56B4E9"   # 蓝色

    # ---- 统计 ----
    m_mean, a_mean = np.mean(manual), np.mean(assisted)
    diffs = assisted - manual
    same_or_better = np.sum(diffs >= -0.005)
    p_val_str = ""
    if HAS_SCIPY and n >= 5:
        try:
            _, p_val = scipy_stats.wilcoxon(manual, assisted, alternative="two-sided")
            p_val_str = f"p = {p_val:.4f}" + (" ***" if p_val < 0.001 else " **" if p_val < 0.01 else " *" if p_val < 0.05 else " n.s.")
        except Exception:
            pass

    # ---- 统计文字 (两个图共用) ----
    summary_lines = [
        f"n = {n} images",
        f"Manual: $\\mu$={m_mean:.4f}, $\\sigma$={np.std(manual, ddof=1):.4f}",
        f"AI-Assisted: $\\mu$={a_mean:.4f}, $\\sigma$={np.std(assisted, ddof=1):.4f}",
        f"Assisted $\\geq$ Manual on {same_or_better}/{n} images ({100*same_or_better/n:.0f}%)",
    ]
    if p_val_str:
        summary_lines.insert(3, f"Wilcoxon signed-rank: {p_val_str}")
    summary_text = " | ".join(summary_lines)

    # ---- 图 1: 箱线图 (Dice Distribution) ----
    fig1 = plt.figure(figsize=(3.8, 4.5))
    ax1 = fig1.add_subplot(1, 1, 1)
    data = [manual, assisted]
    bp = ax1.boxplot(data, labels=["Manual", "AI-Assisted"],
                     patch_artist=True, widths=0.45, showfliers=False,
                     medianprops={"color": "black", "linewidth": MEDIAN_LINE_WIDTH},
                     boxprops={"linewidth": BOX_LINE_WIDTH},
                     whiskerprops={"linewidth": BOX_LINE_WIDTH},
                     capprops={"linewidth": BOX_LINE_WIDTH})
    bp["boxes"][0].set_facecolor(color_manual)
    bp["boxes"][0].set_alpha(BOX_ALPHA)
    bp["boxes"][1].set_facecolor(color_assist)
    bp["boxes"][1].set_alpha(BOX_ALPHA)

    # jitter 散点
    for i, vals in enumerate(data):
        jitter = np.random.normal(0, JITTER_STD, len(vals))
        x = np.ones(len(vals)) * (i + 1) + jitter
        c = color_manual if i == 0 else color_assist
        ax1.scatter(x, vals, alpha=SCATTER_ALPHA_BOXPLOT, s=SCATTER_SIZE_BOXPLOT,
                    c=c, edgecolors="white", linewidth=0.3, zorder=5)

    # 配对连线
    for m_val, a_val in zip(manual, assisted):
        ax1.plot([1, 2], [m_val, a_val], color="gray",
                 alpha=PAIR_LINE_ALPHA, linewidth=PAIR_LINE_WIDTH, zorder=1)

    ax1.set_ylabel("Dice Score", fontsize=11)
    ax1.set_title("Dice Distribution", fontsize=12, fontweight="bold")
    ax1.set_ylim(min(0.5, np.min(data) - 0.05), 1.0)
    ax1.grid(axis="y", alpha=GRID_ALPHA)
    ax1.tick_params(labelsize=10)

    fig1.text(0.5, 0.01, summary_text,
              ha="center", va="bottom", fontsize=9,
              bbox=dict(boxstyle="round,pad=0.4", facecolor="#f5f5f5", edgecolor="#cccccc"))
    fig1.subplots_adjust(bottom=0.16, top=0.94)
    save_svg_then_png(fig1, output_dir, f"{prefix}_boxplot")
    plt.close(fig1)

    # ---- 图 2: 配对散点图 (Paired Comparison) ----
    fig2 = plt.figure(figsize=(5.0, 5.0))
    ax2 = fig2.add_subplot(1, 1, 1)
    ax2.scatter(manual, assisted, c="#333333",
                alpha=SCATTER_ALPHA_PAIRED, s=SCATTER_SIZE_PAIRED,
                edgecolors="white", linewidth=0.3, zorder=5)

    lim_min = max(0.0, min(np.min(manual), np.min(assisted)) - 0.03)
    lim_max = min(1.0, max(np.max(manual), np.max(assisted)) + 0.03)
    ax2.plot([lim_min, lim_max], [lim_min, lim_max], "r--", alpha=0.5, linewidth=1,
             label="y=x (equal)")

    ax2.set_xlabel("Manual Dice", fontsize=11)
    ax2.set_ylabel("AI-Assisted Dice", fontsize=11)
    ax2.set_title("Paired Comparison", fontsize=12, fontweight="bold")
    ax2.set_xlim(lim_min, lim_max)
    ax2.set_ylim(lim_min, lim_max)
    ax2.set_aspect("equal")
    ax2.legend(fontsize=9, loc="lower right")
    ax2.grid(alpha=GRID_ALPHA)
    ax2.tick_params(labelsize=10)

    fig2.text(0.5, 0.01, summary_text,
              ha="center", va="bottom", fontsize=9,
              bbox=dict(boxstyle="round,pad=0.4", facecolor="#f5f5f5", edgecolor="#cccccc"))
    fig2.subplots_adjust(bottom=0.16, top=0.94)
    save_svg_then_png(fig2, output_dir, f"{prefix}_scatter")
    plt.close(fig2)


# ============================================================
# CLI
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="分割性能对比图 — Manual vs AI-Assisted vs GT"
    )
    parser.add_argument("--log", required=True, help="实验日志 CSV")
    parser.add_argument("--mask-dir", required=True, help="corrected mask 目录")
    parser.add_argument("--gt-dir", required=True, help="GT mask 目录")
    parser.add_argument("--output-dir", default="./figures", help="输出目录")
    parser.add_argument("--prefix", default="segmentation_quality",
                        help="输出文件名前缀")

    args = parser.parse_args()

    log_rows = load_log(args.log)
    print(f"Loaded {len(log_rows)} log entries")

    results = compute_paired_dice(log_rows, args.mask_dir, args.gt_dir)
    print(f"Paired images: {len(results)}")

    if len(results) == 0:
        print("No paired data. Check --mask-dir and --gt-dir.")
        return

    plot_segmentation_quality(results, args.output_dir, args.prefix)


if __name__ == "__main__":
    main()
