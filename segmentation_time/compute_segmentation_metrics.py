#!/usr/bin/env python3
"""
分割性能对比图 — 人工 vs AI辅助标注 vs GT
===========================================
读取实验日志 CSV, 加载 corrected mask 和 GT, 生成:
  - 左: Dice 箱线图 + 个体散点
  - 右: 配对散点图 (Manual vs Assisted, y=x 对角线)
  - 底: 统计文字 (均值、p值、优于/持平比例)

输出: PDF + PNG + SVG

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

    # ---- 创建图形 ----
    fig = plt.figure(figsize=(12, 5.5))

    # ==== 左: 箱线图 + 散点 ====
    ax1 = fig.add_subplot(1, 2, 1)
    data = [manual, assisted]
    bp = ax1.boxplot(data, labels=["Manual", "AI-Assisted"],
                     patch_artist=True, widths=0.45, showfliers=False,
                     medianprops={"color": "black", "linewidth": 1.5})
    bp["boxes"][0].set_facecolor(color_manual)
    bp["boxes"][0].set_alpha(0.7)
    bp["boxes"][1].set_facecolor(color_assist)
    bp["boxes"][1].set_alpha(0.7)

    # jitter 散点
    for i, vals in enumerate(data):
        jitter = np.random.normal(0, 0.04, len(vals))
        x = np.ones(len(vals)) * (i + 1) + jitter
        c = color_manual if i == 0 else color_assist
        ax1.scatter(x, vals, alpha=0.55, s=22, c=c, edgecolors="white", linewidth=0.3, zorder=5)

    # 配对连线
    for m_val, a_val in zip(manual, assisted):
        ax1.plot([1, 2], [m_val, a_val], color="gray", alpha=0.2, linewidth=0.6, zorder=1)

    ax1.set_ylabel("Dice Score", fontsize=11)
    ax1.set_title("Dice Distribution", fontsize=12, fontweight="bold")
    ax1.set_ylim(min(0.0, np.min(data) - 0.05), 1.0)
    ax1.grid(axis="y", alpha=0.3)
    ax1.tick_params(labelsize=10)

    # ==== 右: 配对散点 ====
    ax2 = fig.add_subplot(1, 2, 2)
    ax2.scatter(manual, assisted, c="#333333", alpha=0.6, s=40,
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
    ax2.grid(alpha=0.3)
    ax2.tick_params(labelsize=10)

    # ==== 底部统计文字 ====
    summary_lines = [
        f"n = {n} images",
        f"Manual: $\\mu$={m_mean:.4f}, $\\sigma$={np.std(manual, ddof=1):.4f}",
        f"AI-Assisted: $\\mu$={a_mean:.4f}, $\\sigma$={np.std(assisted, ddof=1):.4f}",
        f"Assisted $\\ge$ Manual on {same_or_better}/{n} images ({100*same_or_better/n:.0f}%)",
    ]
    if p_val_str:
        summary_lines.insert(3, f"Wilcoxon signed-rank: {p_val_str}")

    fig.text(0.5, 0.01, " | ".join(summary_lines),
             ha="center", va="bottom", fontsize=9,
             bbox=dict(boxstyle="round,pad=0.4", facecolor="#f5f5f5", edgecolor="#cccccc"))

    plt.subplots_adjust(bottom=0.14, top=0.92, wspace=0.35)
    plt.suptitle("Segmentation Quality: Manual vs AI-Assisted Annotation",
                 fontsize=13, fontweight="bold", y=0.97)

    # ---- 保存 ----
    for fmt in ("png", "pdf", "svg"):
        path = output_dir / f"{prefix}.{fmt}"
        plt.savefig(path, dpi=200, bbox_inches="tight")
        print(f"[PLOT] {path}")

    plt.close()


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
