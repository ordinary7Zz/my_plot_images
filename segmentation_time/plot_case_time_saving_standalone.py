#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
人工 vs AI辅助 分割时间效率可视化 — 完整三子图 + 组合图
=========================================================

用途:
    对比人工手动分割 vs AI辅助分割的时间效率，生成四个独立图表(单位: 秒):

    子图 b — "Segmentation time" (分割耗时分布对比):
        小提琴图 + 箱线图 + 散点叠加，对比 Manual vs AI 的分割耗时分布。
        散点 = 配对病例，菱形 = 均值，三角 = 超出上限的离群值带标签。
        标注缩短百分比和配对 t 检验 P 值。

    子图 c — "Within-case time saving" (病例级时间节省):
        每个病例为一个柱子的条形图，按节省时间升序排列。
        绿色 = AI 更快，红色 = 纯人工更快，橙色虚线 = 平均节省时间。
        标注总体缩短百分比、平均节省秒数、AI 更快占比。

    子图 d — "Annotator-stratified time saving" (标注者分层分析):
        按标注者 (Overall / 各标注者ID) 分组展示分割时间减少百分比。
        使用 Bootstrap (5000次) 计算 95% 置信区间。
        标注每个标注者的节省百分比、秒数和 AI 更快占比。

    组合图 b+c+d — 三子图水平并排。

数据来源:
    读取 annotator.py / run_experiment.py 输出的实验日志 CSV
    (每行一次标注任务，包含 annotator, image_name, mode, time_seconds)，
    按 image_name 透视 mode 列，自动转换为配对格式
    (manual_time_sec vs ai_time_sec)，
    并从日志的 annotator 列提取 ai_physician 用于分层分析。

输出文件 (均为 PNG/PDF/SVG 三种格式):
    {stem}_b_segmentation_time.png/.pdf/.svg       — 子图 b: 分割耗时分布
    {stem}_c_case_time_saving.png/.pdf/.svg        — 子图 c: 病例级时间节省
    {stem}_d_annotator_time_saving.png/.pdf/.svg   — 子图 d: 标注者分层
    {stem}_bcd_combined.png/.pdf/.svg              — 三子图组合

使用示例:
    # 默认路径 (读取 ./experiment_log.csv, 输出到 ./output/)
    python plot_case_time_saving_standalone.py

    # 指定输入和输出
    python plot_case_time_saving_standalone.py \\
        --input experiment_001/experiment_log.csv \\
        --output-dir ./figures

    # 自定义输出文件名前缀
    python plot_case_time_saving_standalone.py \\
        --input experiment_001/experiment_log.csv \\
        --stem segmentation_efficiency

依赖:
    pip install matplotlib numpy pandas scipy
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".matplotlib"))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    from scipy import stats
except ImportError:  # pragma: no cover - scipy 为可选依赖，仅用于 P 值标注
    stats = None

# ---- 默认路径 (适配 run_experiment.py 输出) ----
DEFAULT_INPUT = SCRIPT_DIR / "experiment_log.csv"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "output"
DEFAULT_STEM = "segmentation_efficiency"

# ---- 配色方案 ----
DARK = "#222222"
GREY = "#6f6f6f"
MANUAL = "#7C8FA8"   # 纯人工
AI = "#3BA77A"        # AI辅助
GREEN = "#2E5F37"
RED = "#C8574D"       # AI 比人工慢
ORANGE = "#D9893D"    # 均值/Overall


# ============================================================
# 通用辅助函数
# ============================================================

def clean_ax(ax: plt.Axes) -> None:
    """统一坐标轴样式: 隐藏上/右边框, 设置刻度和网格线"""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.9)
    ax.spines["bottom"].set_linewidth(0.9)
    ax.tick_params(axis="both", width=0.8, length=3.2, color=DARK, labelsize=8)
    ax.grid(axis="y", color="#e6e6e6", linewidth=0.6, zorder=0)
    ax.grid(axis="x", visible=False)


def add_panel_label(ax: plt.Axes, label: str, x: float = -0.10, y: float = 1.08) -> None:
    """在左上角添加子图标签 (如 b, c, d)"""
    ax.text(
        x, y, label,
        transform=ax.transAxes,
        fontsize=20, fontweight="bold",
        ha="left", va="top", color="black",
    )


def save_figure_formats(fig: plt.Figure, output_dir: Path, stem: str) -> None:
    """保存图表为 PNG (450 DPI), PDF, SVG 三种格式"""
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / f"{stem}.png", dpi=450, bbox_inches="tight")
    fig.savefig(output_dir / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(output_dir / f"{stem}.svg", bbox_inches="tight")


def format_p_value(p_value: float) -> str:
    """智能格式化 P 值显示"""
    if p_value < 1e-4:
        return f"{p_value:.1e}"
    if p_value < 0.001:
        return f"{p_value:.4f}"
    if p_value < 0.01:
        return f"{p_value:.3f}"
    return f"{p_value:.2f}"


# ============================================================
# 数据加载与转换
# ============================================================

def load_paired_cases(input_path: Path) -> pd.DataFrame:
    """从 annotator.py 输出的日志 CSV 中读取并转换为配对格式。

    annotator.py 日志每行是一条标注记录:
        timestamp, annotator, image_name, image_path, mode, input_mode,
        mask_path_input, time_seconds, ..., finished

    转换逻辑:
        1. 过滤 finished=True 的已完成任务
        2. 按 image_name 透视 mode 列 (manual/assisted)
        3. 从日志的 annotator 列提取 ai_physician
           (每个病例的 AI 辅助模式由哪位标注者完成)

    返回 DataFrame, 包含:
        image_name, manual_time_sec, ai_time_sec, ai_physician
    """
    raw = pd.read_csv(input_path)
    required = {"image_name", "mode", "time_seconds"}
    missing = required - set(raw.columns)
    if missing:
        raise ValueError(f"Input CSV is missing required columns: {sorted(missing)}")

    # 仅保留已完成的任务
    if "finished" in raw.columns:
        raw = raw[raw["finished"] == True].copy()
    raw["time_seconds"] = pd.to_numeric(raw["time_seconds"], errors="coerce")
    raw = raw.dropna(subset=["time_seconds"])

    mode_counts = raw.groupby("mode").size().to_dict()
    print(f"  Mode distribution: {mode_counts}")

    # 按 image_name 透视: 行=病例, 列=manual/assisted, 值=time_seconds
    by_case = raw.pivot_table(
        index="image_name",
        columns="mode",
        values="time_seconds",
        aggfunc="first",
    ).reset_index()

    by_case = by_case.rename(columns={
        "manual": "manual_time_sec",
        "assisted": "ai_time_sec",
    })

    # 提取每个病例的 AI 辅助标注者 (用于子图 d 的分层分析)
    if "annotator" in raw.columns:
        ai_physicians = (
            raw[raw["mode"] == "assisted"]
            .groupby("image_name")["annotator"]
            .agg("first")
            .reset_index()
            .rename(columns={"annotator": "ai_physician"})
        )
        by_case = by_case.merge(ai_physicians, on="image_name", how="left")

    # 清理数值并过滤仅保留两种模式都有的病例
    by_case["manual_time_sec"] = pd.to_numeric(by_case["manual_time_sec"], errors="coerce")
    by_case["ai_time_sec"] = pd.to_numeric(by_case["ai_time_sec"], errors="coerce")
    result = by_case.dropna(subset=["manual_time_sec", "ai_time_sec"]).copy()
    print(f"  Paired cases: {len(result)} (from {len(by_case)} total cases)")
    if "ai_physician" in result.columns:
        print(f"  Physicians: {sorted(result['ai_physician'].dropna().unique())}")
    return result


# ============================================================
# Bootstrap 置信区间
# ============================================================

def bootstrap_mean_ci(values: np.ndarray, n_bootstrap: int = 5000, seed: int = 20260619) -> tuple[float, float]:
    """Bootstrap 法计算均值的 95% 置信区间"""
    values = np.asarray(values, dtype=float)
    if len(values) == 0:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    samples = rng.choice(values, size=(n_bootstrap, len(values)), replace=True)
    means = samples.mean(axis=1)
    low, high = np.percentile(means, [2.5, 97.5])
    return float(low), float(high)


def bootstrap_reduction_ci(manual: np.ndarray, ai: np.ndarray, n_bootstrap: int = 5000, seed: int = 20260619) -> tuple[float, float]:
    """Bootstrap 法计算时间减少百分比的 95% 置信区间"""
    manual = np.asarray(manual, dtype=float)
    ai = np.asarray(ai, dtype=float)
    if len(manual) == 0:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(manual), size=(n_bootstrap, len(manual)))
    manual_samples = manual[indices]
    ai_samples = ai[indices]
    reductions = (manual_samples.mean(axis=1) - ai_samples.mean(axis=1)) / manual_samples.mean(axis=1) * 100
    low, high = np.percentile(reductions, [2.5, 97.5])
    return float(low), float(high)


# ============================================================
# 标注者分层统计
# ============================================================

def physician_saving_summary(by_case: pd.DataFrame) -> pd.DataFrame:
    """按标注者分层汇总时间节省指标。

    返回每行对应一个标注者 (含 Overall 总体), 包含:
        mean_saving_sec, ci_low, ci_high, reduction_pct,
        reduction_ci_low, reduction_ci_high, ai_faster_pct
    """
    paired = by_case.copy()
    paired["manual_sec"] = paired["manual_time_sec"].to_numpy(dtype=float)
    paired["ai_sec"] = paired["ai_time_sec"].to_numpy(dtype=float)
    paired["saving_sec"] = paired["manual_sec"] - paired["ai_sec"]

    groups: list[tuple[str, pd.DataFrame]] = [("Overall", paired)]
    if "ai_physician" in paired.columns:
        groups.extend(
            (str(name), group) for name, group in paired.groupby("ai_physician", sort=True)
        )

    rows = []
    for idx, (physician, group) in enumerate(groups):
        saving = group["saving_sec"].to_numpy(dtype=float)
        manual = group["manual_sec"].to_numpy(dtype=float)
        ai = group["ai_sec"].to_numpy(dtype=float)
        ci_low, ci_high = bootstrap_mean_ci(saving, seed=20260619 + idx)
        reduction_ci_low, reduction_ci_high = bootstrap_reduction_ci(manual, ai, seed=20260619 + idx)
        rows.append({
            "physician": physician,
            "mean_saving_sec": float(saving.mean()),
            "ci_low": ci_low,
            "ci_high": ci_high,
            "reduction_pct": float(saving.mean() / manual.mean() * 100) if manual.mean() > 0 else 0.0,
            "reduction_ci_low": reduction_ci_low,
            "reduction_ci_high": reduction_ci_high,
            "ai_faster_pct": float((saving > 0).mean() * 100),
        })
    return pd.DataFrame(rows)


# ============================================================
# 子图 b: Segmentation time — 分割耗时分布对比 (小提琴图+箱线图)
# ============================================================

def draw_segmentation_time(ax: plt.Axes, by_case: pd.DataFrame, panel_label: str = "b", clip_min: float = 30.0) -> None:
    """绘制 Manual vs AI 分割耗时的小提琴图 + 箱线图 + 散点叠加 (单位: 秒)"""
    add_panel_label(ax, panel_label, x=-0.12, y=1.08)
    manual = by_case["manual_time_sec"].to_numpy(dtype=float)
    ai = by_case["ai_time_sec"].to_numpy(dtype=float)
    data = [manual, ai]
    colors = [MANUAL, AI]

    # 小提琴图 (半透明底色)
    parts = ax.violinplot(data, positions=[0, 1], widths=0.72,
                          showmeans=False, showmedians=False, showextrema=False)
    for body, color in zip(parts["bodies"], colors):
        body.set_facecolor(color)
        body.set_edgecolor(color)
        body.set_alpha(0.20)
        body.set_linewidth(0.9)

    # 箱线图 (白色填充, 无离群点)
    bp = ax.boxplot(
        data, positions=[0, 1], widths=0.24,
        patch_artist=True, showfliers=False,
        medianprops={"color": DARK, "linewidth": 1.25},
        whiskerprops={"color": DARK, "linewidth": 0.9},
        capprops={"color": DARK, "linewidth": 0.9},
    )
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor("#ffffff")
        patch.set_edgecolor(color)
        patch.set_linewidth(1.25)

    # 散点 (配对病例) + 底部菱形 (均值)
    rng = np.random.default_rng(42)
    top_y = clip_min - 0.38 * (clip_min / 8.0)  # 按比例缩放 top 空白区
    for x_pos, values, color in zip([0, 1], data, colors):
        jitter = rng.normal(0, 0.058 * (clip_min / 8.0), len(values))
        in_range = values <= clip_min
        ax.scatter(
            np.full(in_range.sum(), x_pos) + jitter[in_range], values[in_range],
            s=13, alpha=0.44, color=color,
            edgecolor="white", linewidth=0.25, zorder=3,
        )
        clipped_values = values[~in_range]
        if len(clipped_values):
            triangle_x = x_pos + (-0.30 if x_pos == 0 else 0.30)
            label_x = triangle_x + 0.06
            ax.scatter(
                np.full(len(clipped_values), triangle_x),
                np.full(len(clipped_values), top_y),
                marker="^", s=48, alpha=0.90,
                color=color, edgecolor=DARK, linewidth=0.45,
                zorder=5, clip_on=False,
            )
            for value in clipped_values:
                ax.text(label_x, top_y, f"{value:.1f} s",
                        ha="left", va="center", fontsize=7.0,
                        color=GREY, fontweight="bold")
        ax.scatter([x_pos], [np.mean(values)],
                   marker="D", s=48, color=color,
                   edgecolor=DARK, linewidth=0.65, zorder=4)

    # 顶部标注: 缩短百分比 + 配对 t 检验 P 值
    saving = manual - ai
    reduction = saving.mean() / manual.mean() * 100
    speedup = manual.mean() / ai.mean()
    p_label = f"; P={format_p_value(stats.ttest_rel(manual, ai).pvalue)}" if stats is not None else ""

    bracket_y = clip_min * 0.845
    tick = clip_min * 0.025
    ax.plot([0, 0, 1, 1],
            [bracket_y - tick, bracket_y, bracket_y, bracket_y - tick],
            color=DARK, linewidth=0.9)
    ax.text(0.5, bracket_y + clip_min * 0.025,
            f"{reduction:.1f}% shorter segmentation time",
            ha="center", va="bottom", fontsize=8.4,
            fontweight="bold", color=ORANGE)

    ax.set_title("Segmentation time", fontsize=11, fontweight="bold", pad=17)
    ax.text(0.5, 1.01,
            f"{manual.mean():.1f} -> {ai.mean():.1f} s; {speedup:.2f}x faster{p_label}",
            transform=ax.transAxes, ha="center", va="bottom",
            fontsize=8.2, color=GREY)
    ax.set_xticks([0, 1], [f"Manual\nn={len(manual)}", f"AI-Assisted\nn={len(ai)}"])
    ax.set_ylabel("Segmentation time (s)")
    ax.set_xlim(-0.55, 1.55)
    ax.set_ylim(0, clip_min)
    ax.set_yticks(np.arange(0, clip_min + 0.1, clip_min / 4))
    clean_ax(ax)
    ax.text(0.98, -0.22,
            f"Dots, paired cases; diamonds, means. Top triangles show values >{clip_min:g} s.",
            transform=ax.transAxes, ha="right", va="top",
            fontsize=7.0, color=GREY)


# ============================================================
# 子图 c: Within-case time saving — 病例级时间节省柱状图
# ============================================================

def draw_case_time_saving(ax: plt.Axes, by_case: pd.DataFrame, panel_label: str = "c") -> None:
    """绘制每个病例的时间节省柱状图 (Manual - AI 耗时, 单位: 秒)"""
    add_panel_label(ax, panel_label)
    paired = by_case.copy()
    paired["saving_sec"] = paired["manual_time_sec"] - paired["ai_time_sec"]
    paired = paired.sort_values("saving_sec").reset_index(drop=True)

    values = paired["saving_sec"].to_numpy(dtype=float)
    x = np.arange(len(values))
    colors = np.where(values >= 0, AI, RED)
    mean_saving = float(values.mean())
    ai_faster_pct = float((values > 0).mean() * 100)
    manual_mean_sec = float(paired["manual_time_sec"].mean())
    mean_reduction_pct = mean_saving / manual_mean_sec * 100 if manual_mean_sec > 0 else 0.0

    ax.bar(x, values, color=colors, width=0.86, linewidth=0)
    ax.axhline(0, color=DARK, linewidth=1.0)
    ax.axhline(mean_saving, color=ORANGE, linewidth=1.1, linestyle="--")
    ax.text(
        len(values) * 0.03, mean_saving + 0.34,
        f"{mean_reduction_pct:.1f}% shorter overall\n{mean_saving:.1f} s/case saved",
        ha="left", va="bottom", fontsize=7.7,
        color=ORANGE, fontweight="bold", linespacing=0.95,
    )

    ax.set_title("Within-case time saving", fontsize=11, fontweight="bold", pad=14)
    ax.text(0.5, 1.01,
            f"AI faster in {ai_faster_pct:.1f}% of paired cases",
            transform=ax.transAxes, ha="center", va="bottom",
            fontsize=8.3, color=GREY)
    ax.set_ylabel("Manual \u2212 AI time (s)")
    ax.set_xlabel("Paired cases sorted by time saving")
    ax.set_xticks([])
    ax.set_xlim(-1, len(values))
    ax.margins(y=0.12)
    clean_ax(ax)


# ============================================================
# 子图 d: Annotator-stratified time saving — 标注者分层对比
# ============================================================

def draw_physician_time_saving(ax: plt.Axes, by_case: pd.DataFrame, panel_label: str = "d") -> None:
    """按标注者分层展示分割时间减少百分比 (带 Bootstrap 95% CI)"""
    add_panel_label(ax, panel_label, x=-0.16, y=1.08)
    summary = physician_saving_summary(by_case)

    # 自动为不同的标注者分配颜色和标记
    color_palette = [ORANGE, AI, GREEN, "#5B8EB9", "#C8574D"]
    marker_palette = ["D", "o", "s", "^", "p"]
    all_physicians = summary["physician"].tolist()
    color_map = {p: color_palette[i % len(color_palette)] for i, p in enumerate(all_physicians)}
    marker_map = {p: marker_palette[i % len(marker_palette)] for i, p in enumerate(all_physicians)}

    x_positions = np.arange(len(summary))
    for x_pos, (_, row) in zip(x_positions, summary.iterrows()):
        is_overall = row["physician"] == "Overall"
        physician = row["physician"]
        color = color_map.get(physician, AI)
        marker = marker_map.get(physician, "o")
        lower = row["reduction_pct"] - row["reduction_ci_low"]
        upper = row["reduction_ci_high"] - row["reduction_pct"]

        ax.errorbar(
            [x_pos], [row["reduction_pct"]],
            yerr=np.array([[lower], [upper]]),
            fmt=marker,
            markersize=7.1 if is_overall else 7.0,
            color=color, ecolor=color,
            elinewidth=1.75, capsize=4.2, capthick=1.15,
            markeredgecolor=DARK, markeredgewidth=0.55, zorder=3,
        )
        ax.text(x_pos, row["reduction_ci_high"] + 1.3,
                f"{row['reduction_pct']:.1f}%",
                ha="center", va="bottom", fontsize=8.5,
                fontweight="bold", color=color)
        ax.text(x_pos, -0.07,
                f"{row['mean_saving_sec']:.1f} s/case\n{row['ai_faster_pct']:.1f}% faster",
                transform=ax.get_xaxis_transform(),
                ha="center", va="top", fontsize=6.45,
                color=GREY, linespacing=0.96, clip_on=False)

    upper_limit = max(45, float(summary["reduction_ci_high"].max()) + 7)
    ax.set_xlim(-0.55, len(summary) - 0.45)
    ax.set_ylim(0, upper_limit)
    ax.set_xticks(x_positions, summary["physician"].tolist())
    ax.tick_params(axis="x", length=0, pad=35, labelsize=8)
    ax.set_yticks(np.arange(0, upper_limit + 0.1, 10))
    ax.set_ylabel("Segmentation time reduction (%)", fontsize=8.4)
    ax.set_title("Annotator-stratified time saving", fontsize=10.5, fontweight="bold", pad=9)
    clean_ax(ax)


# ============================================================
# 批量生成所有图表
# ============================================================

def draw_all_panels(by_case: pd.DataFrame, output_dir: Path, stem: str) -> None:
    """生成三个独立子图 + 一个三合一组图, 每个输出 PNG/PDF/SVG"""
    # 三个独立子图
    specs = [
        (f"{stem}_b_segmentation_time", (4.6, 3.35),
         lambda ax: draw_segmentation_time(ax, by_case, "b")),
        (f"{stem}_c_case_time_saving", (5.2, 3.15),
         lambda ax: draw_case_time_saving(ax, by_case, "c")),
        (f"{stem}_d_annotator_time_saving", (4.15, 3.05),
         lambda ax: draw_physician_time_saving(ax, by_case, "d")),
    ]
    for output_stem, figsize, drawer in specs:
        fig, ax = plt.subplots(figsize=figsize, facecolor="white")
        drawer(ax)
        save_figure_formats(fig, output_dir, output_stem)
        plt.close(fig)

    # 三子图水平组合
    fig, axes = plt.subplots(
        1, 3, figsize=(14.4, 3.45), facecolor="white",
        gridspec_kw={"width_ratios": [1.08, 1.22, 1.08]},
    )
    draw_segmentation_time(axes[0], by_case, "b")
    draw_case_time_saving(axes[1], by_case, "c")
    draw_physician_time_saving(axes[2], by_case, "d")
    fig.subplots_adjust(wspace=0.44)
    save_figure_formats(fig, output_dir, f"{stem}_bcd_combined")
    plt.close(fig)


# ============================================================
# CLI 入口
# ============================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="绘制人工 vs AI辅助分割时间效率对比图 (子图 b/c/d + 组合图)。"
                    "输入为 annotator.py / run_experiment.py 输出的实验日志 CSV。"
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT,
                        help="annotator.py 输出的 experiment_log.csv 路径")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR,
                        help="输出目录 (自动创建)")
    parser.add_argument("--stem", default=DEFAULT_STEM,
                        help="输出文件名前缀")
    args = parser.parse_args()

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "axes.titlesize": 9,
        "axes.labelsize": 8,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })

    by_case = load_paired_cases(args.input)
    draw_all_panels(by_case, args.output_dir, args.stem)
    print(f"\nWrote b/c/d panel exports (PNG/PDF/SVG) to {args.output_dir}")
    print(f"  Stem prefix: {args.stem}")
    print(f"  Total: 4 figures × 3 formats = 12 files")


if __name__ == "__main__":
    main()
