#!/usr/bin/env python3
"""
统计分析脚本 — 人工 vs AI辅助 分割时间效率分析
================================================
读取实验日志 CSV, 生成:
  1. 描述性统计 (均值/标准差/中位数/IQR)
  2. 配对统计检验 (paired t-test + Wilcoxon)
  3. 箱线图 + 散点图
  4. 论文用的 LaTeX 表格
  5. 每位标注者单独分析

使用方式:
  python analyze_results.py --log experiment_log.csv [--output-dir ./analysis]

依赖: pip install numpy scipy matplotlib
"""

import os
import sys
import csv
import json
import argparse
from pathlib import Path
from datetime import datetime
from collections import defaultdict
from typing import Dict, List, Tuple, Optional

import numpy as np

# 可选依赖
try:
    from scipy import stats as scipy_stats
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    print("[WARN] scipy not installed. Statistical tests will be skipped.")
    print("       Install: pip install scipy")

try:
    import matplotlib
    matplotlib.use("Agg")  # 无头模式
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker
    HAS_MPL = True
    # 中文字体 (macOS)
    plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
except ImportError:
    HAS_MPL = False
    print("[WARN] matplotlib not installed. Plots will be skipped.")
    print("       Install: pip install matplotlib")


# ============================================================
# 数据加载
# ============================================================

def load_log(log_path: str) -> List[Dict]:
    """加载 CSV 日志"""
    rows = []
    with open(log_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # 类型转换
            for k in ["time_seconds", "brush_strokes", "eraser_strokes",
                       "total_strokes", "undo_count", "redo_count",
                       "brush_pixels", "eraser_pixels", "mask_pixel_count"]:
                if k in row:
                    try:
                        row[k] = float(row[k])
                    except (ValueError, TypeError):
                        row[k] = 0.0
            row["finished"] = row.get("finished", "False") == "True"
            rows.append(row)
    return rows


def filter_completed(rows: List[Dict]) -> List[Dict]:
    """过滤出已完成的标注"""
    return [r for r in rows if r.get("finished")]


def group_by_mode(rows: List[Dict]) -> Dict[str, List[Dict]]:
    """按模式分组: manual vs assisted"""
    grouped = defaultdict(list)
    for r in rows:
        grouped[r.get("mode", "manual")].append(r)
    return dict(grouped)


def group_by_annotator_mode(rows: List[Dict]) -> Dict[str, Dict[str, List[Dict]]]:
    """按标注者和模式分组"""
    grouped = defaultdict(lambda: defaultdict(list))
    for r in rows:
        grouped[r.get("annotator", "unknown")][r.get("mode", "manual")].append(r)
    return dict(grouped)


# ============================================================
# 描述性统计
# ============================================================

def compute_stats(values: List[float]) -> Dict:
    """计算描述性统计"""
    arr = np.array(values)
    n = len(arr)
    if n == 0:
        return {"n": 0}
    return {
        "n": n,
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr, ddof=1)) if n > 1 else 0.0,
        "median": float(np.median(arr)),
        "q1": float(np.percentile(arr, 25)),
        "q3": float(np.percentile(arr, 75)),
        "iqr": float(np.percentile(arr, 75) - np.percentile(arr, 25)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "total": float(np.sum(arr)),
    }


def descriptive_table(rows: List[Dict], metrics: List[str]) -> str:
    """
    生成描述统计表格 (Markdown)
    """
    grouped = group_by_mode(rows)
    lines = []
    lines.append("## Descriptive Statistics\n")
    lines.append(f"| Metric | Manual (n={len(grouped.get('manual',[]))}) | AI-Assisted (n={len(grouped.get('assisted',[]))}) |")
    lines.append("|--------|-------------------|-----------------------|")

    for metric in metrics:
        manual_vals = [r[metric] for r in grouped.get("manual", []) if r.get(metric)]
        assist_vals = [r[metric] for r in grouped.get("assisted", []) if r.get(metric)]

        ms = compute_stats(manual_vals)
        as_ = compute_stats(assist_vals)

        if ms["n"] == 0 or as_["n"] == 0:
            continue

        m_str = f"{ms['mean']:.1f} ± {ms['std']:.1f}"
        a_str = f"{as_['mean']:.1f} ± {as_['std']:.1f}"
        metric_display = metric.replace("_", " ").title()

        # 计算时间节省百分比
        if metric == "time_seconds" and ms["mean"] > 0:
            saving = (1 - as_["mean"] / ms["mean"]) * 100
            a_str += f" (↓{saving:.0f}%)"

        lines.append(f"| {metric_display} | {m_str} | {a_str} |")

    return "\n".join(lines)


# ============================================================
# 统计检验
# ============================================================

def run_statistical_tests(rows: List[Dict]) -> str:
    """
    配对检验: 手动 vs 辅助 的时间和操作次数

    使用配对是因为交叉设计中每位标注者对同类图像都做过两种模式。
    更严格的做法是按图像名配对。这里做的是按标注者+模式分组比较。
    """
    if not HAS_SCIPY:
        return "\n## Statistical Tests\n\n*scipy not available, tests skipped.*\n"

    lines = ["\n## Statistical Tests\n"]

    metrics = ["time_seconds", "total_strokes", "undo_count",
               "brush_pixels", "eraser_pixels"]

    # 全局: 两组独立样本的比较
    grouped = group_by_mode(rows)
    manual_global = grouped.get("manual", [])
    assist_global = grouped.get("assisted", [])

    lines.append("\n### Global (All Annotators Pooled)\n")
    lines.append("| Metric | Manual Mean±SD | Assisted Mean±SD | Test | Statistic | p-value |")
    lines.append("|--------|---------------|-------------------|------|-----------|---------|")

    for metric in metrics:
        m_vals = [r[metric] for r in manual_global if r.get(metric, 0) > 0 or metric == "time_seconds"]
        a_vals = [r[metric] for r in assist_global if r.get(metric, 0) > 0 or metric == "time_seconds"]

        if len(m_vals) < 3 or len(a_vals) < 3:
            continue

        # Mann-Whitney U (非配对)
        try:
            u_stat, u_p = scipy_stats.mannwhitneyu(m_vals, a_vals, alternative="two-sided")
            sig = "***" if u_p < 0.001 else "**" if u_p < 0.01 else "*" if u_p < 0.05 else "ns"
            lines.append(
                f"| {metric} | {np.mean(m_vals):.1f}±{np.std(m_vals):.1f} "
                f"| {np.mean(a_vals):.1f}±{np.std(a_vals):.1f} "
                f"| MW-U | U={u_stat:.1f} | {u_p:.4f} {sig} |"
            )
        except Exception as e:
            lines.append(f"| {metric} | - | - | ERROR | - | {e} |")

    # 每位标注者单独分析
    by_ann = group_by_annotator_mode(rows)
    for ann in sorted(by_ann.keys()):
        ann_data = by_ann[ann]
        lines.append(f"\n### Annotator: {ann}\n")

        for metric in ["time_seconds", "total_strokes"]:
            m_vals = [r[metric] for r in ann_data.get("manual", [])]
            a_vals = [r[metric] for r in ann_data.get("assisted", [])]
            if len(m_vals) < 3 or len(a_vals) < 3:
                continue
            try:
                u_stat, u_p = scipy_stats.mannwhitneyu(m_vals, a_vals, alternative="two-sided")
                sig = "***" if u_p < 0.001 else "**" if u_p < 0.01 else "*" if u_p < 0.05 else "ns"
                lines.append(
                    f"- **{metric}**: Manual {np.mean(m_vals):.1f}±{np.std(m_vals):.1f}s vs "
                    f"Assisted {np.mean(a_vals):.1f}±{np.std(a_vals):.1f}s, "
                    f"p={u_p:.4f} {sig}"
                )
            except Exception as e:
                lines.append(f"- **{metric}**: ERROR {e}")

    return "\n".join(lines)


# ============================================================
# 绑定配对 (同一图像由不同标注者处理)
# ============================================================

def build_paired_data(rows: List[Dict]) -> Dict[str, Dict[str, List[Dict]]]:
    """
    尝试按图像名配对: 每个图像可能有 manual 和 assisted 的记录
    (来自不同标注者, 或同一标注者两轮)

    返回: {image_name: {"manual": [...], "assisted": [...]}}
    """
    paired = defaultdict(lambda: {"manual": [], "assisted": []})
    for r in rows:
        img = r.get("image_name", "unknown")
        mode = r.get("mode", "manual")
        paired[img][mode].append(r)
    return dict(paired)


def paired_statistics(rows: List[Dict]) -> str:
    """
    按图像配对的统计检验 (每个图像取平均时间)
    使用 Wilcoxon signed-rank test
    """
    if not HAS_SCIPY:
        return ""

    paired = build_paired_data(rows)
    lines = ["\n### Per-Image Paired Analysis\n"]

    manual_times = []
    assist_times = []
    pairs = []

    for img_name, modes in paired.items():
        m_records = modes.get("manual", [])
        a_records = modes.get("assisted", [])

        if m_records and a_records:
            m_mean = np.mean([r["time_seconds"] for r in m_records])
            a_mean = np.mean([r["time_seconds"] for r in a_records])
            manual_times.append(m_mean)
            assist_times.append(a_mean)
            pairs.append((img_name, m_mean, a_mean))

    if len(pairs) < 5:
        return f"\n*Not enough paired images ({len(pairs)} pairs, need >=5)*\n"

    m_arr = np.array(manual_times)
    a_arr = np.array(assist_times)

    try:
        w_stat, w_p = scipy_stats.wilcoxon(m_arr, a_arr, alternative="two-sided")
        t_stat, t_p = scipy_stats.ttest_rel(m_arr, a_arr)

        # 时间节省
        total_manual = m_arr.sum()
        total_assist = a_arr.sum()
        saving_pct = (1 - total_assist / total_manual) * 100 if total_manual > 0 else 0
        mean_saving_pct = np.mean((1 - a_arr / m_arr) * 100)

        sig_w = "***" if w_p < 0.001 else "**" if w_p < 0.01 else "*" if w_p < 0.05 else "ns"

        lines.append(f"- Paired images: {len(pairs)}")
        lines.append(f"- Manual:     {m_arr.mean():.1f} ± {m_arr.std():.1f}s (median={np.median(m_arr):.1f}s)")
        lines.append(f"- Assisted:   {a_arr.mean():.1f} ± {a_arr.std():.1f}s (median={np.median(a_arr):.1f}s)")
        lines.append(f"- **Time saved: {saving_pct:.1f}% (total), {mean_saving_pct:.1f}% (per-image mean)**")
        lines.append(f"- Paired t-test: t={t_stat:.3f}, p={t_p:.4f}")
        lines.append(f"- **Wilcoxon signed-rank: W={w_stat:.1f}, p={w_p:.6f} {sig_w}**")

    except Exception as e:
        lines.append(f"*Test error: {e}*")

    return "\n".join(lines)


# ============================================================
# 图表生成
# ============================================================

def plot_boxplot_time(rows: List[Dict], output_dir: str):
    """时间对比箱线图"""
    if not HAS_MPL:
        return

    grouped = group_by_mode(rows)
    manual_times = [r["time_seconds"] for r in grouped.get("manual", [])]
    assist_times = [r["time_seconds"] for r in grouped.get("assisted", [])]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # 左: 箱线图
    ax = axes[0]
    data = [manual_times, assist_times]
    bp = ax.boxplot(data, labels=["Manual", "AI-Assisted"],
                    patch_artist=True, widths=0.5)
    bp["boxes"][0].set_facecolor("#E69F00")
    bp["boxes"][0].set_alpha(0.7)
    bp["boxes"][1].set_facecolor("#56B4E9")
    bp["boxes"][1].set_alpha(0.7)

    # 叠加散点
    for i, vals in enumerate(data):
        jitter = np.random.normal(0, 0.04, len(vals))
        ax.scatter(np.ones(len(vals)) * (i + 1) + jitter, vals,
                   alpha=0.5, s=20, c="black", edgecolors="none")

    ax.set_ylabel("Time (seconds)")
    ax.set_title("Segmentation Time Comparison")
    ax.grid(axis="y", alpha=0.3)

    # 右: 每位标注者分组箱线图
    ax2 = axes[1]
    by_ann = group_by_annotator_mode(rows)
    annotators = sorted(by_ann.keys())
    positions = []
    labels = []
    colors_manual = []
    colors_assist = []

    for i, ann in enumerate(annotators):
        m_times = [r["time_seconds"] for r in by_ann[ann].get("manual", [])]
        a_times = [r["time_seconds"] for r in by_ann[ann].get("assisted", [])]

        pos_m = i * 2.5 + 1
        pos_a = i * 2.5 + 2
        positions.extend([pos_m, pos_a])
        labels.extend([f"{ann}\nManual", f"{ann}\nAssisted"])

        if m_times:
            bp2 = ax2.boxplot([m_times], positions=[pos_m], widths=0.6,
                              patch_artist=True)
            bp2["boxes"][0].set_facecolor("#E69F00")
            bp2["boxes"][0].set_alpha(0.7)
        if a_times:
            bp2 = ax2.boxplot([a_times], positions=[pos_a], widths=0.6,
                              patch_artist=True)
            bp2["boxes"][0].set_facecolor("#56B4E9")
            bp2["boxes"][0].set_alpha(0.7)

    ax2.set_xticks(positions)
    ax2.set_xticklabels(labels, fontsize=8)
    ax2.set_ylabel("Time (seconds)")
    ax2.set_title("Per-Annotator Time Comparison")
    ax2.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    out_path = os.path.join(output_dir, "time_comparison_boxplot.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[PLOT] {out_path}")


def plot_bar_comparison(rows: List[Dict], output_dir: str):
    """均值柱状图 (时间 + 操作次数)"""
    if not HAS_MPL:
        return

    grouped = group_by_mode(rows)
    metrics = [
        ("time_seconds", "Time (s)"),
        ("total_strokes", "Total Strokes"),
        ("undo_count", "Undo Count"),
        ("brush_pixels", "Brush Pixels (added)"),
        ("eraser_pixels", "Eraser Pixels (removed)"),
    ]

    fig, axes = plt.subplots(1, len(metrics), figsize=(16, 4))

    for ax, (metric, label) in zip(axes, metrics):
        m_vals = [r[metric] for r in grouped.get("manual", [])]
        a_vals = [r[metric] for r in grouped.get("assisted", [])]

        means = [np.mean(m_vals) if m_vals else 0,
                 np.mean(a_vals) if a_vals else 0]
        stds = [np.std(m_vals) if m_vals else 0,
                np.std(a_vals) if a_vals else 0]

        bars = ax.bar(["Manual", "Assisted"], means, yerr=stds,
                      capsize=8, color=["#E69F00", "#56B4E9"],
                      alpha=0.8, edgecolor="black", linewidth=0.5)

        # 在柱上标注均值
        for bar, mean_val in zip(bars, means):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(stds) * 0.1,
                    f"{mean_val:.1f}", ha="center", va="bottom", fontsize=9)

        ax.set_title(label, fontsize=9)
        ax.grid(axis="y", alpha=0.3)

        # 节省比例
        if means[0] > 0 and metric == "time_seconds":
            saving = (1 - means[1] / means[0]) * 100
            ax.set_title(f"{label}\n({saving:.0f}% time saved)", fontsize=9)

    plt.suptitle("Manual vs AI-Assisted Segmentation Comparison",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    out_path = os.path.join(output_dir, "metrics_bar_comparison.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[PLOT] {out_path}")


def plot_scatter_comparison(rows: List[Dict], output_dir: str):
    """散点图: 手动 vs 辅助时间 (按图像配对)"""
    if not HAS_MPL:
        return

    paired = build_paired_data(rows)
    pairs = []
    for img, modes in paired.items():
        m_records = modes.get("manual", [])
        a_records = modes.get("assisted", [])
        if m_records and a_records:
            m_mean = np.mean([r["time_seconds"] for r in m_records])
            a_mean = np.mean([r["time_seconds"] for r in a_records])
            pairs.append((img, m_mean, a_mean))

    if len(pairs) < 3:
        return

    fig, ax = plt.subplots(figsize=(8, 7))

    m_times = [p[1] for p in pairs]
    a_times = [p[2] for p in pairs]

    max_val = max(max(m_times), max(a_times)) * 1.1

    ax.scatter(m_times, a_times, c="#333333", alpha=0.6, s=40, edgecolors="white", linewidth=0.5)
    # 对角线 y=x (手动=辅助)
    ax.plot([0, max_val], [0, max_val], "r--", alpha=0.5, linewidth=1, label="y=x (no difference)")
    # 线性拟合
    if len(pairs) > 2:
        m_arr = np.array(m_times)
        a_arr = np.array(a_times)
        coeffs = np.polyfit(m_arr, a_arr, 1)
        poly = np.poly1d(coeffs)
        x_fit = np.linspace(0, max_val, 100)
        ax.plot(x_fit, poly(x_fit), "b-", alpha=0.4, linewidth=1.5,
                label=f"Fit: y={coeffs[0]:.2f}x+{coeffs[1]:.1f}")

    ax.set_xlabel("Manual Time (s)")
    ax.set_ylabel("AI-Assisted Time (s)")
    ax.set_title(f"Per-Image Time Comparison ({len(pairs)} paired images)")
    ax.legend(fontsize=9)
    ax.set_xlim(0, max_val)
    ax.set_ylim(0, max_val)
    ax.set_aspect("equal")
    ax.grid(alpha=0.3)

    plt.tight_layout()
    out_path = os.path.join(output_dir, "scatter_paired_comparison.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[PLOT] {out_path}")


# ============================================================
# LaTeX 表格
# ============================================================

def generate_latex_table(rows: List[Dict]) -> str:
    """生成论文用的 LaTeX 表格"""
    grouped = group_by_mode(rows)
    m_vals = [r["time_seconds"] for r in grouped.get("manual", [])]
    a_vals = [r["time_seconds"] for r in grouped.get("assisted", [])]

    ms = compute_stats(m_vals)
    as_ = compute_stats(a_vals)

    saving = (1 - as_["mean"] / ms["mean"]) * 100 if ms["mean"] > 0 else 0

    # 统计检验
    p_str = "-"
    if HAS_SCIPY and len(m_vals) >= 5 and len(a_vals) >= 5:
        try:
            _, p_val = scipy_stats.mannwhitneyu(m_vals, a_vals, alternative="two-sided")
            p_str = f"{p_val:.4f}"
        except:
            pass

    latex = r"""\begin{table}[htbp]
\centering
\caption{Comparison of segmentation time between pure manual annotation and AI-assisted annotation (pre-segmentation correction).}
\label{tab:seg_time}
\begin{tabular}{lccc}
\toprule
\textbf{Metric} & \textbf{Manual} & \textbf{AI-Assisted} & \textbf{p-value} \\
\midrule
"""
    latex += f"Images                              & {ms['n']} & {as_['n']} & - \\\\\n"
    latex += f"Mean time (s)                       & {ms['mean']:.1f} $\\pm$ {ms['std']:.1f} & {as_['mean']:.1f} $\\pm$ {as_['std']:.1f} & {p_str} \\\\\n"
    latex += f"Median time (s)                     & {ms['median']:.1f} & {as_['median']:.1f} & - \\\\\n"
    latex += f"IQR (s)                             & {ms['iqr']:.1f} & {as_['iqr']:.1f} & - \\\\\n"
    latex += f"Time saved                          & - & {saving:.1f}\\% & - \\\\\n"

    # 操作次数
    for metric, label in [("total_strokes", "Mean brush strokes"),
                           ("undo_count", "Mean undo count")]:
        m_metric = [r[metric] for r in grouped.get("manual", [])]
        a_metric = [r[metric] for r in grouped.get("assisted", [])]
        ms2 = compute_stats(m_metric)
        as2 = compute_stats(a_metric)
        latex += f"{label}                 & {ms2['mean']:.1f} $\\pm$ {ms2['std']:.1f} & {as2['mean']:.1f} $\\pm$ {as2['std']:.1f} & - \\\\\n"

    latex += r"""\bottomrule
\end{tabular}
\end{table}
"""
    return latex


# ============================================================
# 主函数
# ============================================================

def run_analysis(log_path: str, output_dir: str):
    """运行完整分析"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 加载数据
    rows = load_log(log_path)
    completed = filter_completed(rows)
    print(f"Loaded {len(rows)} records, {len(completed)} completed.")

    if len(completed) == 0:
        print("No completed records found. Exiting.")
        return

    grouped = group_by_mode(completed)
    print(f"  Manual:   {len(grouped.get('manual', []))} records")
    print(f"  Assisted: {len(grouped.get('assisted', []))} records")

    # 生成报告
    report_parts = []
    report_parts.append(f"# Segmentation Time Efficiency Analysis\n")
    report_parts.append(f"Generated: {datetime.now().isoformat()}\n")
    report_parts.append(f"Data: {log_path}\n")

    # 描述统计
    metrics = ["time_seconds", "total_strokes", "undo_count", "redo_count",
               "brush_pixels", "eraser_pixels", "mask_pixel_count"]
    report_parts.append(descriptive_table(completed, metrics))

    # 配对统计
    report_parts.append(paired_statistics(completed))

    # 全局检验
    report_parts.append(run_statistical_tests(completed))

    # 每位标注者单独统计
    by_ann = group_by_annotator_mode(completed)
    report_parts.append("\n## Per-Annotator Summary\n")
    for ann in sorted(by_ann.keys()):
        report_parts.append(f"### {ann}")
        report_parts.append(descriptive_table(
            by_ann[ann].get("manual", []) + by_ann[ann].get("assisted", []),
            ["time_seconds", "total_strokes"]
        ))
        report_parts.append("")

    # LaTeX 表格
    report_parts.append("\n## LaTeX Table\n")
    report_parts.append("```latex")
    report_parts.append(generate_latex_table(completed))
    report_parts.append("```")

    # 写入报告
    report_path = output_dir / "analysis_report.md"
    with open(report_path, "w") as f:
        f.write("\n".join(report_parts))
    print(f"[REPORT] {report_path}")

    # 生成图表
    if HAS_MPL:
        plot_boxplot_time(completed, str(output_dir))
        plot_bar_comparison(completed, str(output_dir))
        plot_scatter_comparison(completed, str(output_dir))

    # 也保存 JSON 统计
    stats_json = {}
    for mode in ["manual", "assisted"]:
        mode_rows = grouped.get(mode, [])
        stats_json[mode] = {
            "n": len(mode_rows),
            "time": compute_stats([r["time_seconds"] for r in mode_rows]),
            "strokes": compute_stats([r["total_strokes"] for r in mode_rows]),
            "undo": compute_stats([r["undo_count"] for r in mode_rows]),
        }

    # 时间节省
    m_mean = stats_json.get("manual", {}).get("time", {}).get("mean", 0)
    a_mean = stats_json.get("assisted", {}).get("time", {}).get("mean", 0)
    if m_mean > 0:
        stats_json["time_saving_pct"] = round((1 - a_mean / m_mean) * 100, 1)

    json_path = output_dir / "statistics.json"
    with open(json_path, "w") as f:
        json.dump(stats_json, f, indent=2)
    print(f"[JSON]  {json_path}")

    print(f"\n{'='*60}")
    print(f"  Analysis complete!")
    print(f"  Output: {output_dir}")
    print(f"{'='*60}")


# ============================================================
# CLI
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="人工 vs AI辅助 分割时间效率分析"
    )
    parser.add_argument("--log", required=True,
                        help="实验日志 CSV 路径")
    parser.add_argument("--output-dir", default="./analysis_output",
                        help="分析输出目录")
    parser.add_argument("--latex-only", action="store_true",
                        help="只输出 LaTeX 表格到 stdout")
    args = parser.parse_args()

    if args.latex_only:
        rows = load_log(args.log)
        completed = filter_completed(rows)
        print(generate_latex_table(completed))
    else:
        run_analysis(args.log, args.output_dir)


if __name__ == "__main__":
    main()
