#!/usr/bin/env python3
"""
全局性能单面板条形图：
  - Dice / AUROC / AUPRC：从 x=0 向右延伸（越长越好）
  - HD95：从 x=1 向左延伸（越短越好，长度=HD95/110）
  分类任务排除 AutoGluon。CI95 误差条。
"""
import re
import csv
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial Unicode MS", "PingFang SC", "Heiti TC",
                        "STHeiti", "SimHei", "DejaVu Sans"],
    "axes.unicode_minus": False,
})

CSV_PATH = Path(__file__).parent / "data" / "per_by_center_ci.csv"
OUT_DIR = Path(__file__).parent / "out"
EXCLUDE_MODELS = {"autogluon"}
HD95_VMAX = 110.0

MODEL_DISPLAY: Dict[str, str] = {
    "dinov3_unet": "ThyroidXAgent",
    "transunet": "TransUNet",
    "medsam2": "MedSAM2",
    "medsegx": "MedSegX",
    "ultrafedfm": "UltraFedFM",
    "biomedclip": "BiomedCLIP",
    "medsiglip": "MedSigLIP",
    "dinov3_unet_multitask": "ThyroidXAgent\n(MT)",
    "autogluon": "AutoGluon",
}

SEG_ONLY = ["dinov3_unet", "medsam2", "medsegx", "transunet"]
CLS_ONLY = ["dinov3_unet_multitask", "medsiglip", "biomedclip"]
# UltraFedFM 在两组分别显示为独立行
SEG_CROSS = ["ultrafedfm"]
CLS_CROSS = ["ultrafedfm"]

# 每行的"显示名 + 适用指标列"，分开 UltraFedFM
# 行配置：(model_code, display_name, applicable_metric_indices, group_label)
ROWS = []
for code in SEG_ONLY:
    ROWS.append((code, MODEL_DISPLAY[code], [0, 1], "Seg"))
for code in SEG_CROSS:
    ROWS.append((code, "UltraFedFM\n(seg)", [0, 1], "Seg"))
for code in CLS_ONLY:
    ROWS.append((code, MODEL_DISPLAY[code], [2, 3], "Cls"))
for code in CLS_CROSS:
    ROWS.append((code, "UltraFedFM\n(cls)", [2, 3], "Cls"))

# 4 个指标
METRICS = [
    {"task": "nodule", "section": "seg", "field_idx": 4,
     "label": "Nodule Dice", "color": "#4C72B0", "reverse": False, "vmax": 1.0,
     "fmt": "{:.3f}"},
    {"task": "nodule", "section": "seg", "field_idx": 5,
     "label": "Nodule HD95 (↓better)", "color": "#55A868", "reverse": True, "vmax": HD95_VMAX,
     "fmt": "{:.1f} mm"},
    {"task": "binary", "section": "cls", "field_idx": 4,
     "label": "Binary AUROC", "color": "#8172B3", "reverse": False, "vmax": 1.0,
     "fmt": "{:.3f}"},
    {"task": "binary", "section": "cls", "field_idx": 5,
     "label": "Binary AUPRC", "color": "#C44E52", "reverse": False, "vmax": 1.0,
     "fmt": "{:.3f}"},
]


def parse_ci(s):
    if not s or not s.strip():
        return None, None, None
    m = re.match(r"\s*([\d.]+)\s*\[\s*([\d.]+)\s*,\s*([\d.]+)\s*\]", s.strip())
    if not m:
        pm = re.match(r"\s*([\d.]+)", s.strip())
        return (float(pm.group(1)) if pm else None, None, None)
    return float(m.group(1)), float(m.group(2)), float(m.group(3))


def read_global_rows(path):
    rows = []
    current = None
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line.startswith("=== 分割任务 ==="):
                current = "seg"; continue
            if line.startswith("=== 分类任务 ==="):
                current = "cls"; continue
            if line.startswith("task,") or not line:
                continue
            if current is None:
                continue
            fields = next(csv.reader([line]))
            if len(fields) < 5 or fields[2].strip() != "全局":
                continue
            rows.append({"task": fields[0], "model": fields[1],
                         "section": current, "fields": fields})
    return rows


def get_model_metrics(rows, model_code, applicable_indices):
    """返回 {metric_idx: {pt, lo, hi}}，仅查 applicable_indices 中的指标。"""
    out = {}
    for j in applicable_indices:
        m = METRICS[j]
        for r in rows:
            if (r["task"] == m["task"] and r["section"] == m["section"]
                    and r["model"] == model_code):
                pt, lo, hi = parse_ci(r["fields"][m["field_idx"]])
                if pt is not None:
                    out[j] = {"pt": pt, "lo": lo or pt, "hi": hi or pt}
                break
    return out


def main():
    rows = read_global_rows(CSV_PATH)

    # 打印概览
    for code, name, applicable, grp in ROWS:
        data = get_model_metrics(rows, code, applicable)
        print(f"\n[{grp}] {name}:")
        for j, v in sorted(data.items()):
            print(f"  {METRICS[j]['label']:24s} {v['pt']:.4f}  [{v['lo']:.4f}, {v['hi']:.4f}]")

    # ── 绘图 ──
    bar_h = 0.42
    gap_in_row = 0.05
    row_spacing = 1.0
    group_gap = 0.6

    # 每行 y 位置 + 各 bar 的 y 偏移
    row_y_data = []
    cur_y = 0.0
    prev_grp = None
    for code, name, applicable, grp in ROWS:
        if prev_grp is not None and grp != prev_grp:
            cur_y -= group_gap
        n = len(applicable)
        offs = np.linspace((n-1)/2, -(n-1)/2, n) * (bar_h + gap_in_row)
        row_y_data.append((cur_y, [cur_y + o for o in offs]))
        cur_y -= row_spacing
        prev_grp = grp

    fig, ax = plt.subplots(figsize=(11, 14), dpi=400)
    fig.patch.set_alpha(0.0)
    ax.set_facecolor("white")

    for (code, name, applicable, grp), (model_y, bar_ys) in zip(ROWS, row_y_data):
        data = get_model_metrics(rows, code, applicable)
        for k, j in enumerate(applicable):
            if j not in data:
                continue
            m = METRICS[j]
            v = data[j]
            by = bar_ys[k]
            pt, lo, hi = v["pt"], v["lo"], v["hi"]

            if m["reverse"]:
                norm_pt = pt / m["vmax"]
                norm_hi = hi / m["vmax"]
                norm_lo = lo / m["vmax"]
                ax.barh(by, norm_pt, height=bar_h, color=m["color"],
                        edgecolor="white", linewidth=0.6, zorder=3)
                lo_e = max(0, norm_pt - norm_lo)
                hi_e = max(0, norm_hi - norm_pt)
                ax.errorbar(norm_pt, by, xerr=[[lo_e], [hi_e]], fmt="none",
                            ecolor="black", elinewidth=1.0, capsize=3,
                            capthick=1.0, zorder=4)
                ax.text(norm_hi + 0.012, by, m["fmt"].format(pt),
                        va="center", ha="left", fontsize=9, color="black")
            else:
                ax.barh(by, pt, height=bar_h, color=m["color"],
                        edgecolor="white", linewidth=0.6, zorder=3)
                lo_e = max(0, pt - lo)
                hi_e = max(0, hi - pt)
                ax.errorbar(pt, by, xerr=[[lo_e], [hi_e]], fmt="none",
                            ecolor="black", elinewidth=1.0, capsize=3,
                            capthick=1.0, zorder=4)
                ax.text(hi + 0.012, by, m["fmt"].format(pt),
                        va="center", ha="left", fontsize=9, color="black")

    # y 轴标签：用 ax.text 手动放（避免 transparent 下 ytick 文字消失）
    ax.set_yticks([r[0] for r in row_y_data])
    ax.set_yticklabels([])  # 关掉默认 ytick 文字
    for (code, name, applicable, grp), (model_y, _) in zip(ROWS, row_y_data):
        ax.text(-0.005, model_y, name, transform=ax.get_yaxis_transform(),
                ha="right", va="center", fontsize=12)

    # 组分隔横线
    grp_changes = []
    prev_grp = None
    for i, (_, _, _, grp) in enumerate(ROWS):
        if prev_grp is not None and grp != prev_grp:
            y_prev = row_y_data[i-1][0]
            y_cur = row_y_data[i][0]
            grp_changes.append((y_prev + y_cur) / 2)
        prev_grp = grp
    for gy in grp_changes:
        ax.axhline(gy, color="lightgray", ls="-", lw=0.8, zorder=1)

    # 组标签（左侧）
    grp_ranges = {}
    for (code, name, applicable, grp), (model_y, _) in zip(ROWS, row_y_data):
        if grp not in grp_ranges:
            grp_ranges[grp] = [model_y, model_y]
        else:
            grp_ranges[grp][0] = max(grp_ranges[grp][0], model_y)
            grp_ranges[grp][1] = min(grp_ranges[grp][1], model_y)
    grp_labels = {"Seg": "Segmentation", "Cls": "Classification"}
    for grp, (y_top, y_bot) in grp_ranges.items():
        ax.text(-0.01, (y_top + y_bot) / 2, grp_labels.get(grp, grp),
                transform=ax.get_yaxis_transform(),
                rotation=90, va="center", ha="right",
                fontsize=11, fontweight="bold", color="#444")

    # 0.5 随机基线
    ax.axvline(0.5, color="gray", ls="--", lw=0.8, zorder=1)

    # y 轴范围：贴紧柱形上下
    y_top = row_y_data[0][0] + 0.5  # 第一行 + 半行间距
    y_bot = row_y_data[-1][0] - 0.5
    ax.set_ylim(y_bot, y_top)

    ax.set_xlim(0, 1.20)
    ax.set_xticks([0, 0.2, 0.4, 0.5, 0.6, 0.8, 1.0])
    ax.set_xticklabels(["0", "0.2", "0.4", "0.5\nchance", "0.6", "0.8", "1.0"], fontsize=10)
    ax.set_xlabel("Normalized score (0~1; HD95 shows /110)  |  Note: HD95 shorter=better",
                  fontsize=10, color="#444")
    ax.set_title("Overall Performance: Nodule Segmentation & Binary Classification",
                 fontsize=14, fontweight="bold", pad=14)

    ax.grid(axis="x", ls=":", color="lightgray", zorder=0)
    ax.set_axisbelow(True)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)

    # 图例（色块加大）
    legend_handles = [Patch(facecolor=m["color"], edgecolor="white",
                            label=m["label"])
                      for m in METRICS]
    legend = ax.legend(handles=legend_handles, loc="lower center",
                       bbox_to_anchor=(0.5, -0.10), ncol=4, frameon=False,
                       fontsize=12, handlelength=3.5, handleheight=2.2,
                       handletextpad=1.2, columnspacing=2.5)

    fig.subplots_adjust(left=0.22, right=0.96, top=0.93, bottom=0.10)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    plt.rcParams["svg.fonttype"] = "none"
    out_png = OUT_DIR / "global_bars_unidirectional.png"
    out_svg = OUT_DIR / "global_bars_unidirectional.svg"
    # PNG 带文字
    fig.savefig(out_png, dpi=400, transparent=True)
    # SVG 不带文字但保留图例色块：隐藏所有文字，保留图形元素
    for txt in ax.texts:
        txt.set_visible(False)
    for lbl in ax.get_xticklabels() + ax.get_yticklabels():
        lbl.set_visible(False)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_title("")
    leg = ax.get_legend()
    if leg is not None:
        for txt in leg.get_texts():
            txt.set_visible(False)  # 仅隐藏图例文字，保留色块
    fig.savefig(out_svg, format="svg", transparent=True)
    plt.close(fig)
    plt.close(fig)
    print(f"\nSaved: {out_png}")
    print(f"Saved: {out_svg}")


if __name__ == "__main__":
    main()
