#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D


DEFAULT_CONFIG_PATH = Path(__file__).with_name("paired_comparison_config.json")
SUPPORTED_MANUAL_VALUE_KEYS = ("manual_values", "manual_metric_values")
DEFAULT_COMPARISON_PLOT_STYLE = {
    "file_suffix": "paired_comparison",
    "x_label": "Performance score",
    "auto_xlim": True,
    "xlim_min": None,
    "xlim_max": None,
    "xticks": None,
    "figsize": None,
    "dpi": 600,
    "value_decimals": 3,
    "percent_decimals": 1,
    "point_size": 22,
    "line_width": 0.7,
    "title_pad": 8,
    "value_offset_points": 5,
    "improvement_offset_points": 6,
    "improvement_position": "above",
    "value_label_mode": "target",
    "show_improvement_labels": True,
    "show_title": False,
    "top_margin": 0.93,
    "bottom_margin": 0.14,
    "left_margin": 0.25,
    "right_margin": 0.97,
    "row_spacing": 1.0,
    "y_margin": 0.10,
    "title_fontsize": 10.5,
    "xlabel_fontsize": 8.5,
    "ylabel_fontsize": 7.0,
    "value_fontsize": 10.0,
    "percent_fontsize": 10.0,
    "legend_fontsize": 7.0,
    "xtick_fontsize": 7.5,
    "grid_alpha": 0.12,
    "grid_linewidth": 0.45,
    "show_confidence_intervals": True,
    "baseline_color": "#7a8798",
    "target_color": "#b35c44",
    "line_color": "#c7c7c7",
    "improvement_color": "#2f6b3b",
    "grid_color": "#e7e7e7",
    "spine_color": "#3b3b3b",
    "marker_edgecolor": "#ffffff",
    "marker_edgewidth": 0.5,
    "tick_length": 2.5,
    "tick_width": 0.5,
    "legend_marker_size": 4.8,
    "label_linespacing": 1.08,
    "axes_bg_color": "#ffffff",
}
DATASET_COMPARISON_PLOT_STYLE = {
    "malignancy_tasks": {
        "baseline_model": "Baseline",
        "target_model": "ThyroidXAgent",
        "baseline_label": "Baseline",
        "target_label": "ThyroidXAgent",
        "title": "Impact of ThyroidXAgent on Classification Performance",
        "auto_xlim": False,
        "xlim_min": 0.69,
        "xlim_max": 0.91,
        "xticks": [0.70, 0.75, 0.80, 0.85, 0.90],
        "figsize": [6.8, 3.0],
        "point_size": 26,
        "line_width": 0.75,
        "title_pad": 8,
        "value_offset_points": 5,
        "improvement_offset_points": 6,
        "improvement_position": "above",
        "value_label_mode": "target",
        "show_improvement_labels": True,
        "show_title": False,
        "top_margin": 0.94,
        "bottom_margin": 0.13,
        "left_margin": 0.22,
        "right_margin": 0.97,
        "title_fontsize": 10.5,
        "xlabel_fontsize": 8.5,
        "ylabel_fontsize": 8.0,
        "value_fontsize": 10.0,
        "percent_fontsize": 10.0,
        "legend_fontsize": 7.0,
        "xtick_fontsize": 7.5,
        "grid_alpha": 0.10,
        "grid_linewidth": 0.45,
        "show_confidence_intervals": False,
        "baseline_color": "#7f8894",
        "target_color": "#b65c46",
        "line_color": "#c9c9c9",
        "improvement_color": "#2f6b3b",
        "grid_color": "#e7e7e7",
        "spine_color": "#3b3b3b",
        "marker_edgecolor": "#ffffff",
        "marker_edgewidth": 0.5,
        "tick_length": 2.5,
        "tick_width": 0.5,
        "legend_marker_size": 4.8,
        "label_linespacing": 1.08,
        "axes_bg_color": "#ffffff",
    }
}


def load_json(path: Path) -> Dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def style_matplotlib() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "axes.unicode_minus": False,
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.linewidth": 0.7,
            "xtick.direction": "out",
            "ytick.direction": "out",
        }
    )


def load_manual_metric_values(dataset_cfg: Dict[str, Any]) -> Dict[str, Dict[str, float]]:
    manual_values = None
    for key in SUPPORTED_MANUAL_VALUE_KEYS:
        candidate = dataset_cfg.get(key)
        if isinstance(candidate, dict):
            manual_values = candidate
            break
    if manual_values is None:
        return {}

    return {
        str(model_name): {str(label): float(value) for label, value in metric_values.items()}
        for model_name, metric_values in manual_values.items()
        if isinstance(metric_values, dict)
    }


def build_dataset_labels(dataset_cfg: Dict[str, Any]) -> List[str]:
    category_map = dataset_cfg.get("category_map", {})
    return [label for labels in category_map.values() for label in labels]


def format_comparison_label(label: str) -> str:
    parts = [part.strip() for part in str(label).splitlines() if part.strip()]
    if len(parts) >= 3:
        head = parts[0]
        middle = parts[1].rstrip("- ").strip()
        tail = " · ".join(parts[2:])
        return f"{head}\n{middle} · {tail}"
    return "\n".join(parts) if parts else str(label)


def plot_paired_comparison_chart(
    metric_name: str,
    dataset_cfg: Dict[str, Any],
    model_order: List[str],
    output_dir: Path,
) -> None:
    comparison_cfg = dataset_cfg.get("comparison_plot", {})
    style_cfg = {
        **DEFAULT_COMPARISON_PLOT_STYLE,
        **DATASET_COMPARISON_PLOT_STYLE.get(metric_name, {}),
    }
    custom_rows = comparison_cfg.get("rows")
    baseline_model = str(style_cfg.get("baseline_model", model_order[1] if len(model_order) > 1 else model_order[0]))
    target_model = str(style_cfg.get("target_model", model_order[0]))
    baseline_label = str(style_cfg.get("baseline_label", baseline_model))
    target_label = str(style_cfg.get("target_label", target_model))

    rows = []
    if isinstance(custom_rows, list) and custom_rows:
        for item in custom_rows:
            if not isinstance(item, dict):
                continue
            baseline_value = item.get("baseline")
            target_value = item.get("target")
            if baseline_value is None or target_value is None:
                continue
            baseline_value = float(baseline_value)
            target_value = float(target_value)
            improvement = ((target_value - baseline_value) / baseline_value * 100.0) if baseline_value != 0 else np.nan
            rows.append(
                {
                    "label": str(item.get("label", "")),
                    "baseline": baseline_value,
                    "target": target_value,
                    "improvement": float(item.get("improvement", improvement)),
                    "baseline_ci": item.get("baseline_ci"),
                    "target_ci": item.get("target_ci"),
                }
            )
    else:
        dataset_labels = build_dataset_labels(dataset_cfg)
        values_by_model = load_manual_metric_values(dataset_cfg)
        confidence_intervals = load_manual_confidence_intervals(dataset_cfg)

        if not dataset_labels or baseline_model not in values_by_model or target_model not in values_by_model:
            print(f"[{metric_name}] Missing paired comparison data.")
            return

        for label in dataset_labels:
            baseline_value = values_by_model[baseline_model].get(label)
            target_value = values_by_model[target_model].get(label)
            if baseline_value is None or target_value is None:
                continue
            improvement = ((target_value - baseline_value) / baseline_value * 100.0) if baseline_value != 0 else np.nan
            rows.append(
                {
                    "label": label,
                    "baseline": float(baseline_value),
                    "target": float(target_value),
                    "improvement": float(improvement),
                    "baseline_ci": confidence_intervals.get(baseline_model, {}).get(label),
                    "target_ci": confidence_intervals.get(target_model, {}).get(label),
                }
            )

        sort_by = str(style_cfg.get("sort_by", "target")).lower()
        if sort_by == "improvement":
            rows.sort(key=lambda row: row["improvement"])
        elif sort_by == "baseline":
            rows.sort(key=lambda row: row["baseline"])
        else:
            rows.sort(key=lambda row: row["target"])

    if not rows:
        print(f"[{metric_name}] No valid paired comparison rows.")
        return

    baseline_color = style_cfg["baseline_color"]
    target_color = style_cfg["target_color"]
    line_color = style_cfg["line_color"]
    improvement_color = style_cfg["improvement_color"]

    metric_title = dataset_cfg.get("metric_title", metric_name)
    figsize = style_cfg["figsize"] or [9, max(3.8, 1.2 * len(rows) + 1.8)]
    dpi = int(style_cfg["dpi"])
    title = style_cfg.get("title") or f"Impact of {target_label} on {metric_title} Performance"
    x_label = style_cfg["x_label"]
    file_suffix = style_cfg["file_suffix"]
    value_decimals = int(style_cfg["value_decimals"])
    percent_decimals = int(style_cfg["percent_decimals"])
    point_size = float(style_cfg["point_size"])
    line_width = float(style_cfg["line_width"])
    title_pad = float(style_cfg["title_pad"])
    value_offset_points = float(style_cfg["value_offset_points"])
    improvement_offset_points = float(style_cfg["improvement_offset_points"])
    improvement_position = str(style_cfg.get("improvement_position", "above")).lower()
    value_label_mode = str(style_cfg.get("value_label_mode", "both")).lower()
    show_improvement_labels = bool(style_cfg.get("show_improvement_labels", True))
    show_title = bool(style_cfg.get("show_title", True))
    title_fontsize = float(style_cfg["title_fontsize"])
    xlabel_fontsize = float(style_cfg["xlabel_fontsize"])
    ylabel_fontsize = float(style_cfg["ylabel_fontsize"])
    value_fontsize = float(style_cfg["value_fontsize"])
    percent_fontsize = float(style_cfg["percent_fontsize"])
    legend_fontsize = float(style_cfg["legend_fontsize"])
    xtick_fontsize = float(style_cfg["xtick_fontsize"])
    show_confidence_intervals = bool(style_cfg["show_confidence_intervals"])
    auto_xlim = bool(style_cfg["auto_xlim"])
    top_margin = float(style_cfg["top_margin"])
    bottom_margin = float(style_cfg["bottom_margin"])
    left_margin = float(style_cfg["left_margin"])
    right_margin = float(style_cfg["right_margin"])
    grid_alpha = float(style_cfg["grid_alpha"])
    grid_linewidth = float(style_cfg["grid_linewidth"])
    grid_color = str(style_cfg.get("grid_color", "#e6e6e6"))
    spine_color = str(style_cfg.get("spine_color", "#3b3b3b"))
    marker_edgecolor = str(style_cfg.get("marker_edgecolor", "#ffffff"))
    marker_edgewidth = float(style_cfg.get("marker_edgewidth", 0.6))
    tick_length = float(style_cfg.get("tick_length", 3.0))
    tick_width = float(style_cfg.get("tick_width", 0.6))
    legend_marker_size = float(style_cfg.get("legend_marker_size", 5.5))
    label_linespacing = float(style_cfg.get("label_linespacing", 1.0))
    axes_bg_color = str(style_cfg.get("axes_bg_color", "#ffffff"))

    values = [row["baseline"] for row in rows] + [row["target"] for row in rows]
    if show_confidence_intervals:
        for row in rows:
            baseline_ci = row["baseline_ci"]
            target_ci = row["target_ci"]
            if baseline_ci:
                values.extend([float(baseline_ci[0]), float(baseline_ci[1])])
            if target_ci:
                values.extend([float(target_ci[0]), float(target_ci[1])])

    data_min = min(values)
    data_max = max(values)
    data_span = data_max - data_min
    padding = max(data_span * 0.08, 0.01 if data_max <= 1.0 else 1.0)

    configured_xlim_min = style_cfg.get("xlim_min")
    configured_xlim_max = style_cfg.get("xlim_max")
    if auto_xlim:
        vmin = data_min - padding
        vmax = data_max + padding
    else:
        vmin = float(configured_xlim_min) if configured_xlim_min is not None else data_min - padding
        vmax = float(configured_xlim_max) if configured_xlim_max is not None else data_max + padding

    fig, ax = plt.subplots(figsize=tuple(figsize))
    ax.set_facecolor(axes_bg_color)
    fig.subplots_adjust(left=left_margin, right=right_margin, top=top_margin, bottom=bottom_margin)
    y_positions = np.arange(len(rows))

    for y, row in zip(y_positions, rows):
        ax.hlines(y, row["baseline"], row["target"], color=line_color, linewidth=line_width, zorder=1)

        baseline_ci = row["baseline_ci"]
        if show_confidence_intervals and baseline_ci:
            ax.errorbar(
                row["baseline"],
                y,
                xerr=[[row["baseline"] - baseline_ci[0]], [baseline_ci[1] - row["baseline"]]],
                fmt="none",
                ecolor=baseline_color,
                elinewidth=1.0,
                capsize=2,
                zorder=2,
            )

        target_ci = row["target_ci"]
        if show_confidence_intervals and target_ci:
            ax.errorbar(
                row["target"],
                y,
                xerr=[[row["target"] - target_ci[0]], [target_ci[1] - row["target"]]],
                fmt="none",
                ecolor=target_color,
                elinewidth=1.0,
                capsize=2,
                zorder=2,
            )

        ax.scatter(
            row["baseline"],
            y,
            color=baseline_color,
            s=point_size,
            edgecolors=marker_edgecolor,
            linewidths=marker_edgewidth,
            zorder=3,
        )
        ax.scatter(
            row["target"],
            y,
            color=target_color,
            s=point_size,
            edgecolors=marker_edgecolor,
            linewidths=marker_edgewidth,
            zorder=3,
        )

        if value_label_mode in {"both", "baseline"}:
            ax.annotate(
                f"{row['baseline']:.{value_decimals}f}",
                xy=(row["baseline"], y),
                xytext=(0, -value_offset_points),
                textcoords="offset points",
                color=baseline_color,
                fontsize=value_fontsize,
                ha="center",
                va="top",
            )
        if value_label_mode in {"both", "target"}:
            ax.annotate(
                f"{row['target']:.{value_decimals}f}",
                xy=(row["target"], y),
                xytext=(0, -value_offset_points),
                textcoords="offset points",
                color=target_color,
                fontsize=value_fontsize,
                ha="center",
                va="top",
            )

        if show_improvement_labels and np.isfinite(row["improvement"]):
            improvement_va = "bottom" if improvement_position == "above" else "top"
            improvement_dy = improvement_offset_points if improvement_position == "above" else -improvement_offset_points
            ax.annotate(
                f"+{row['improvement']:.{percent_decimals}f}%",
                xy=((row["baseline"] + row["target"]) / 2.0, y),
                xytext=(0, improvement_dy),
                textcoords="offset points",
                color=improvement_color,
                fontsize=percent_fontsize,
                fontweight="normal",
                ha="center",
                va=improvement_va,
            )

    ax.set_yticks(y_positions)
    formatted_labels = [format_comparison_label(row["label"]) for row in rows]
    ax.set_yticklabels(formatted_labels, fontsize=ylabel_fontsize)
    for tick_label in ax.get_yticklabels():
        tick_label.set_linespacing(label_linespacing)
        tick_label.set_horizontalalignment("right")
    ax.invert_yaxis()
    ax.margins(y=0.10)
    ax.set_xlabel(x_label, fontsize=xlabel_fontsize, fontweight="normal")
    if show_title:
        ax.set_title(title, fontsize=title_fontsize, pad=title_pad)
    ax.set_xlim(vmin, vmax)

    xticks = style_cfg.get("xticks")
    if not auto_xlim and isinstance(xticks, list) and xticks:
        ax.set_xticks([float(value) for value in xticks])

    ax.grid(axis="x", linestyle=":", alpha=grid_alpha, linewidth=grid_linewidth, color=grid_color)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_linewidth(0.7)
    ax.spines["bottom"].set_color(spine_color)
    ax.tick_params(axis="x", labelsize=xtick_fontsize, width=tick_width, length=tick_length, color=spine_color)
    ax.tick_params(axis="y", length=0, pad=6)

    legend_handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=baseline_color, markeredgecolor=marker_edgecolor, markeredgewidth=marker_edgewidth, markersize=legend_marker_size, label=baseline_label),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=target_color, markeredgecolor=marker_edgecolor, markeredgewidth=marker_edgewidth, markersize=legend_marker_size, label=target_label),
    ]
    ax.legend(handles=legend_handles, loc="upper right", frameon=False, fontsize=legend_fontsize, handletextpad=0.5, borderaxespad=0.2)

    output_dir.mkdir(parents=True, exist_ok=True)
    file_slug = dataset_cfg.get("file_slug", metric_name)
    output_png = output_dir / f"{file_slug}_{file_suffix}.png"
    fig.savefig(output_png, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {output_png}")


def parse_args(dataset_keys: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate paired comparison charts for config-defined metrics.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument(
        "--datasets",
        nargs="*",
        default=None,
        help=f"Datasets to plot. Available: {', '.join(dataset_keys)}",
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    cfg = load_json(DEFAULT_CONFIG_PATH)
    dataset_keys = list(cfg.get("datasets", {}).keys())
    args = parse_args(dataset_keys)
    if args.config != DEFAULT_CONFIG_PATH:
        cfg = load_json(args.config)

    style_matplotlib()
    output_dir = args.output_dir or Path(cfg["paths"]["default_output_dir"])
    target_datasets = args.datasets if args.datasets else list(cfg.get("default_datasets", []))
    model_order = list(cfg.get("plot", {}).get("model_order", []))

    for dataset_name in target_datasets:
        dataset_cfg = cfg["datasets"].get(dataset_name)
        if not dataset_cfg:
            print(f"Unknown dataset: {dataset_name}")
            continue
        plot_paired_comparison_chart(dataset_name, dataset_cfg, model_order, output_dir)


if __name__ == "__main__":
    main()
