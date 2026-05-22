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
    "x_label": "Performance Score",
    "auto_xlim": True,
    "xlim_min": None,
    "xlim_max": None,
    "xticks": None,
    "figsize": None,
    "dpi": 600,
    "value_decimals": 4,
    "percent_decimals": 1,
    "point_size": 28,
    "line_width": 1.2,
    "label_offset_y": 0.14,
    "improvement_offset_y": 0.18,
    "title_fontsize": 12,
    "xlabel_fontsize": 10,
    "ylabel_fontsize": 9,
    "value_fontsize": 8,
    "percent_fontsize": 8,
    "legend_fontsize": 8,
    "show_confidence_intervals": True,
    "baseline_color": "#4c72b0",
    "target_color": "#dd8452",
    "line_color": "#a8a8a8",
    "improvement_color": "#2e7d32",
}
DATASET_COMPARISON_PLOT_STYLE = {
    "malignancy_tasks": {
        "baseline_model": "Baseline",
        "target_model": "ThyroidXAgent",
        "baseline_label": "Baseline",
        "target_label": "ThyroidXAgent",
        "title": "Impact of ThyroidXAgent on Classification Performance",
        "auto_xlim": False,
        "xlim_min": 0.70,
        "xlim_max": 0.90,
        "xticks": [0.70, 0.75, 0.80, 0.85, 0.90],
        "figsize": [7.2, 4.2],
        "point_size": 14,
        "line_width": 1.0,
        "label_offset_y": 0.13,
        "improvement_offset_y": 0.17,
        "title_fontsize": 10,
        "xlabel_fontsize": 8,
        "ylabel_fontsize": 7,
        "value_fontsize": 8,
        "percent_fontsize": 7,
        "show_confidence_intervals": False,
    }
}


def load_json(path: Path) -> Dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


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


def load_manual_confidence_intervals(dataset_cfg: Dict[str, Any]) -> Dict[str, Dict[str, List[float]]]:
    manual_intervals = dataset_cfg.get("manual_confidence_intervals")
    if not isinstance(manual_intervals, dict):
        return {}

    result: Dict[str, Dict[str, List[float]]] = {}
    for model_name, dataset_intervals in manual_intervals.items():
        if not isinstance(dataset_intervals, dict):
            continue
        result[str(model_name)] = {}
        for label, interval in dataset_intervals.items():
            if not isinstance(interval, list) or len(interval) != 2:
                continue
            result[str(model_name)][str(label)] = [float(interval[0]), float(interval[1])]
    return result


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
    label_offset_y = float(style_cfg["label_offset_y"])
    improvement_offset_y = float(style_cfg["improvement_offset_y"])
    title_fontsize = float(style_cfg["title_fontsize"])
    xlabel_fontsize = float(style_cfg["xlabel_fontsize"])
    ylabel_fontsize = float(style_cfg["ylabel_fontsize"])
    value_fontsize = float(style_cfg["value_fontsize"])
    percent_fontsize = float(style_cfg["percent_fontsize"])
    legend_fontsize = float(style_cfg["legend_fontsize"])
    show_confidence_intervals = bool(style_cfg["show_confidence_intervals"])
    auto_xlim = bool(style_cfg["auto_xlim"])

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

        ax.scatter(row["baseline"], y, color=baseline_color, s=point_size, zorder=3)
        ax.scatter(row["target"], y, color=target_color, s=point_size, zorder=3)

        ax.text(row["baseline"], y + label_offset_y, f"{row['baseline']:.{value_decimals}f}", color=baseline_color, fontsize=value_fontsize, ha="center", va="bottom")
        ax.text(row["target"], y + label_offset_y, f"{row['target']:.{value_decimals}f}", color=target_color, fontsize=value_fontsize, ha="center", va="bottom")

        if np.isfinite(row["improvement"]):
            ax.text(
                (row["baseline"] + row["target"]) / 2.0,
                y - improvement_offset_y,
                f"+{row['improvement']:.{percent_decimals}f}%",
                color=improvement_color,
                fontsize=percent_fontsize,
                fontweight="bold",
                ha="center",
                va="center",
            )

    ax.set_yticks(y_positions)
    ax.set_yticklabels([row["label"] for row in rows], fontsize=ylabel_fontsize)
    ax.invert_yaxis()
    ax.set_xlabel(x_label, fontsize=xlabel_fontsize, fontweight="bold")
    ax.set_title(title, fontsize=title_fontsize, pad=10)
    ax.set_xlim(vmin, vmax)

    xticks = style_cfg.get("xticks")
    if not auto_xlim and isinstance(xticks, list) and xticks:
        ax.set_xticks([float(value) for value in xticks])

    ax.grid(axis="x", linestyle="--", alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)

    legend_handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=baseline_color, markersize=6, label=baseline_label),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=target_color, markersize=6, label=target_label),
    ]
    ax.legend(handles=legend_handles, loc="upper right", frameon=False, fontsize=legend_fontsize)

    output_dir.mkdir(parents=True, exist_ok=True)
    file_slug = dataset_cfg.get("file_slug", metric_name)
    output_png = output_dir / f"{file_slug}_{file_suffix}.png"
    fig.savefig(output_png, dpi=dpi, bbox_inches="tight")
    fig.savefig(output_png.with_suffix(".pdf"), dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {output_png}")
    print(f"Saved: {output_png.with_suffix('.pdf')}")


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
