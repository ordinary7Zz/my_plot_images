#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np


DEFAULT_CONFIG_PATH = Path(__file__).with_name("lp_radiomap_config.json")
SUPPORTED_MANUAL_VALUE_KEYS = ("manual_values", "manual_metric_values")
MODEL_COLORS = [
    "#d62728", "#1f77b4", "#2ca02c", "#9467bd", "#ff7f0e", "#8c564b", "#e377c2", "#17becf",
    "#bcbd22", "#7f7f7f",
]


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


def should_reverse_radar(metric_name: str, dataset_cfg: Dict[str, Any], plot_cfg: Dict[str, Any]) -> bool:
    if "reverse_axis" in plot_cfg:
        return bool(plot_cfg["reverse_axis"])

    metric_keys = {
        str(metric_name).strip().lower(),
        str(dataset_cfg.get("metric_title", "")).strip().lower(),
        str(dataset_cfg.get("file_slug", "")).strip().lower(),
    }
    return "hd95" in metric_keys


def spread_label_radii(values: List[float], vmin: float, vmax: float, min_gap: float) -> List[float]:
    indexed_values = sorted((float(np.clip(value, vmin, vmax)), idx) for idx, value in enumerate(values))
    if not indexed_values:
        return []

    adjusted_sorted = [indexed_values[0][0]]
    for value, _ in indexed_values[1:]:
        adjusted_sorted.append(max(value, adjusted_sorted[-1] + min_gap))

    overflow = adjusted_sorted[-1] - vmax
    if overflow > 0:
        adjusted_sorted = [value - overflow for value in adjusted_sorted]

    underflow = vmin - adjusted_sorted[0]
    if underflow > 0:
        adjusted_sorted = [value + underflow for value in adjusted_sorted]

    adjusted = [0.0] * len(values)
    for adjusted_value, (_, original_idx) in zip(adjusted_sorted, indexed_values):
        adjusted[original_idx] = float(np.clip(adjusted_value, vmin, vmax))
    return adjusted


def build_radial_scale(plot_cfg: Dict[str, Any], vmin: float, vmax: float):
    radial_breaks = plot_cfg.get("radial_breaks")
    radial_positions = plot_cfg.get("radial_positions")
    if radial_breaks and radial_positions:
        breaks = [float(value) for value in radial_breaks]
        positions = [float(value) for value in radial_positions]

        def transform(value: float) -> float:
            clipped_value = float(np.clip(value, vmin, vmax))
            return float(np.interp(clipped_value, breaks, positions))

        tick_positions = positions
        tick_labels = [f"{value:.1f}" for value in breaks]
        return transform, positions[0], positions[-1], tick_positions, tick_labels

    def identity(value: float) -> float:
        return float(np.clip(value, vmin, vmax))

    tick_values = np.linspace(vmin, vmax, 5)
    tick_labels = [f"{tick:.1f}" for tick in tick_values]
    return identity, vmin, vmax, tick_values.tolist(), tick_labels


def plot_radar_chart(metric_name: str, dataset_cfg: Dict[str, Any], model_order: List[str], output_dir: Path) -> None:
    dataset_labels = build_dataset_labels(dataset_cfg)
    values_by_model = load_manual_metric_values(dataset_cfg)
    models = [model for model in model_order if model in values_by_model]
    if not dataset_labels or not models:
        print(f"[{metric_name}] Missing dataset axes or model values.")
        return

    num_axes = len(dataset_labels)
    angles = np.linspace(0, 2 * np.pi, num_axes, endpoint=False).tolist()
    angles += angles[:1]

    plot_cfg = dataset_cfg.get("plot", {})
    vmin = float(plot_cfg.get("vmin", 0.0))
    vmax = float(plot_cfg.get("vmax", 100.0))
    reverse_axis = should_reverse_radar(metric_name, dataset_cfg, plot_cfg)
    radial_transform, radial_min, radial_max, tick_positions, tick_labels = build_radial_scale(plot_cfg, vmin, vmax)
    label_gap = float(plot_cfg.get("label_gap", max((radial_max - radial_min) * 0.035, 0.8)))

    fig, ax = plt.subplots(figsize=(11, 11), subplot_kw={"projection": "polar"})
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    if reverse_axis:
        ax.set_ylim(radial_max, radial_min)
    else:
        ax.set_ylim(radial_min, radial_max)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(dataset_labels, fontsize=18)
    ax.set_yticks(tick_positions)
    ax.set_yticklabels(tick_labels, fontsize=15)
    ax.set_rlabel_position(18)
    ax.grid(True, alpha=0.35)
    ax.set_title(dataset_cfg.get("metric_title", metric_name), pad=28, fontsize=22)

    model_colors = {model_name: MODEL_COLORS[idx % len(MODEL_COLORS)] for idx, model_name in enumerate(models)}

    for model_name in models:
        values = [values_by_model[model_name].get(dataset_name, np.nan) for dataset_name in dataset_labels]
        if any(np.isnan(values)):
            continue
        radial_values = [radial_transform(value) for value in values]
        closed_values = radial_values + radial_values[:1]
        color = model_colors[model_name]
        ax.plot(angles, closed_values, linewidth=2.4, label=model_name, color=color)
        ax.fill(angles, closed_values, color=color, alpha=0.08)

    for axis_idx, angle in enumerate(angles[:-1]):
        axis_values = [values_by_model[model_name].get(dataset_labels[axis_idx], np.nan) for model_name in models]
        if any(np.isnan(axis_values)):
            continue
        label_base_radii = [radial_transform(value) for value in axis_values]
        label_radii = spread_label_radii(label_base_radii, radial_min, radial_max, label_gap)
        for model_name, value, label_radius in zip(models, axis_values, label_radii):
            color = model_colors[model_name]
            va = "top" if reverse_axis else "bottom"
            ax.text(
                angle,
                label_radius,
                f"{value:.2f}",
                fontsize=12,
                color=color,
                ha="center",
                va=va,
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.65, "pad": 0.35},
                clip_on=False,
            )

    ax.legend(loc="upper right", bbox_to_anchor=(1.28, 1.10), frameon=False, fontsize=16)

    output_dir.mkdir(parents=True, exist_ok=True)
    file_slug = dataset_cfg.get("file_slug", metric_name)
    output_png = output_dir / f"{file_slug}_radar.png"
    fig.savefig(output_png, dpi=600, bbox_inches="tight")
    fig.savefig(output_png.with_suffix(".pdf"), dpi=600, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {output_png}")
    print(f"Saved: {output_png.with_suffix('.pdf')}")


def parse_args(dataset_keys: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate radar charts for config-defined metrics.")
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
        plot_radar_chart(dataset_name, dataset_cfg, model_order, output_dir)


if __name__ == "__main__":
    main()
