#!/usr/bin/env python3
"""
Config-driven radial AUROC heatmap from linear-prob summary.json files.

All dataset-specific settings are loaded from a JSON config file.
No dataset names are hardcoded in this script.
"""

import argparse
import glob
import json
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np


DEFAULT_CONFIG_PATH = Path(__file__).with_name("lp_radiomap_config.json")
SUPPORTED_MANUAL_VALUE_KEYS = ("manual_values", "manual_metric_values")


def load_json(path: Path) -> Dict[str, Any]:
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except OSError as e:
        raise RuntimeError(f"Failed to read config file: {path} ({e})") from e
    except json.JSONDecodeError as e:
        raise RuntimeError(f"Invalid JSON in config file: {path} ({e})") from e


def _validate_required_config(cfg: Dict[str, Any]) -> None:
    required_top = ["paths", "plot", "models", "datasets", "category_colors"]
    for key in required_top:
        if key not in cfg:
            raise ValueError(f"Config missing required key: {key}")

    if not isinstance(cfg["datasets"], dict) or not cfg["datasets"]:
        raise ValueError("Config key 'datasets' must be a non-empty object.")

    if not isinstance(cfg["models"], list) or not cfg["models"]:
        raise ValueError("Config key 'models' must be a non-empty array.")


def find_summary_json(clear_dir: Path, sota_dir: Path, model_folder: str) -> Optional[Path]:
    candidates = [
        clear_dir / model_folder / "results_concept_based" / "summary.json",
        clear_dir / model_folder / "results_direct" / "summary.json",
        clear_dir / model_folder / "results" / "summary.json",
        clear_dir / model_folder / "summary.json",
    ]
    for path in candidates:
        if path.exists():
            return path

    sota_folder = "clip" if model_folder == "ctclip" and (sota_dir / "clip").exists() else model_folder
    sota_path = sota_dir / sota_folder
    if sota_path.exists():
        runs = sorted(glob.glob(str(sota_path / "run_*")))
        if runs:
            run_summary = Path(runs[-1]) / "summary.json"
            if run_summary.exists():
                return run_summary
        summary = sota_path / "summary.json"
        if summary.exists():
            return summary
    return None


def load_summary_json(clear_dir: Path, sota_dir: Path, model_folder: str) -> Optional[Dict[str, Any]]:
    path = find_summary_json(clear_dir, sota_dir, model_folder)
    if not path:
        return None

    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except OSError as e:
        print(f"Error loading {path}: {e}")
        return None

    if "individual_aucs" not in data and "predictions" not in data:
        for key, value in data.items():
            if isinstance(value, dict) and "individual_aucs" in value:
                data = value
                break

    return data


def build_pathology_order(category_map: Dict[str, List[str]]) -> List[Tuple[str, str]]:
    return [(pathology, category) for category, plist in category_map.items() for pathology in plist]


def make_label_display(dataset_cfg: Dict[str, Any]) -> Callable[[str], str]:
    label_map = dataset_cfg.get("label_map", {})
    label_style = dataset_cfg.get("label_style", "map_then_identity")

    def _display(label: str) -> str:
        if label in label_map:
            return label_map[label]
        if label_style == "map_then_title_from_underscore":
            return label.replace("_", " ").title()
        return label

    return _display


def dataset_fonts(cfg: Dict[str, Any], dataset_cfg: Dict[str, Any]) -> Dict[str, int]:
    fonts_cfg = cfg.get("fonts", {})
    default_fonts = fonts_cfg.get(
        "default",
        {"category": 18, "cbar_title": 18, "cbar_ticks": 15, "legend": 18, "pathology": 17},
    )
    font_key = dataset_cfg.get("font_key", "default")
    selected = fonts_cfg.get(font_key, {})
    merged = dict(default_fonts)
    merged.update(selected)
    return merged


def load_all_model_metrics_for_dataset(cfg: Dict[str, Any], dataset_name: str) -> Dict[str, Dict[str, float]]:
    clear_base = Path(cfg["paths"]["clear_base"])
    sota_base = Path(cfg["paths"]["sota_base"])
    clear_dir = clear_base / dataset_name
    sota_dir = sota_base / dataset_name

    if not clear_dir.is_dir():
        print(f"[{dataset_name}] Missing linear-prob dir: {clear_dir}")
        return {}

    out: Dict[str, Dict[str, float]] = {}
    for model in cfg["models"]:
        display_name = model["display_name"]
        model_folder = model.get("folders", {}).get(dataset_name)
        if not model_folder:
            continue
        data = load_summary_json(clear_dir, sota_dir, model_folder)
        if data and isinstance(data.get("individual_aucs"), dict):
            out[display_name] = dict(data["individual_aucs"])

    return out



def load_manual_metric_values(dataset_cfg: Dict[str, Any]) -> Dict[str, Dict[str, float]]:
    manual_values = None
    for key in SUPPORTED_MANUAL_VALUE_KEYS:
        candidate = dataset_cfg.get(key)
        if isinstance(candidate, dict):
            manual_values = candidate
            break
    if manual_values is None:
        return {}

    out: Dict[str, Dict[str, float]] = {}
    for model_name, metric_values in manual_values.items():
        if not isinstance(metric_values, dict):
            continue
        out[str(model_name)] = {str(label): float(value) for label, value in metric_values.items()}
    return out


def plot_radial_heatmap(
    all_aucs: Dict[str, Dict[str, float]],
    category_map: Dict[str, List[str]],
    category_colors: Dict[str, str],
    label_display: Callable[[str], str],
    model_order: List[str],
    plot_cfg: Dict[str, Any],
    fonts: Dict[str, int],
    output_path: Path,
    metric_title: str,
) -> None:
    order = build_pathology_order(category_map)
    if not order:
        print("Category map is empty. Please fill dataset category_map in the config.")
        return

    pathologies = [p for p, _ in order]
    categories = [c for _, c in order]

    available_pathologies = set().union(*[set(v.keys()) for v in all_aucs.values()]) if all_aucs else set()
    keep_idx = [i for i, p in enumerate(pathologies) if p in available_pathologies]
    if not keep_idx:
        print("No overlapping pathologies found between config category_map and summary.json keys.")
        return

    pathologies = [pathologies[i] for i in keep_idx]
    categories = [categories[i] for i in keep_idx]
    n_path = len(pathologies)

    models = [m for m in model_order if m in all_aucs]
    if not models:
        print("No models available for plotting after filtering by model_order.")
        return

    r_inner = float(plot_cfg.get("r_inner", 1.2))
    r_per_ring = float(plot_cfg.get("r_per_ring", 0.75))
    gap_units = float(plot_cfg.get("gap_units", 0.16))
    r_outer = r_inner + len(models) * r_per_ring

    base_theta = 2 * np.pi / (n_path * (1.0 + gap_units))
    gap_theta = base_theta * gap_units
    bar_width = base_theta
    centers = np.array([(i * (base_theta + gap_theta)) + base_theta / 2 for i in range(n_path)])

    figsize = tuple(plot_cfg.get("figsize", [20, 20]))
    fig, ax = plt.subplots(subplot_kw=dict(projection="polar"), figsize=figsize)
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)

    cmap_name = plot_cfg.get("cmap", "coolwarm")
    cmap = plt.get_cmap(cmap_name)
    missing_color = plot_cfg.get("missing_color", "#E0E0E0")
    vmin = float(plot_cfg.get("vmin", 0.0))
    vmax = float(plot_cfg.get("vmax", 1.0))
    lower_is_better = metric_title.strip().lower() == "hd95"

    for ring_idx, model_name in enumerate(models):
        metric_values = all_aucs[model_name]
        r0 = r_inner + (len(models) - 1 - ring_idx) * r_per_ring
        for i, pathology in enumerate(pathologies):
            metric_value = metric_values.get(pathology, np.nan)
            if np.isnan(metric_value):
                color = missing_color
            else:
                if lower_is_better:
                    metric_value = min(metric_value, vmax)
                normalized = (metric_value - vmin) / (vmax - vmin) if vmax > vmin else 0.5
                normalized = np.clip(normalized, 0.0, 1.0)
                if lower_is_better:
                    normalized = 1.0 - normalized
                color = cmap(normalized)
            ax.bar(
                centers[i],
                r_per_ring,
                width=bar_width,
                bottom=r0,
                color=color,
                edgecolor="white",
                linewidth=1.5,
                align="center",
            )

    bounds = [0] + [i for i in range(1, len(categories)) if categories[i] != categories[i - 1]] + [n_path]
    for j in range(len(bounds) - 1):
        start, end = bounds[j], bounds[j + 1]
        category = categories[start]
        if start == 0 and end == n_path and len(bounds) == 2:
            continue

        th_l = centers[start] - bar_width / 2
        th_r = centers[end - 1] + bar_width / 2
        arc = np.linspace(th_l, th_r, 120)
        ax.plot(arc, np.full_like(arc, r_inner - 0.12), color=category_colors.get(category, "#888"), lw=6)

        th_mid = 0.5 * (th_l + th_r)
        deg = np.degrees(th_mid)
        if 0 <= deg <= 180:
            rot = 90 - deg
            ha = "right"
        else:
            rot = 270 - deg
            ha = "left"
        ax.text(
            th_mid,
            r_inner - 0.42,
            category,
            ha=ha,
            va="center",
            rotation=rot,
            fontsize=fonts["category"],
            color="black",
            rotation_mode="anchor",
        )

    cax = fig.add_axes([0.84, 0.68, 0.015, 0.15])
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0.0, vmax=1.0))
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cax)
    tick_positions = np.linspace(0.0, 1.0, 5)
    if lower_is_better:
        tick_labels = [f"{value:.1f}" for value in np.linspace(vmax, vmin, 5)]
    else:
        tick_labels = [f"{value:.1f}" for value in np.linspace(vmin, vmax, 5)]
    cbar.set_ticks(tick_positions)
    cbar.set_ticklabels(tick_labels)
    cbar.ax.set_title(metric_title, fontsize=fonts["cbar_title"], pad=15)
    cbar.ax.tick_params(labelsize=fonts["cbar_ticks"])

    legend_labels = [f"{idx + 1}. {model_name}" for idx, model_name in enumerate(models)]
    fig.text(
        0.03,
        0.85,
        "Outer → Inner\n" + "\n".join(legend_labels),
        ha="left",
        va="top",
        fontsize=fonts["legend"],
        linespacing=2,
    )

    label_r = r_outer + 0.15
    for i, pathology in enumerate(pathologies):
        theta = centers[i]
        deg = np.degrees(theta)
        if 0 <= deg <= 180:
            rot = 90 - deg
            ha = "left"
        else:
            rot = 270 - deg
            ha = "right"
        ax.text(
            theta,
            label_r,
            label_display(pathology),
            rotation=rot,
            ha=ha,
            va="center",
            fontsize=fonts["pathology"],
            rotation_mode="anchor",
            clip_on=False,
        )

    ax.set_ylim(0, label_r + 1.0)
    ax.set_yticks([])
    ax.set_xticks([])
    ax.grid(False)
    ax.spines["polar"].set_visible(False)

    dpi = int(plot_cfg.get("dpi", 600))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    fig.savefig(output_path.with_suffix(".pdf"), dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {output_path}")
    print(f"Saved: {output_path.with_suffix('.pdf')}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate radial heatmaps from config-defined datasets and model mappings."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help=f"Path to config JSON (default: {DEFAULT_CONFIG_PATH}).",
    )
    parser.add_argument(
        "--datasets",
        nargs="*",
        default=None,
        help="Datasets to plot (space separated). If omitted, use config default_datasets.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Override output directory from config.paths.default_output_dir.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    cfg = load_json(args.config)
    _validate_required_config(cfg)

    dataset_keys = list(cfg["datasets"].keys())

    output_dir = args.output_dir or Path(cfg["paths"]["default_output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    configured_defaults = cfg.get("default_datasets")
    if configured_defaults:
        target_datasets = args.datasets if args.datasets else list(configured_defaults)
    else:
        target_datasets = args.datasets if args.datasets else list(dataset_keys)

    unknown = [d for d in target_datasets if d not in cfg["datasets"]]
    if unknown:
        raise ValueError(f"Unknown datasets in --datasets: {unknown}. Available: {list(cfg['datasets'].keys())}")

    model_order = list(cfg["plot"].get("model_order", []))
    if not model_order:
        model_order = [m["display_name"] for m in cfg["models"]]

    for dataset_name in target_datasets:
        dataset_cfg = cfg["datasets"][dataset_name]
        all_aucs = load_manual_metric_values(dataset_cfg)
        if not all_aucs:
            all_aucs = load_all_model_metrics_for_dataset(cfg, dataset_name)
        if not all_aucs:
            print(f"[{dataset_name}] No model results found.")
            continue

        print(f"[{dataset_name}] Loaded {len(all_aucs)} models: {list(all_aucs.keys())}")

        file_slug = dataset_cfg.get("file_slug", f"lp_radiomap_{dataset_name}")
        output_png = output_dir / f"{file_slug}_radial_heatmap.png"

        plot_cfg = dict(cfg.get("plot", {}))
        plot_cfg.update(dataset_cfg.get("plot", {}))
        metric_title = dataset_cfg.get("metric_title", plot_cfg.get("metric_title", "AUROC"))

        plot_radial_heatmap(
            all_aucs=all_aucs,
            category_map=dataset_cfg.get("category_map", {}),
            category_colors=cfg.get("category_colors", {}),
            label_display=make_label_display(dataset_cfg),
            model_order=model_order,
            plot_cfg=plot_cfg,
            fonts=dataset_fonts(cfg, dataset_cfg),
            output_path=output_png,
            metric_title=metric_title,
        )


if __name__ == "__main__":
    main()
