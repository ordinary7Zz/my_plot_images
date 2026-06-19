#!/usr/bin/env python3
import argparse
import glob
import json
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np


DEFAULT_CONFIG_PATH = Path(__file__).with_name("circular_map_config.json")
SUPPORTED_MANUAL_VALUE_KEYS = ("manual_values", "manual_metric_values")
DEFAULT_PLOT_STYLE = {
    "start_angle": 90.0,  # 圆环绘图起始角度（单位：度）
    "end_angle": 360.0,  # 圆环绘图结束角度（单位：度）
    "endspace": False,  # 结束位置是否保留空白扇区
    "sector_spacing": 12.0,  # 不同扇区之间的间隔角度（单位：度）
    "track_outer_radius": 100.0,  # 最外层轨道的外半径
    "track_thickness": 10.0,  # 每一层轨道的厚度
    "track_gap": 2.0,  # 相邻轨道之间的径向间距
    "track_inner_radius": 0.0,  # 最内层轨道的内半径
    "track_pad_ratio": 0.1,  # 轨道内部留白比例，用于控制热图与边缘的间距
    "figsize": [12, 12],  # 画布大小（宽, 高），单位英寸
    "dpi": 600,  # 输出图像分辨率
    "cmap": "Purples",  # 默认颜色映射方案
    "group_cmap": "Set1",  # 分组标签使用的调色板
    "missing_color": "#E0E0E0",  # 缺失值显示颜色
    "rect_edgecolor": "white",  # 热图矩形边框颜色
    "rect_linewidth": 0.3,  # 热图矩形边框宽度
    "label_margin": 3.0,  # 外圈标签与圆环之间的距离
    "label_orientation": "vertical",  # 外圈标签的显示方向
    "show_outer_axis": False,  # 是否显示最外层坐标轴
    "metric_title": "Score",  # 色条标题文字
    "colorbar_bounds": [0.88, 0.75, 0.025, 0.2],  # 色条位置与大小：[left, bottom, width, height]
    "colorbar_labelpad": 14.0,  # 色条标题与色条之间的间距
    "colorbar_label_rotation": 90.0,  # 色条标题旋转角度
    "model_text_prefix": " ",  # 模型名文字前缀，通常留一个空格避免贴边
    "model_text_ha": "left",  # 模型名文字的水平对齐方式
    "model_text_color": "black",  # 模型名文字颜色
    "model_text_deg": 0.0,  # 模型名文字所在角度
    "model_text_orientation": "horizontal",  # 模型名文字排布方向
}
DEFAULT_FONT_STYLE = {
    "category": 52, 
    "cbar_title": 40, # 色条标题
    "cbar_ticks": 40, # 色条数字
    "legend": 50, # 模型名
    "pathology": 42, # 外圈标签
}
DATASET_PLOT_OVERRIDES = {
    "auroc": {
        "vmin": 0.5,
        "vmax": 1.0,
        "figsize": [14, 14],
        "color_breaks": [0.5,1.0],
        "color_positions": [0.0, 1.0],
    },
    "auprc": {
        "vmin": 0.3,
        "vmax": 1.0,
        "figsize": [14, 14],
        "color_breaks": [0.3, 1.0],
        "color_positions": [0.0, 1.0],
    },
    "dice": {
        "vmin": 70.0,
        "vmax": 90.0,
        "figsize": [14, 14],
        "cmap": "PuBu",
        "color_breaks": [70.0, 90.0],
        "color_positions": [0.0, 1.0],
    },
    "hd95": {
        "vmin": 5.0,
        "vmax": 35.0,
        "figsize": [14, 14],
        "cmap": "PuBu",
        "lower_is_better": True,
        "invert_colorbar": True,
        "color_breaks": [5.0, 30.0],
        "color_positions": [0.0, 1.0],
    },
}


def build_plot_config(dataset_name: str, dataset_cfg: Dict[str, Any]) -> Dict[str, Any]:
    plot_cfg = dict(DEFAULT_PLOT_STYLE)
    plot_cfg.update(DATASET_PLOT_OVERRIDES.get(dataset_name, {}))
    dataset_plot = dataset_cfg.get("plot", {})
    if isinstance(dataset_plot, dict):
        plot_cfg.update(dataset_plot)
    return plot_cfg


def dataset_fonts(dataset_name: str) -> Dict[str, int]:
    return dict(DEFAULT_FONT_STYLE)


def build_color_normalizer(plot_cfg: Dict[str, Any], vmin: float, vmax: float) -> Tuple[Callable[[np.ndarray], np.ndarray], List[float], List[str]]:
    breaks = plot_cfg.get("color_breaks")
    positions = plot_cfg.get("color_positions")
    if not isinstance(breaks, list) or not isinstance(positions, list):
        tick_values = np.linspace(vmin, vmax, 5).tolist()
        tick_labels = [f"{tick:.1f}" if vmax <= 1.0 else f"{tick:.0f}" for tick in tick_values]

        def linear_map(values: np.ndarray) -> np.ndarray:
            clipped = np.clip(values, vmin, vmax)
            return (clipped - vmin) / (vmax - vmin) if vmax != vmin else np.zeros_like(clipped)

        return linear_map, tick_values, tick_labels

    if len(breaks) != len(positions) or len(breaks) < 2:
        raise ValueError("plot.color_breaks and plot.color_positions must have the same length >= 2.")

    break_values = [float(value) for value in breaks]
    position_values = [float(value) for value in positions]

    if break_values != sorted(break_values):
        raise ValueError("plot.color_breaks must be sorted in ascending order.")
    if position_values != sorted(position_values):
        raise ValueError("plot.color_positions must be sorted in ascending order.")
    if break_values[0] < vmin or break_values[-1] > vmax:
        raise ValueError("plot.color_breaks must stay within [vmin, vmax].")
    if position_values[0] < 0.0 or position_values[-1] > 1.0:
        raise ValueError("plot.color_positions must stay within [0, 1].")

    def piecewise_map(values: np.ndarray) -> np.ndarray:
        return np.interp(np.clip(values, vmin, vmax), break_values, position_values)

    tick_labels = [f"{tick:.1f}" if vmax <= 1.0 else f"{tick:.0f}" for tick in break_values]
    return piecewise_map, break_values, tick_labels


def load_json(path: Path) -> Dict[str, Any]:
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except OSError as e:
        raise RuntimeError(f"Failed to read config file: {path} ({e})") from e
    except json.JSONDecodeError as e:
        raise RuntimeError(f"Invalid JSON in config file: {path} ({e})") from e


def validate_required_config(cfg: Dict[str, Any]) -> None:
    required_top = ["paths", "models", "datasets"]
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
        for value in data.values():
            if isinstance(value, dict) and "individual_aucs" in value:
                data = value
                break

    return data


def load_all_model_metrics_for_dataset(cfg: Dict[str, Any], dataset_name: str) -> Dict[str, Dict[str, float]]:
    clear_base = Path(cfg["paths"].get("clear_base", ""))
    sota_base = Path(cfg["paths"].get("sota_base", ""))
    clear_dir = clear_base / dataset_name
    sota_dir = sota_base / dataset_name

    if not clear_dir.is_dir():
        print(f"[{dataset_name}] Missing linear-prob dir: {clear_dir}")
        return {}

    out: Dict[str, Dict[str, float]] = {}
    for model in cfg["models"]:
        display_name = str(model["display_name"])
        model_folder = model.get("folders", {}).get(dataset_name)
        if not model_folder:
            continue
        data = load_summary_json(clear_dir, sota_dir, str(model_folder))
        if data and isinstance(data.get("individual_aucs"), dict):
            out[display_name] = {str(k): float(v) for k, v in data["individual_aucs"].items()}

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


def make_label_display(dataset_cfg: Dict[str, Any]) -> Callable[[str], str]:
    label_map = dataset_cfg.get("label_map", {})
    label_style = dataset_cfg.get("label_style", "map_then_identity")

    def _display(label: str) -> str:
        if label in label_map:
            return str(label_map[label])
        if label_style == "map_then_title_from_underscore":
            return label.replace("_", " ").title()
        return label

    return _display


def resolve_model_order(cfg: Dict[str, Any]) -> List[str]:
    return [str(model["display_name"]) for model in cfg.get("models", [])]


def build_group_order(dataset_cfg: Dict[str, Any]) -> List[str]:
    configured = dataset_cfg.get("group_order")
    category_map = dataset_cfg.get("category_map", {})
    if isinstance(configured, list) and configured:
        ordered = [str(group_name) for group_name in configured if group_name in category_map]
        missing = [group_name for group_name in category_map if group_name not in ordered]
        return ordered + missing
    return [str(group_name) for group_name in category_map.keys()]


def build_track_ranges(model_count: int, plot_cfg: Dict[str, Any]) -> List[Tuple[float, float]]:
    configured = plot_cfg.get("track_ranges")
    if isinstance(configured, list) and configured:
        track_ranges: List[Tuple[float, float]] = []
        for item in configured[:model_count]:
            if not isinstance(item, list) or len(item) != 2:
                raise ValueError("Each plot.track_ranges item must be a two-element array.")
            track_ranges.append((float(item[0]), float(item[1])))
        if len(track_ranges) < model_count:
            raise ValueError("plot.track_ranges does not provide enough ranges for the selected models.")
        return track_ranges

    track_outer_radius = float(plot_cfg.get("track_outer_radius", DEFAULT_PLOT_STYLE["track_outer_radius"]))
    track_gap = float(plot_cfg.get("track_gap", DEFAULT_PLOT_STYLE["track_gap"]))
    track_inner_radius = float(plot_cfg.get("track_inner_radius", DEFAULT_PLOT_STYLE["track_inner_radius"]))

    if model_count <= 0:
        return []
    if track_inner_radius < 0.0:
        raise ValueError("plot.track_inner_radius must be >= 0.")
    if track_inner_radius >= track_outer_radius:
        raise ValueError("plot.track_inner_radius must be smaller than plot.track_outer_radius.")

    available_radius = track_outer_radius - track_inner_radius
    if model_count == 1:
        return [(track_inner_radius, track_outer_radius)]

    total_gap = track_gap * (model_count - 1)
    if total_gap >= available_radius:
        track_gap = available_radius / (model_count * 10.0)
        total_gap = track_gap * (model_count - 1)

    track_thickness = (available_radius - total_gap) / model_count
    if track_thickness <= 0.0:
        raise ValueError("Track layout cannot fit the selected models inside the configured radius.")

    track_ranges = []
    current_outer = track_outer_radius
    for _ in range(model_count):
        current_inner = current_outer - track_thickness
        track_ranges.append((current_inner, current_outer))
        current_outer = current_inner - track_gap
    return track_ranges


def import_circos():
    try:
        from pycirclize import Circos
    except ImportError as e:
        raise RuntimeError(
            "pycirclize is required for circular_map_from_config.py. Install it with `pip install pycirclize`."
        ) from e
    return Circos


def plot_circular_heatmap(
    all_metrics: Dict[str, Dict[str, float]],
    category_map: Dict[str, List[str]],
    label_display: Callable[[str], str],
    model_order: List[str],
    plot_cfg: Dict[str, Any],
    fonts: Dict[str, int],
    output_path: Path,
    metric_title: str,
    group_order: List[str],
) -> None:
    Circos = import_circos()

    models = [model_name for model_name in model_order if model_name in all_metrics]
    if not models:
        print("No models available for plotting after filtering by model_order.")
        return

    available_labels = set().union(*[set(metric_values.keys()) for metric_values in all_metrics.values()]) if all_metrics else set()
    sector_labels: Dict[str, List[str]] = {}
    for group_name in group_order:
        labels = [str(label) for label in category_map.get(group_name, []) if label in available_labels]
        for label in labels:
            sector_labels[label] = [label]

    if not sector_labels:
        print("No overlapping labels found between config category_map and loaded metrics.")
        return

    sectors = {sector_name: len(labels) for sector_name, labels in sector_labels.items()}
    track_ranges = build_track_ranges(len(models), plot_cfg)

    start_angle = float(plot_cfg.get("start_angle", DEFAULT_PLOT_STYLE["start_angle"]))
    end_angle = float(plot_cfg.get("end_angle", DEFAULT_PLOT_STYLE["end_angle"]))
    sector_spacing = float(plot_cfg.get("sector_spacing", DEFAULT_PLOT_STYLE["sector_spacing"]))
    endspace = bool(plot_cfg.get("endspace", DEFAULT_PLOT_STYLE["endspace"]))
    vmin = float(plot_cfg.get("vmin", 0.0))
    vmax = float(plot_cfg.get("vmax", 1.0))
    lower_is_better = bool(plot_cfg.get("lower_is_better", False))
    base_cmap = plt.colormaps.get_cmap(str(plot_cfg.get("cmap", DEFAULT_PLOT_STYLE["cmap"])))
    cmap = (base_cmap.reversed() if lower_is_better else base_cmap).copy()
    missing_color = str(plot_cfg.get("missing_color", DEFAULT_PLOT_STYLE["missing_color"]))
    cmap.set_bad(missing_color)
    color_mapper, colorbar_ticks, colorbar_ticklabels = build_color_normalizer(plot_cfg, vmin, vmax)
    dpi = int(plot_cfg.get("dpi", DEFAULT_PLOT_STYLE["dpi"]))
    figsize = tuple(plot_cfg.get("figsize", DEFAULT_PLOT_STYLE["figsize"]))
    track_pad_ratio = float(plot_cfg.get("track_pad_ratio", DEFAULT_PLOT_STYLE["track_pad_ratio"]))
    rect_edgecolor = str(plot_cfg.get("rect_edgecolor", DEFAULT_PLOT_STYLE["rect_edgecolor"]))
    rect_linewidth = float(plot_cfg.get("rect_linewidth", DEFAULT_PLOT_STYLE["rect_linewidth"]))
    label_margin = float(plot_cfg.get("label_margin", DEFAULT_PLOT_STYLE["label_margin"]))
    label_orientation = str(plot_cfg.get("label_orientation", DEFAULT_PLOT_STYLE["label_orientation"]))
    show_outer_axis = bool(plot_cfg.get("show_outer_axis", DEFAULT_PLOT_STYLE["show_outer_axis"]))
    model_text_prefix = str(plot_cfg.get("model_text_prefix", DEFAULT_PLOT_STYLE["model_text_prefix"]))
    model_text_labels = plot_cfg.get("model_text_labels", {})
    model_text_size = float(fonts["legend"])
    model_text_ha = str(plot_cfg.get("model_text_ha", DEFAULT_PLOT_STYLE["model_text_ha"]))
    model_text_color = str(plot_cfg.get("model_text_color", DEFAULT_PLOT_STYLE["model_text_color"]))
    model_text_deg = float(plot_cfg.get("model_text_deg", DEFAULT_PLOT_STYLE["model_text_deg"]))
    model_text_orientation = str(plot_cfg.get("model_text_orientation", DEFAULT_PLOT_STYLE["model_text_orientation"]))

    plt.rcParams.update({"font.family": "Arial", "font.sans-serif": ["Arial"], "axes.unicode_minus": False})

    circos = Circos(sectors, space=sector_spacing, start=start_angle, end=end_angle, endspace=endspace)
    sector_outer_tracks: List[Tuple[Any, List[str]]] = []

    for sector in circos.sectors:
        labels = sector_labels[sector.name]
        outer_track = None

        for track_index, model_name in enumerate(models):
            metric_values = [all_metrics.get(model_name, {}).get(label, np.nan) for label in labels]
            track = sector.add_track(track_ranges[track_index], r_pad_ratio=track_pad_ratio)
            if track_index == 0 and show_outer_axis:
                track.axis()
            heatmap_values = np.array([metric_values], dtype=float)
            mapped_values = color_mapper(heatmap_values)
            masked_values = np.ma.masked_invalid(mapped_values)
            track.heatmap(
                masked_values,
                cmap=cmap,
                vmin=0.0,
                vmax=1.0,
                rect_kws={"edgecolor": rect_edgecolor, "linewidth": rect_linewidth},
            )
            if track_index == 0:
                outer_track = track

        if outer_track is not None:
            sector_outer_tracks.append((outer_track, labels))

    colorbar_bounds = plot_cfg.get("colorbar_bounds", DEFAULT_PLOT_STYLE["colorbar_bounds"])
    colorbar_labelpad = float(plot_cfg.get("colorbar_labelpad", DEFAULT_PLOT_STYLE["colorbar_labelpad"]))
    colorbar_label_rotation = float(plot_cfg.get("colorbar_label_rotation", DEFAULT_PLOT_STYLE["colorbar_label_rotation"]))

    def add_colorbar(fig, include_labels: bool) -> None:
        colorbar_ax = fig.add_axes(tuple(float(value) for value in colorbar_bounds))
        scalar_mappable = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0.0, vmax=1.0))
        scalar_mappable.set_array([])
        colorbar = fig.colorbar(scalar_mappable, cax=colorbar_ax, orientation="vertical")
        if include_labels:
            colorbar.set_ticks(color_mapper(np.array(colorbar_ticks, dtype=float)).tolist())
            colorbar.set_ticklabels(colorbar_ticklabels)
            if bool(plot_cfg.get("invert_colorbar", False)):
                colorbar.ax.invert_yaxis()
            colorbar.ax.tick_params(labelsize=fonts["cbar_ticks"], colors="black")
            colorbar.set_label(
                metric_title,
                size=fonts["cbar_title"],
                color="black",
                labelpad=colorbar_labelpad,
                rotation=colorbar_label_rotation,
                loc="top",
            )
        else:
            colorbar.set_ticks([])
            colorbar.ax.tick_params(length=0)

    fig = circos.plotfig(dpi=dpi, figsize=figsize)
    fig.patch.set_alpha(0.0)
    add_colorbar(fig, include_labels=False)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_svg = output_path.with_suffix(".svg")
    output_svg_with_text = output_path.with_name(f"{output_path.stem}_with_text.svg")
    fig.savefig(output_svg, format="svg", bbox_inches="tight", transparent=True)
    plt.close(fig)

    for outer_track, labels in sector_outer_tracks:
        outer_track.xticks(
            [index + 0.5 for index in range(len(labels))],
            labels=[label_display(label) for label in labels],
            outer=True,
            tick_length=0,
            label_margin=label_margin,
            label_size=fonts["pathology"],
            label_orientation=label_orientation,
        )

    for track_range, model_name in zip(track_ranges, models):
        circos.text(
            f"{model_text_prefix}{model_text_labels.get(model_name, model_name)}",
            r=(track_range[0] + track_range[1]) / 2,
            deg=model_text_deg,
            orientation=model_text_orientation,
            color=model_text_color,
            ha=model_text_ha,
            va="center",
            size=model_text_size,
        )

    fig = circos.plotfig(dpi=dpi, figsize=figsize)
    fig.patch.set_alpha(0.0)
    add_colorbar(fig, include_labels=True)

    previous_svg_fonttype = plt.rcParams.get("svg.fonttype")
    plt.rcParams["svg.fonttype"] = "none"
    fig.savefig(output_svg_with_text, format="svg", bbox_inches="tight", transparent=True)
    if previous_svg_fonttype is not None:
        plt.rcParams["svg.fonttype"] = previous_svg_fonttype
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight", transparent=True)
    plt.close(fig)

    print(f"Saved: {output_path}")
    print(f"Saved: {output_svg}")
    print(f"Saved: {output_svg_with_text}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate circos heatmaps from config-defined datasets and model mappings.")
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
        help="Datasets to plot. Defaults to config.default_datasets or all configured datasets.",
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
    dataset_keys = list(cfg.get("datasets", {}).keys())

    validate_required_config(cfg)
    output_dir = args.output_dir or Path(cfg["paths"]["default_output_dir"])
    configured_defaults = cfg.get("default_datasets")
    target_datasets = args.datasets if args.datasets else list(configured_defaults or cfg["datasets"].keys())
    model_order = resolve_model_order(cfg)

    for dataset_name in target_datasets:
        dataset_cfg = cfg["datasets"].get(dataset_name)
        if not dataset_cfg:
            print(f"Unknown dataset: {dataset_name}")
            continue

        all_metrics = load_manual_metric_values(dataset_cfg)
        if not all_metrics:
            all_metrics = load_all_model_metrics_for_dataset(cfg, dataset_name)
        if not all_metrics:
            print(f"[{dataset_name}] No model results found.")
            continue

        print(f"[{dataset_name}] Loaded {len(all_metrics)} models: {list(all_metrics.keys())}")
        file_slug = dataset_cfg.get("file_slug", f"circular_map_{dataset_name}")
        output_png = output_dir / f"{file_slug}_circos_heatmap.png"
        plot_cfg = build_plot_config(dataset_name, dataset_cfg)
        metric_title = str(dataset_cfg.get("metric_title", plot_cfg.get("metric_title", "AUROC")))

        try:
            plot_circular_heatmap(
                all_metrics=all_metrics,
                category_map=dataset_cfg.get("category_map", {}),
                label_display=make_label_display(dataset_cfg),
                model_order=model_order,
                plot_cfg=plot_cfg,
                fonts=dataset_fonts(dataset_name),
                output_path=output_png,
                metric_title=metric_title,
                group_order=build_group_order(dataset_cfg),
            )
        except RuntimeError as e:
            print(e)
            return


if __name__ == "__main__":
    main()
