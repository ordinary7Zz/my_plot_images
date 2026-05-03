#!/usr/bin/env python3
"""
Radial (circular) AUROC heatmap from linear-prob `summary.json` results.

Outermost ring = TRACE; innermost = OpenAI CLIP (among models that load successfully).
Pathology caption placement matches the Inspect radial layout (same polar text rules as the
original zero-shot radiomap: outer labels + category arcs).
"""

import argparse
import glob
import json
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

from inspect_label_utils import inspect_category_map, shorten_names_map

# --- Linear-prob result roots (same folder layout as historical `2_linear_prob.py`) ---
CLEAR_BASE = Path("/cbica/home/gongha/202512_CLEAR/ChestCT/clear_results/3_linear_prob")
SOTA_BASE = Path("/cbica/home/gongha/202512_CLEAR/ChestCT/results_sota")

# Model order: outermost ring = TRACE, innermost = OpenAI CLIP
MODEL_ORDER = ["TRACE", "Merlin", "CT-CLIP", "MedSigLIP", "BiomedCLIP", "OpenAI CLIP"]

# (Display Name, folder mapping per dataset, reserved swatch — Wong 2011 subset, Nature-style categorical)
MODEL_CONFIG: List[Tuple[str, Dict[str, str], str]] = [
    ("TRACE", {"ctrate": "densenet121_w32", "inspect": "densenet121_w32", "merlin": "densemamba50_w32_qwen"}, "#E69F00"),
    ("Merlin", {"ctrate": "merlin", "inspect": "merlin", "merlin": "merlin"}, "#56B4E9"),
    ("CT-CLIP", {"ctrate": "ctclip", "inspect": "ctclip", "merlin": "ctclip"}, "#009E73"),
    ("MedSigLIP", {"ctrate": "medsiglip", "inspect": "medsiglip", "merlin": "medsiglip"}, "#0072B2"),
    ("BiomedCLIP", {"ctrate": "biomed_clip", "inspect": "biomed_clip", "merlin": "biomed_clip"}, "#D55E00"),
    ("OpenAI CLIP", {"ctrate": "clip", "inspect": "clip", "merlin": "clip"}, "#CC79A7"),
]

SHORTEN_NAMES = shorten_names_map()
INSPECT_CATEGORY_MAP = inspect_category_map()

CT_RATE_CATEGORY_MAP = {
    "Cardiac": [
        "Cardiomegaly", "Pericardial effusion",
    ],
    "Vascular": [
        "Arterial wall calcification", "Coronary artery wall calcification",
    ],
    "Pleural": [
        "Pleural effusion",
    ],
    "Airways": [
        "Peribronchial thickening", "Bronchiectasis",
    ],
    "Parenchyma": [
        "Emphysema", "Atelectasis", "Consolidation", "Lung opacity", "Lung nodule",
        "Pulmonary fibrotic sequela", "Mosaic attenuation pattern", "Interlobular septal thickening",
    ],
    "Other": ["Medical material", "Lymphadenopathy", "Hiatal hernia"],
}

MERLIN_CATEGORY_MAP = {
    "Cardiac": [
        "aortic_valve_calcification", "coronary_calcification", "cardiomegaly",
    ],
    "Vascular": [
        "atherosclerosis", "thrombosis", "abdominal_aortic_aneurysm",
    ],
    "Pleural": ["pleural_effusion"],
    "Airways": ["atelectasis", "submucosal_edema", "free_air"],
    "Abdominal": [
        "ascites", "hiatal_hernia", "hepatomegaly", "splenomegaly",
        "pancreatic_atrophy", "hepatic_steatosis",
    ],
    "GI / Biliary": [
        "gallstones", "surgically_absent_gallbladder", "appendicitis",
        "bowel_obstruction", "biliary_ductal_dilation",
    ],
    "MSK": ["osteopenia", "fracture"],
    "Renal": ["renal_hypodensities", "renal_cyst", "hydronephrosis"],
    "GU": ["prostatomegaly"],
    "Oncologic": ["metastatic_disease", "lymphadenopathy"],
    "Systemic": ["anasarca"],
}

CATEGORY_COLORS = {
    "Cardiac": "#f9a825", "Respiratory": "#43a047", "Vascular": "#e53935",
    "Renal": "#1e88e5", "Metabolic": "#8e24aa", "Gastrointestinal": "#fb8c00",
    "Hematologic": "#00acc1", "Infectious": "#7cb342", "Malignancy": "#d32f2f",
    "Neurological": "#5e35b1", "Musculoskeletal": "#6d4c41", "Miscellaneous": "#ff7043",
    "Cardiac / Vascular": "#f9a825", "Pleural / Airways": "#43a047",
    "Pleural": "#00897b", "Airways": "#689f38", "Parenchyma": "#1e88e5",
    "Other": "#78909c",
    "Thoracic": "#43a047", "Abdominal / GI": "#fb8c00", "MSK": "#6d4c41",
    "Renal / GU": "#1e88e5", "Oncologic / systemic": "#d32f2f",
    "Abdominal": "#ef6c00", "GI / Biliary": "#fb8c00", "GU": "#7b1fa2",
    "Oncologic": "#c62828", "Systemic": "#455a64",
}

DEFAULT_OUTPUT_DIR = Path(
    "/cbica/home/gongha/202512_CLEAR/ChestCT/clear_results/5_vis/radial_heatmap_lp"
)

LP_DATASETS = ("ctrate", "inspect", "merlin")
DEFAULT_LP_DATASETS = ("ctrate", "inspect")


def _merlin_display_name(path: str) -> str:
    return path.replace("_", " ").title()


def find_summary_json(clear_dir: Path, sota_dir: Path, model_folder: str) -> Optional[Path]:
    path = clear_dir / model_folder / "results_concept_based" / "summary.json"
    if path.exists():
        return path
    path = clear_dir / model_folder / "results_direct" / "summary.json"
    if path.exists():
        return path
    path = clear_dir / model_folder / "results" / "summary.json"
    if path.exists():
        return path
    path = clear_dir / model_folder / "summary.json"
    if path.exists():
        return path

    sota_folder = "clip" if model_folder == "ctclip" and (sota_dir / "clip").exists() else model_folder
    sota_path = sota_dir / sota_folder
    if sota_path.exists():
        runs = sorted(glob.glob(str(sota_path / "run_*")))
        if runs:
            cand = Path(runs[-1]) / "summary.json"
            if cand.exists():
                return cand
        cand = sota_path / "summary.json"
        if cand.exists():
            return cand
    return None


def load_summary_json(clear_dir: Path, sota_dir: Path, model_folder: str) -> Optional[dict]:
    path = find_summary_json(clear_dir, sota_dir, model_folder)
    if not path:
        return None
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        if "individual_aucs" not in data and "predictions" not in data:
            for key in ("inspect", "ctrate", "merlin"):
                if key in data and isinstance(data[key], dict):
                    data = data[key]
                    break
        return data
    except OSError as e:
        print(f"Error loading {path}: {e}")
        return None


def load_lp_all_model_aucs(dataset_name: str) -> Dict[str, Dict[str, float]]:
    clear_dir = CLEAR_BASE / dataset_name
    sota_dir = SOTA_BASE / dataset_name
    if not clear_dir.is_dir():
        print(f"[{dataset_name}] Missing linear-prob dir: {clear_dir}")
        return {}

    out: Dict[str, Dict[str, float]] = {}
    for display_name, folder_map, _ in MODEL_CONFIG:
        model_folder = folder_map.get(dataset_name)
        if not model_folder:
            continue
        data = load_summary_json(clear_dir, sota_dir, model_folder)
        if data and isinstance(data.get("individual_aucs"), dict):
            out[display_name] = dict(data["individual_aucs"])
    return out


def _default_radial_fonts() -> Dict[str, int]:
    """Font sizes for radial heatmap (Inspect baseline)."""
    return {
        "category": 18,
        "cbar_title": 18,
        "cbar_ticks": 15,
        "legend": 18,
        "pathology": 17,
    }


def _merlin_radial_fonts() -> Dict[str, int]:
    """Merlin: long snake_case captions — bump over Inspect default."""
    return {
        "category": 21,
        "cbar_title": 21,
        "cbar_ticks": 16,
        "legend": 21,
        "pathology": 20,
    }


def _lp_dataset_configs() -> Dict[str, dict]:
    # CT-RATE: longer pathology names — bump over default
    ctrate_radial_fonts = {
        "category": 24,
        "cbar_title": 24,
        "cbar_ticks": 18,
        "legend": 24,
        "pathology": 22,
    }
    return {
        "ctrate": {
            "file_slug": "lp_radiomap_ctrate",
            "title": "CT-RATE test set (linear prob)",
            "category_map": CT_RATE_CATEGORY_MAP,
            "label_display": lambda p: SHORTEN_NAMES.get(p, p),
            "radial_fonts": ctrate_radial_fonts,
        },
        "inspect": {
            "file_slug": "lp_radiomap_inspect",
            "title": "Inspect test set (linear prob)",
            "category_map": INSPECT_CATEGORY_MAP,
            "label_display": lambda p: SHORTEN_NAMES.get(p, p),
        },
        "merlin": {
            "file_slug": "lp_radiomap_merlin",
            "title": "Merlin test set (linear prob)",
            "category_map": MERLIN_CATEGORY_MAP,
            "label_display": lambda p: SHORTEN_NAMES.get(p, _merlin_display_name(p)),
            "radial_fonts": _merlin_radial_fonts(),
        },
    }


def build_pathology_order(category_map: Dict[str, List[str]]) -> List[Tuple[str, str]]:
    return [(p, cat) for cat, plist in category_map.items() for p in plist]


def plot_radial_heatmap(
    all_aucs: dict,
    category_map: Dict[str, List[str]],
    label_display: Callable[[str], str],
    output_path: Path,
    radial_fonts: Optional[Dict[str, int]] = None,
) -> None:
    """
    Polar AUROC heatmap. Caption rules (rotation, ha, label_r margin) match the Inspect
    zero-shot / linear-prob radial figures: outer pathology text + inner category arcs.
    """
    fonts = {**_default_radial_fonts(), **(radial_fonts or {})}
    order = build_pathology_order(category_map)
    pathologies = [p for p, _ in order]
    categories = [c for _, c in order]
    available = set().union(*[set(v.keys()) for v in all_aucs.values()])
    keep_idx = [i for i, p in enumerate(pathologies) if p in available]
    if not keep_idx:
        print("No overlapping pathologies found.")
        return
    pathologies = [pathologies[i] for i in keep_idx]
    categories = [categories[i] for i in keep_idx]
    n_path = len(pathologies)

    models = [m for m in MODEL_ORDER if m in all_aucs]
    n_models = len(models)
    models_draw = list(reversed(models))
    r_inner, r_per_ring = 3.0, 0.5
    r_outer = r_inner + n_models * r_per_ring

    gap_units = 1.0
    n_gaps = sum(1 for i in range(1, n_path) if categories[i] != categories[i - 1]) + 1
    total_units = n_path + n_gaps * gap_units
    dtheta = 2 * np.pi / total_units
    centers, unit = [], gap_units / 2
    for i in range(n_path):
        if i > 0 and categories[i] != categories[i - 1]:
            unit += gap_units
        centers.append((unit + 0.5) * dtheta)
        unit += 1
    centers = np.array(centers)

    # Purple sequential (ColorBrewer Purples): higher AUROC → darker purple (better)
    cmap = plt.cm.Purples
    vmin, vmax = 0.0, 1.0
    fig, ax = plt.subplots(subplot_kw=dict(projection="polar"), figsize=(20, 20))
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)

    for k, model in enumerate(models_draw):
        aucs = all_aucs[model]
        r0 = r_inner + k * r_per_ring
        for i, path in enumerate(pathologies):
            auroc = aucs.get(path, np.nan)
            color = "#E0E0E0" if np.isnan(auroc) else cmap((auroc - vmin) / (vmax - vmin))
            ax.bar(
                centers[i], r_per_ring, width=dtheta, bottom=r0,
                color=color, edgecolor="white", linewidth=1.5,
            )

    bounds = [0] + [i for i in range(1, len(categories)) if categories[i] != categories[i - 1]] + [n_path]
    for j in range(len(bounds) - 1):
        s, e = bounds[j], bounds[j + 1]
        cat = categories[s]
        th_l = centers[s] - dtheta / 2
        th_r = centers[e - 1] + dtheta / 2
        arc = np.linspace(th_l, th_r, 60)
        ax.plot(arc, np.full_like(arc, r_inner - 0.1), color=CATEGORY_COLORS.get(cat, "#888"), lw=6)
        th_mid = 0.5 * (th_l + th_r)
        deg = np.degrees(th_mid)
        if 0 <= deg <= 180:
            rot = 90 - deg
            ha = "right"
        else:
            rot = 270 - deg
            ha = "left"
        ax.text(
            th_mid, r_inner - 0.4, cat, ha=ha, va="center", rotation=rot,
            fontsize=fonts["category"], color="black", rotation_mode="anchor",
        )

    cax = fig.add_axes([0.88, 0.75, 0.015, 0.15])
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax))
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cax)
    cbar.ax.set_title("AUROC", fontsize=fonts["cbar_title"], pad=15)
    cbar.ax.tick_params(labelsize=fonts["cbar_ticks"])

    handles = []
    for idx, model in enumerate(models):
        color_val = 0.9 - idx * (0.4 / max(1, len(models) - 1))
        handles.append(mpatches.Patch(facecolor=cmap(color_val), edgecolor="white", lw=1, label=model))

    fig.legend(
        handles=handles, loc="upper left", bbox_to_anchor=(0.04, 0.92),
        frameon=False, fontsize=fonts["legend"], handlelength=1.5, handleheight=1.5,
    )

    # Outer pathology captions — same radial offset / rotation as Inspect radiomap
    label_r = r_outer + 0.1
    for i, path in enumerate(pathologies):
        theta = centers[i]
        deg = np.degrees(theta)
        if 0 <= deg <= 180:
            rot = 90 - deg
            ha = "left"
        else:
            rot = 270 - deg
            ha = "right"
        display_path = label_display(path)
        ax.text(
            theta, label_r, display_path,
            rotation=rot, ha=ha, va="center",
            fontsize=fonts["pathology"], rotation_mode="anchor", clip_on=False,
        )

    ax.set_ylim(0, label_r + 2.5)
    ax.set_yticks([])
    ax.set_xticks([])
    ax.spines["polar"].set_visible(False)

    fig.savefig(output_path, dpi=600, bbox_inches="tight")
    fig.savefig(output_path.with_suffix(".pdf"), dpi=600, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")
    print(f"Saved: {output_path.with_suffix('.pdf')}")


def run_one_dataset(dataset_key: str, cfg: dict, output_dir: Path) -> None:
    all_aucs = load_lp_all_model_aucs(dataset_key)
    if not all_aucs:
        print(f"[{dataset_key}] No linear-prob model results found.")
        return
    print(f"[{dataset_key}] Loaded {len(all_aucs)} models: {list(all_aucs.keys())}")
    out_png = output_dir / f"{cfg['file_slug']}_radial_heatmap.png"
    plot_radial_heatmap(
        all_aucs,
        cfg["category_map"],
        cfg["label_display"],
        out_png,
        radial_fonts=cfg.get("radial_fonts"),
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Radial AUROC heatmaps from linear-prob summaries (self-contained)."
    )
    parser.add_argument(
        "--datasets",
        nargs="*",
        default=list(DEFAULT_LP_DATASETS),
        choices=list(LP_DATASETS),
        help=(
            "Linear-prob test sets to plot (default: ctrate inspect). "
            "Use e.g. --datasets ctrate inspect merlin for all three."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory for PNG/PDF (default: {DEFAULT_OUTPUT_DIR}).",
    )
    args = parser.parse_args()
    configs = _lp_dataset_configs()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name in args.datasets:
        run_one_dataset(name, configs[name], args.output_dir)


if __name__ == "__main__":
    main()
