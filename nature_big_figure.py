#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")

import numpy as np

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.gridspec import GridSpec


DEFAULT_PANEL_LAYOUT: List[Tuple[str, str]] = [
    ("A", "cls_auroc_circos_heatmap.png"),
    ("B", "cls_auprc_circos_heatmap.png"),
    ("C", "seg_dice_circos_heatmap.png"),
    ("D", "seg_hd95_circos_heatmap.png"),
    ("E", "BM_auroc.png"),
    ("F", "FTCPTC_auroc.png"),
    ("G", "LNMCN01_auroc.png"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compose a Nature-style multi-panel figure from existing PNGs.")
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path(__file__).with_name("out"),
        help="Directory containing the input PNG files.",
    )
    parser.add_argument(
        "--output-basename",
        type=str,
        default="nature_big_figure",
        help="Output file basename written into the input directory.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=600,
        help="DPI for the exported figure.",
    )
    return parser.parse_args()


def load_image(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Missing image: {path}")

    image = mpimg.imread(path)
    rgb = image[..., :3]
    mask = np.any(rgb < 0.995, axis=-1)
    if mask.any():
        rows = np.where(mask.any(axis=1))[0]
        cols = np.where(mask.any(axis=0))[0]
        top = max(rows[0], 0)
        bottom = min(rows[-1], image.shape[0])
        left = max(cols[0], 0)
        right = min(cols[-1], image.shape[1])
        image = image[top:bottom, left:right]
    return image


def style_matplotlib() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "axes.unicode_minus": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )


def scale_axes(ax, scale: float) -> None:
    pos = ax.get_position()
    cx = pos.x0 + pos.width / 2
    cy = pos.y0 + pos.height / 2
    width = pos.width * scale
    height = pos.height * scale
    ax.set_position([cx - width / 2, cy - height / 2, width, height])


def add_panel(ax, image, panel_label: str, anchor: str = "C") -> None:
    ax.set_anchor(anchor)
    ax.imshow(image)
    ax.set_axis_off()
    ax.text(
        0.02,
        0.98,
        panel_label,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=18,
        fontweight="bold",
        color="black",
        family="serif",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.82, "pad": 1.8},
    )


def build_figure(panel_layout: Sequence[Tuple[str, str]], input_dir: Path):
    images = [(label, load_image(input_dir / filename)) for label, filename in panel_layout]

    fig = plt.figure(figsize=(16.0, 12), facecolor="white")
    outer = GridSpec(2, 12, figure=fig, height_ratios=[1.15, 1.0], hspace=0, wspace=0.0)

    top_axes = [
        fig.add_subplot(outer[0, 0:3]),
        fig.add_subplot(outer[0, 3:6]),
        fig.add_subplot(outer[0, 6:9]),
        fig.add_subplot(outer[0, 9:12]),
    ]
    bottom_axes = [
        fig.add_subplot(outer[1, 0:4]),
        fig.add_subplot(outer[1, 4:8]),
        fig.add_subplot(outer[1, 8:12]),
    ]

    for ax, (panel_label, image) in zip(top_axes, images[:4]):
        add_panel(ax, image, panel_label, anchor="S")

    for ax, (panel_label, image) in zip(bottom_axes, images[4:]):
        add_panel(ax, image, panel_label, anchor="S")
        scale_axes(ax, 0.90)

    return fig


def save_figure(fig, output_base: Path, dpi: int) -> None:
    png_path = output_base.with_suffix(".png")
    fig.savefig(png_path, dpi=dpi, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)
    print(f"Saved {png_path}")


def main() -> int:
    args = parse_args()
    style_matplotlib()

    input_dir = args.input_dir
    output_base = input_dir / args.output_basename

    fig = build_figure(DEFAULT_PANEL_LAYOUT, input_dir)
    save_figure(fig, output_base, args.dpi)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
