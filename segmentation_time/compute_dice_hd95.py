#!/usr/bin/env python3
"""
计算 Manual / AI-Assisted mask 与 GT 之间的 Dice 和 HD95。
读取 experiment_log.csv 配对 manual / assisted，逐图计算并输出 CSV + 汇总统计。
HD95 单位为像素 (2D 图像无物理间距信息)。
"""
import csv
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from scipy import stats as scipy_stats
from scipy.ndimage import binary_erosion, distance_transform_edt


# ---------- mask 读取 ----------
def load_mask(path: str, target_shape: Optional[Tuple[int, int]] = None) -> np.ndarray:
    mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise ValueError(f"Cannot read: {path}")
    if target_shape and mask.shape[:2] != target_shape:
        mask = cv2.resize(mask, (target_shape[1], target_shape[0]))
    return mask > 128


# ---------- 指标 ----------
def compute_dice(pred: np.ndarray, gt: np.ndarray) -> float:
    inter = np.logical_and(pred, gt).sum()
    total = pred.sum() + gt.sum()
    return (2.0 * inter) / total if total > 0 else 0.0


def __surface_distances(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    """pred 表面点到 gt 表面的最短距离数组。"""
    pred_border = pred ^ binary_erosion(pred)
    gt_border = gt ^ binary_erosion(gt)
    if pred_border.sum() == 0 or gt_border.sum() == 0:
        return np.array([])
    dt = distance_transform_edt(~gt_border)
    return dt[pred_border]


def compute_hd95(pred: np.ndarray, gt: np.ndarray) -> float:
    if pred.sum() == 0 or gt.sum() == 0:
        return float("nan")
    d_p2g = __surface_distances(pred, gt)
    d_g2p = __surface_distances(gt, pred)
    if len(d_p2g) == 0 or len(d_g2p) == 0:
        return float("nan")
    return float(max(np.percentile(d_p2g, 95), np.percentile(d_g2p, 95)))


# ---------- 路径查找 ----------
def find_corrected_mask(mask_dir: Path, image_name: str, annotator: str, mode: str) -> Optional[str]:
    stem = Path(image_name).stem
    p = mask_dir / f"{stem}_{annotator}_{mode}_corrected.png"
    return str(p) if p.exists() else None


def find_gt_mask(gt_dir: Path, image_name: str) -> Optional[str]:
    stem = Path(image_name).stem
    for ext in (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"):
        p = gt_dir / f"{stem}{ext}"
        if p.exists():
            return str(p)
    return None


def load_log(log_path: str) -> List[Dict]:
    with open(log_path, "r") as f:
        return list(csv.DictReader(f))


# ---------- 配对 + 批量计算 ----------
def compute_paired(log_rows, mask_dir, gt_dir):
    mask_dir, gt_dir = Path(mask_dir), Path(gt_dir)
    pairs: Dict[Tuple[str, str], Dict[str, dict]] = {}
    for row in log_rows:
        if row.get("finished") != "True":
            continue
        key = (row["annotator"], row["image_name"])
        pairs.setdefault(key, {})[row["mode"]] = row

    results = []
    for (annotator, image_name), modes in sorted(pairs.items()):
        if "manual" not in modes or "assisted" not in modes:
            continue
        gt_path = find_gt_mask(gt_dir, image_name)
        if gt_path is None:
            continue
        gt = load_mask(gt_path)

        item = {"annotator": annotator, "image_name": image_name}
        for mode in ("manual", "assisted"):
            mp = find_corrected_mask(mask_dir, image_name, annotator, mode)
            if mp is None:
                item[f"{mode}_dice"] = None
                item[f"{mode}_hd95"] = None
            else:
                pred = load_mask(mp, target_shape=gt.shape)
                item[f"{mode}_dice"] = compute_dice(pred, gt)
                item[f"{mode}_hd95"] = compute_hd95(pred, gt)
        if item.get("manual_dice") is not None and item.get("assisted_dice") is not None:
            results.append(item)
    return results


# ---------- 汇总 + 检验 ----------
def summarize(vals, name):
    vals = np.array([v for v in vals if v is not None and not (isinstance(v, float) and np.isnan(v))])
    if len(vals) == 0:
        return f"{name}: N/A"
    mean, std = np.mean(vals), np.std(vals, ddof=1) if len(vals) > 1 else 0.0
    med = np.median(vals)
    return f"{name}: n={len(vals)}  mean={mean:.4f}  std={std:.4f}  median={med:.4f}  min={vals.min():.4f}  max={vals.max():.4f}"


def main():
    base = Path(__file__).parent
    log = base / "experiment_single" / "experiment_log.csv"
    mask_dir = base / "experiment_single" / "masks"
    gt_dir = base / "datasets" / "gt"
    out_csv = base / "dice_hd95_results.csv"

    rows = load_log(str(log))
    print(f"Loaded {len(rows)} log entries")
    results = compute_paired(rows, mask_dir, gt_dir)
    print(f"Paired images: {len(results)}")

    # 输出 CSV
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["image_name", "manual_dice", "manual_hd95",
                    "assisted_dice", "assisted_hd95"])
        for r in results:
            w.writerow([r["image_name"],
                        f"{r['manual_dice']:.6f}", f"{r['manual_hd95']:.4f}",
                        f"{r['assisted_dice']:.6f}", f"{r['assisted_hd95']:.4f}"])
    print(f"\nCSV -> {out_csv}")

    manual_dice = [r["manual_dice"] for r in results]
    assisted_dice = [r["assisted_dice"] for r in results]
    manual_hd95 = [r["manual_hd95"] for r in results]
    assisted_hd95 = [r["assisted_hd95"] for r in results]

    print("\n========== DICE ==========")
    print(summarize(manual_dice, "Manual   "))
    print(summarize(assisted_dice, "Assisted "))

    print("\n========== HD95 (pixels) ==========")
    print(summarize(manual_hd95, "Manual   "))
    print(summarize(assisted_hd95, "Assisted "))

    # Wilcoxon 检验
    n = len(results)
    if n >= 5:
        print("\n========== Wilcoxon signed-rank ==========")
        for name, m, a in [("Dice", manual_dice, assisted_dice),
                           ("HD95", manual_hd95, assisted_hd95)]:
            try:
                _, p = scipy_stats.wilcoxon(m, a, alternative="two-sided")
                star = " ***" if p < 0.001 else " **" if p < 0.01 else " *" if p < 0.05 else " n.s."
                print(f"{name:6s}: p = {p:.4f}{star}")
            except Exception as e:
                print(f"{name:6s}: Wilcoxon failed ({e})")


if __name__ == "__main__":
    main()
