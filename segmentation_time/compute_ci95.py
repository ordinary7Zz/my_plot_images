#!/usr/bin/env python3
"""
读取 dice_hd95_results.csv，用 Bootstrap 计算 Dice / HD95 的 95% CI。
- 各组均值的 Bootstrap 95% CI
- 配对差异 (Assisted - Manual) 的 Bootstrap 95% CI（差异 CI 不含 0 即显著）
- 附 t 分布 CI 作为对照
"""
import csv
from pathlib import Path

import numpy as np
from scipy import stats as scipy_stats


def load_csv(path):
    md, mh, ad, ah = [], [], [], []
    with open(path) as f:
        for r in csv.DictReader(f):
            md.append(float(r["manual_dice"]))
            mh.append(float(r["manual_hd95"]))
            ad.append(float(r["assisted_dice"]))
            ah.append(float(r["assisted_hd95"]))
    return md, mh, ad, ah


def boot_mean_ci(data, rng, n_boot=10000, ci=95):
    data = np.asarray(data, dtype=float)
    n = len(data)
    boots = np.empty(n_boot)
    for i in range(n_boot):
        boots[i] = rng.choice(data, size=n, replace=True).mean()
    lo = np.percentile(boots, (100 - ci) / 2)
    hi = np.percentile(boots, 100 - (100 - ci) / 2)
    return float(lo), float(hi)


def boot_diff_ci(a, b, rng, n_boot=10000, ci=95):
    """配对差异 (a - b) 均值的 Bootstrap CI。"""
    a, b = np.asarray(a, float), np.asarray(b, float)
    n = len(a)
    diffs = a - b
    boots = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.choice(n, size=n, replace=True)
        boots[i] = diffs[idx].mean()
    lo = np.percentile(boots, (100 - ci) / 2)
    hi = np.percentile(boots, 100 - (100 - ci) / 2)
    return float(lo), float(hi)


def t_ci(data):
    data = np.asarray(data, float)
    n = len(data)
    m, se = data.mean(), data.std(ddof=1) / np.sqrt(n)
    t = scipy_stats.t.ppf(0.975, n - 1)
    return float(m - t * se), float(m + t * se)


def fmt(x, d=4):
    return f"{x:.{d}f}"


def main():
    csv_path = Path(__file__).parent / "dice_hd95_results.csv"
    md, mh, ad, ah = load_csv(str(csv_path))
    rng = np.random.default_rng(42)  # 固定种子, 可复现
    out_csv = Path(__file__).parent / "ci95_results.csv"

    rows = []  # 用于写文件

    print(f"n = {len(md)} paired images\n")

    # ---- 各组均值 CI ----
    print("=" * 78)
    print(f"{'指标':<10} {'模式':<12} {'mean':>8} {'Boot 95% CI':>22} {'t 95% CI':>22}")
    print("-" * 78)
    for name, data, d in [
        ("Dice", md, 4), ("Dice", ad, 4),
        ("HD95", mh, 2), ("HD95", ah, 2),
    ]:
        m = np.mean(data)
        blo, bhi = boot_mean_ci(data, rng)
        tlo, thi = t_ci(data)
        label = "Manual" if data in (md, mh) else "AI-Assisted"
        print(f"{name:<10} {label:<12} {fmt(m, d):>8} "
              f"[{fmt(blo, d)}, {fmt(bhi, d)}]   [{fmt(tlo, d)}, {fmt(thi, d)}]")
        rows.append({
            "type": "mean", "metric": name, "mode": label,
            "mean": fmt(m, d),
            "boot_ci_low": fmt(blo, d), "boot_ci_high": fmt(bhi, d),
            "t_ci_low": fmt(tlo, d), "t_ci_high": fmt(thi, d),
            "significant": "", "note": "",
        })

    # ---- 配对差异 CI (Assisted - Manual) ----
    print("\n" + "=" * 78)
    print("配对差异 (AI-Assisted - Manual) 的均值 Bootstrap 95% CI")
    print("-" * 78)
    for name, a, b, d, better in [
        ("Dice", ad, md, 4, "正=AI更好"),
        ("HD95", ah, mh, 2, "负=AI更好"),
    ]:
        diff_mean = np.mean(np.asarray(a) - np.asarray(b))
        dlo, dhi = boot_diff_ci(a, b, rng)
        contains_zero = dlo <= 0 <= dhi
        sig = "不显著(含0)" if contains_zero else "显著(不含0) ★"
        print(f"{name:<6} diff mean = {fmt(diff_mean, d):>8}   "
              f"95% CI [{fmt(dlo, d)}, {fmt(dhi, d)}]   {sig}   ({better})")
        rows.append({
            "type": "diff(AI-Manual)", "metric": name, "mode": "",
            "mean": fmt(diff_mean, d),
            "boot_ci_low": fmt(dlo, d), "boot_ci_high": fmt(dhi, d),
            "t_ci_low": "", "t_ci_high": "",
            "significant": "yes" if not contains_zero else "no",
            "note": better,
        })

    print("\n注: Bootstrap n_boot=10000, seed=42; t-CI 为 Student t 分布对照。")

    # ---- 写 CSV ----
    import csv as _csv
    with open(out_csv, "w", newline="") as f:
        w = _csv.DictWriter(f, fieldnames=[
            "type", "metric", "mode", "mean",
            "boot_ci_low", "boot_ci_high",
            "t_ci_low", "t_ci_high", "significant", "note"])
        w.writeheader()
        w.writerows(rows)
    print(f"\nCI 结果已保存 -> {out_csv}")


if __name__ == "__main__":
    main()
