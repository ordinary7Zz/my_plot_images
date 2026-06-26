#!/usr/bin/env python3
"""
实验运行脚本 — 人工 vs AI辅助 分割时间效率对比
====================================================
功能:
  1. 配置实验参数 (图像列表、标注者、分组等)
  2. 交叉设计: 一半人先手动后辅助, 另一半反过来
  3. 随机打乱图像顺序, 顺序启动标注工具
  4. 自动收集所有日志, 统一汇总

使用方式:
  # 生成实验配置
  python run_experiment.py --generate-config --image-dir ./images --mask-dir ./masks \
      --annotators alice bob --output-dir ./experiment_001

  # 运行实验 (交互式, 逐个启动标注任务)
  python run_experiment.py --config experiment_001/experiment_config.json --run

依赖: 本目录下的 annotator.py
"""

import os
import sys
import json
import random
import argparse
import subprocess
import csv
import shutil
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
from collections import defaultdict


# ============================================================
# 配置生成
# ============================================================

def find_images_in_dir(image_dir: str, exts=(".png", ".jpg", ".jpeg", ".bmp")):
    """查找目录下的图像文件"""
    image_dir = Path(image_dir)
    images = []
    for ext in exts:
        images.extend(sorted(image_dir.glob(f"*{ext}")))
        images.extend(sorted(image_dir.glob(f"*{ext.upper()}")))
    return sorted(set(images))


def generate_experiment_config(
    image_dir: str,
    mask_dir: str,
    output_dir: str,
    annotators: List[str],
    num_images: int = 30,
    random_seed: int = 42,
    cross_over: bool = True,
) -> Dict:
    """
    生成实验配置

    交叉设计 (cross_over=True):
      每位标注者对每张图只做一种模式
      一半图做手动, 一半图做辅助
      不同标注者之间, 同一张图的模式可以不同 (如 Alice 对手动, Bob 对辅助)

      这样保证: 每个标注者都做过两种模式, 且同一张图不会同被一个标注者做两次
    """
    random.seed(random_seed)

    images = find_images_in_dir(image_dir)
    if num_images and num_images < len(images):
        images = random.sample(images, num_images)

    if not images:
        raise ValueError(f"No images found in {image_dir}")

    # 匹配 mask
    mask_dir_path = Path(mask_dir)
    image_mask_pairs = []
    for img_path in images:
        stem = img_path.stem
        # 尝试多种 mask 命名方式
        candidates = [
            mask_dir_path / f"{stem}_mask.png",
            mask_dir_path / f"{stem}.png",
            mask_dir_path / f"{stem}_pred.png",
        ]
        found = None
        for c in candidates:
            if c.exists():
                found = str(c)
                break
        image_mask_pairs.append({
            "image": str(img_path),
            "mask": found,  # None 如果没有预分割 mask
        })

    # 对每位标注者, 随机分配图像到手动组和辅助组
    total = len(image_mask_pairs)
    task_list = []

    for annotator in annotators:
        # 打乱图像顺序
        shuffled = image_mask_pairs.copy()
        random.shuffle(shuffled)

        if cross_over:
            half = total // 2
            manual_group = shuffled[:half]
            assisted_group = shuffled[half:]

            # 第一轮: 手动组
            for item in manual_group:
                task_list.append({
                    "annotator": annotator,
                    "round": 1,
                    "mode": "manual",
                    "image": item["image"],
                    "mask": None,  # 纯手动不用 mask
                })
            # 第二轮: 辅助组
            for item in assisted_group:
                task_list.append({
                    "annotator": annotator,
                    "round": 2,
                    "mode": "assisted",
                    "image": item["image"],
                    "mask": item["mask"],
                })
        else:
            # 简单分组: 所有图像都做手动, 所有图像都做辅助
            for item in shuffled:
                task_list.append({
                    "annotator": annotator,
                    "round": 1,
                    "mode": "manual",
                    "image": item["image"],
                    "mask": None,
                })
            # 打乱后第二轮
            shuffled2 = image_mask_pairs.copy()
            random.shuffle(shuffled2)
            for item in shuffled2:
                task_list.append({
                    "annotator": annotator,
                    "round": 2,
                    "mode": "assisted",
                    "image": item["image"],
                    "mask": item["mask"],
                })

    # 保存配置
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    config = {
        "experiment_id": output_dir.name,
        "created_at": datetime.now().isoformat(),
        "image_dir": os.path.abspath(image_dir),
        "mask_dir": os.path.abspath(mask_dir),
        "output_dir": str(output_dir),
        "annotators": annotators,
        "num_images_total": len(image_mask_pairs),
        "cross_over": cross_over,
        "random_seed": random_seed,
        "tasks": task_list,
        "task_count": len(task_list),
        "log_file": str(output_dir / "experiment_log.csv"),
        "status": "configured",
    }

    config_path = output_dir / "experiment_config.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    # 汇总打印
    print("=" * 60)
    print(f"  Experiment Config Generated")
    print(f"  ID:       {config['experiment_id']}")
    print(f"  Images:   {config['num_images_total']}")
    print(f"  Annotators: {', '.join(annotators)}")
    print(f"  Tasks:    {config['task_count']} total")
    print(f"  Cross-over: {cross_over}")
    print("-" * 60)
    for ann in annotators:
        manual_n = sum(1 for t in task_list if t["annotator"] == ann and t["mode"] == "manual")
        assist_n = sum(1 for t in task_list if t["annotator"] == ann and t["mode"] == "assisted")
        print(f"    {ann}: {manual_n} manual + {assist_n} assisted")
    print(f"  Config:   {config_path}")
    print("=" * 60)

    return config


# ============================================================
# 实验运行
# ============================================================

def load_config(config_path: str) -> Dict:
    """加载实验配置"""
    with open(config_path, "r") as f:
        return json.load(f)


def save_config(config: Dict, config_path: str):
    """保存实验配置"""
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)


def get_annotator_script() -> str:
    """获取 annotator.py 的绝对路径"""
    return str(Path(__file__).parent / "annotator.py")


def get_pending_tasks(config: Dict, annotator: Optional[str] = None) -> List[Dict]:
    """获取待完成的标注任务"""
    completed_images = set()
    log_file = config.get("log_file", "")
    if log_file and os.path.exists(log_file):
        with open(log_file, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get("finished") == "True":
                    key = (row.get("annotator", ""), row.get("image_name", ""), row.get("mode", ""))
                    completed_images.add(key)

    pending = []
    for task in config["tasks"]:
        img_name = os.path.basename(task["image"])
        key = (task["annotator"], img_name, task["mode"])
        if key not in completed_images:
            pending.append(task)

    if annotator:
        pending = [t for t in pending if t["annotator"] == annotator]

    return pending


def run_single_task(task: Dict, config: Dict):
    """运行单个标注任务 (启动 annotator.py)"""
    img_name = os.path.basename(task["image"])

    # 输出 mask 路径
    output_dir = Path(config["output_dir"]) / "masks"
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(img_name).stem
    output_mask = str(output_dir / f"{stem}_{task['annotator']}_{task['mode']}_corrected.png")

    # 构建命令
    cmd = [
        sys.executable, get_annotator_script(),
        "--image", task["image"],
        "--annotator", task["annotator"],
        "--output-log", config["log_file"],
        "--output-mask", output_mask,
    ]
    if task.get("mask"):
        cmd.extend(["--mask", task["mask"]])

    print(f"\n{'='*60}")
    print(f"  TASK")
    print(f"  Annotator: {task['annotator']}")
    print(f"  Round:     {task['round']}")
    print(f"  Mode:      {task['mode']}")
    print(f"  Image:     {img_name}")
    print(f"{'='*60}")
    print(f"\n  [LAUNCHING annotator...]\n")

    subprocess.run(cmd)


def run_experiment_interactive(config_path: str, annotator_filter: Optional[str] = None):
    """
    交互式运行实验

    逐个显示待完成任务, 标注者按 Enter 开始下一个
    支持中途退出和断点续传
    """
    config = load_config(config_path)
    pending = get_pending_tasks(config, annotator_filter)

    total_pending = len(pending)

    if total_pending == 0:
        print("\n  All tasks completed! ")
        return

    # 按轮次和标注者分组显示
    by_annotator = defaultdict(list)
    for t in pending:
        by_annotator[t["annotator"]].append(t)

    print(f"\n{'='*60}")
    print(f"  EXPERIMENT: {config['experiment_id']}")
    print(f"  Pending tasks: {total_pending}")
    print(f"{'='*60}")
    for ann in config["annotators"]:
        n = len(by_annotator.get(ann, []))
        n_manual = sum(1 for t in by_annotator.get(ann, []) if t["mode"] == "manual")
        n_assist = sum(1 for t in by_annotator.get(ann, []) if t["mode"] == "assisted")
        status = f"{n} remaining" if n > 0 else "DONE"
        print(f"    {ann}: {status} ({n_manual} manual + {n_assist} assisted)")

    if annotator_filter:
        print(f"\n  Filtered to annotator: {annotator_filter}")

    # 逐个运行
    for i, task in enumerate(pending):
        print(f"\n\n{'#'*60}")
        print(f"  Task {i+1}/{total_pending}")
        print(f"  Annotator: {task['annotator']} | Round: {task['round']} | Mode: {task['mode']}")
        print(f"  Press ENTER to start, or type 'skip'/'q' to quit")
        print(f"{'#'*60}")

        user_input = input("  > ").strip().lower()
        if user_input in ("q", "quit", "exit"):
            print("\n  Experiment paused. Run again to continue.")
            break
        elif user_input == "skip":
            print(f"  Skipped: {os.path.basename(task['image'])}")
            continue

        run_single_task(task, config)

    # 最终状态
    remaining = len(get_pending_tasks(config))
    print(f"\n{'='*60}")
    if remaining == 0:
        print(f"  ALL TASKS COMPLETE!")
    else:
        print(f"  {remaining} tasks remaining.")
    print(f"{'='*60}")


# ============================================================
# 断点续传: 显示进度
# ============================================================

def show_progress(config_path: str):
    """显示实验进度"""
    config = load_config(config_path)
    total = len(config["tasks"])
    pending = get_pending_tasks(config)
    completed = total - len(pending)

    print(f"\n  Experiment: {config['experiment_id']}")
    print(f"  Progress:   {completed}/{total} ({100*completed/total:.0f}%)")
    print(f"  Cross-over: {config['cross_over']}")
    print()

    for ann in config["annotators"]:
        ann_total = sum(1 for t in config["tasks"] if t["annotator"] == ann)
        ann_pending = get_pending_tasks(config, ann)
        ann_done = ann_total - len(ann_pending)
        print(f"  {ann}: {ann_done}/{ann_total}")

    # 汇总已完成的时间数据
    log_file = config.get("log_file", "")
    if log_file and os.path.exists(log_file):
        stats = defaultdict(list)
        with open(log_file, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                mode = row.get("mode", "")
                t = float(row.get("time_seconds", 0))
                if t > 0:
                    stats[mode].append(t)

        if stats:
            print(f"\n  Timing preview:")
            for mode in ["manual", "assisted"]:
                if stats[mode]:
                    import statistics
                    mean_t = statistics.mean(stats[mode])
                    median_t = statistics.median(stats[mode])
                    print(f"    {mode:10s}: mean={mean_t:.1f}s, median={median_t:.1f}s, n={len(stats[mode])}")


# ============================================================
# CLI
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="人工 vs AI辅助 分割时间效率对比 实验运行脚本"
    )
    sub = parser.add_subparsers(dest="command")

    # ---- generate-config ----
    gen = sub.add_parser("generate-config", help="生成实验配置")
    gen.add_argument("--image-dir", required=True, help="原始图像目录")
    gen.add_argument("--mask-dir", required=True, help="预分割 mask 目录")
    gen.add_argument("--output-dir", required=True, help="实验输出目录")
    gen.add_argument("--annotators", nargs="+", required=True,
                     help="标注者 ID 列表 (如 alice bob)")
    gen.add_argument("--num-images", type=int, default=30,
                     help="随机选取的图像数量 (默认 30)")
    gen.add_argument("--seed", type=int, default=42,
                     help="随机种子")
    gen.add_argument("--no-cross-over", action="store_true",
                     help="不使用交叉设计 (每张图会做两次)")

    # ---- run ----
    run_p = sub.add_parser("run", help="运行实验 (交互式)")
    run_p.add_argument("--config", required=True, help="实验配置 JSON 路径")
    run_p.add_argument("--annotator", default=None,
                       help="只运行特定标注者的任务")

    # ---- status ----
    stat = sub.add_parser("status", help="查看实验进度")
    stat.add_argument("--config", required=True, help="实验配置 JSON 路径")

    args = parser.parse_args()

    if args.command == "generate-config":
        generate_experiment_config(
            image_dir=args.image_dir,
            mask_dir=args.mask_dir,
            output_dir=args.output_dir,
            annotators=args.annotators,
            num_images=args.num_images,
            random_seed=args.seed,
            cross_over=not args.no_cross_over,
        )

    elif args.command == "run":
        run_experiment_interactive(args.config, args.annotator)

    elif args.command == "status":
        show_progress(args.config)

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
