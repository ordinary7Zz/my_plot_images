#!/usr/bin/env python3
"""
批量推理脚本 — 用分割模型对测试集图像生成预分割 mask
============================================================
使用方式:
  1. 修改下方 MODEL_INFERENCE_FUNC 为你自己的模型推理逻辑
  2. 准备好图像目录
  3. 运行: python generate_masks.py --image-dir /path/to/images --output-dir /path/to/masks

模型推理接口约定:
  def predict(image_path: str) -> np.ndarray:
      # 返回二值 mask, shape=(H,W), dtype=uint8, 值 0/255
      return binary_mask

依赖: pip install opencv-python numpy
"""

import os
import sys
import argparse
import time
import json
from pathlib import Path
import cv2
import numpy as np


# ============================================================
# ★★★ 用户必须修改此函数为自己的模型推理逻辑 ★★★
# ============================================================

def model_inference(image_path: str) -> np.ndarray:
    """
    模型推理接口: 输入图像路径, 输出二值 mask (0/255, uint8)

    示例: 加载预训练的 PyTorch 模型进行推理

    TODO: 替换为你自己的模型
    """
    # ---- 示例实现 ----
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Cannot read: {image_path}")

    H, W = img.shape

    # ====== 在此处替换为你自己的推理代码 ======
    # 示例: 简单阈值分割 (仅用于测试工具链)
    _, binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # ====== 更真实的示例: PyTorch 模型推理 ======
    # import torch
    # from your_model import ThyroidXAgent
    #
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # model = ThyroidXAgent.load_from_checkpoint("checkpoint.ckpt")
    # model.to(device)
    # model.eval()
    #
    # # 预处理
    # from torchvision import transforms
    # transform = transforms.Compose([
    #     transforms.ToPILImage(),
    #     transforms.Resize((256, 256)),
    #     transforms.ToTensor(),
    # ])
    # input_tensor = transform(img).unsqueeze(0).to(device)
    #
    # with torch.no_grad():
    #     pred = model(input_tensor)
    #     pred_mask = torch.sigmoid(pred).squeeze().cpu().numpy()
    #
    # binary = (pred_mask > 0.5).astype(np.uint8) * 255
    # binary = cv2.resize(binary, (W, H), interpolation=cv2.INTER_NEAREST)

    return binary


# ============================================================
# 批量推理
# ============================================================

def find_images(image_dir: str, extensions=(".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")):
    """递归查找所有图像文件"""
    image_dir = Path(image_dir).expanduser().resolve()
    images = []
    for ext in extensions:
        images.extend(image_dir.rglob(f"*{ext}"))
        images.extend(image_dir.rglob(f"*{ext.upper()}"))
    return sorted(set(images))


def run_batch_inference(
    image_dir: str,
    output_dir: str,
    extensions: tuple = (".png", ".jpg", ".jpeg", ".bmp"),
    skip_existing: bool = True,
    trim_suffixes: tuple = (),
) -> list:
    """
    批量推理

    参数:
        image_dir: 图像目录
        output_dir: mask 输出目录
        skip_existing: 是否跳过已存在的 mask
        trim_suffixes: 文件名中要去掉的后缀 (如 '_image', '_img')

    返回:
        list[dict]: 每个文件的推理结果
    """
    images = find_images(image_dir, extensions)
    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Found {len(images)} images in {image_dir}")
    print(f"Output directory: {output_dir}")
    print("-" * 55)

    results = []
    total_time = 0.0

    for i, img_path in enumerate(images):
        # 生成 mask 文件名
        stem = img_path.stem
        for sfx in trim_suffixes:
            if stem.endswith(sfx):
                stem = stem[:-len(sfx)]

        mask_name = f"{stem}_mask.png"
        mask_path = output_dir / mask_name

        if skip_existing and mask_path.exists():
            print(f"[{i+1:4d}/{len(images)}] SKIP (exists): {mask_name}")
            results.append({
                "image": str(img_path),
                "mask": str(mask_path),
                "status": "skipped",
                "time_seconds": 0.0,
            })
            continue

        # 推理
        t0 = time.time()
        try:
            mask = model_inference(str(img_path))
            t_elapsed = time.time() - t0
            cv2.imwrite(str(mask_path), mask)
            total_time += t_elapsed

            mask_pixels = int((mask > 128).sum())
            print(f"[{i+1:4d}/{len(images)}] OK  {mask_name}  "
                  f"({t_elapsed:.2f}s, mask_px={mask_pixels})")

            results.append({
                "image": str(img_path),
                "mask": str(mask_path),
                "status": "ok",
                "time_seconds": round(t_elapsed, 2),
                "mask_pixels": mask_pixels,
            })
        except Exception as e:
            t_elapsed = time.time() - t0
            print(f"[{i+1:4d}/{len(images)}] ERR {mask_name}: {e}")
            results.append({
                "image": str(img_path),
                "mask": str(mask_path),
                "status": f"error: {e}",
                "time_seconds": round(t_elapsed, 2),
            })

    # 汇总
    n_ok = sum(1 for r in results if r["status"] == "ok")
    n_skip = sum(1 for r in results if r["status"] == "skipped")
    n_err = sum(1 for r in results if r["status"].startswith("error"))

    print("=" * 55)
    print(f"Done: {n_ok} ok, {n_skip} skipped, {n_err} errors")
    print(f"Total inference time: {total_time:.1f}s")
    print(f"Masks saved to: {output_dir}")
    return results


# ============================================================
# CLI
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="批量用分割模型生成预分割 mask"
    )
    parser.add_argument("--image-dir", required=True,
                        help="原始图像目录")
    parser.add_argument("--output-dir", required=True,
                        help="mask 输出目录")
    parser.add_argument("--extensions", nargs="+",
                        default=[".png", ".jpg", ".jpeg", ".bmp"],
                        help="图像扩展名 (默认 .png .jpg .jpeg .bmp)")
    parser.add_argument("--trim-suffixes", nargs="*", default=[],
                        help="从文件名中去掉的后缀 (如 _image)")
    parser.add_argument("--no-skip", action="store_true",
                        help="不跳过已存在的 mask (会覆盖)")
    parser.add_argument("--save-results", default=None,
                        help="保存推理结果摘要为 JSON")

    args = parser.parse_args()

    if not os.path.isdir(args.image_dir):
        print(f"ERROR: image-dir 不存在: {args.image_dir}")
        sys.exit(1)

    results = run_batch_inference(
        image_dir=args.image_dir,
        output_dir=args.output_dir,
        extensions=tuple(args.extensions),
        skip_existing=not args.no_skip,
        trim_suffixes=tuple(args.trim_suffixes),
    )

    if args.save_results:
        with open(args.save_results, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to: {args.save_results}")


if __name__ == "__main__":
    main()
