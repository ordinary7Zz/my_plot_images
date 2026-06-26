#!/usr/bin/env python3
"""
2D 医学图像分割标注工具 — 用于人工 vs AI辅助时间效率对比实验
================================================================
功能:
  - 两种模式: 纯人工(空白mask) / AI辅助(导入预分割mask修改)
  - 画笔涂抹添加区域 | 橡皮擦删除区域
  - 撤销 (Ctrl+Z)  + 重做 (Ctrl+Y / Ctrl+Shift+Z)
  - 自动计时 + 操作次数统计 (涂抹次数、擦除次数、撤销次数)
  - 日志自动保存为 CSV

依赖: pip install opencv-python numpy

作者: segmentation_time 实验套件
"""

import cv2
import numpy as np
import time
import os
import sys
import csv
import argparse
from datetime import datetime
from collections import deque
from typing import Optional, Tuple, List


# ============================================================
# 配置常量
# ============================================================
MAX_UNDO_STEPS = 200           # 最大撤销步数
DEFAULT_BRUSH_SIZE = 3
BRUSH_MIN, BRUSH_MAX = 1, 50
MASK_ALPHA = 0.5               # mask 叠加透明度
MASK_COLOR = (0, 0, 255)       # 红色 (BGR)
ERASER_CURSOR_COLOR = (255, 255, 255)
BRUSH_CURSOR_COLOR = (255, 0, 0)


class MaskAnnotator:
    """
    2D 分割标注器

    支持:
      - 画笔模式 (左键): 在 mask 上添加区域
      - 橡皮擦模式 (右键): 从 mask 上擦除区域
      - 撤销 (Ctrl+Z) / 重做 (Ctrl+Y 或 Ctrl+Shift+Z)
      - 自动计时、操作计数
    """

    def __init__(
        self,
        image_path: str,
        mask_path: Optional[str] = None,
        window_name: str = "Segmentation Annotator",
        brush_size: int = DEFAULT_BRUSH_SIZE,
        mask_alpha: float = MASK_ALPHA,
    ):
        # ---- 加载图像 ----
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"图像不存在: {image_path}")
        self.img = cv2.imread(image_path)
        if self.img is None:
            raise ValueError(f"无法读取图像: {image_path}")
        self.orig_h, self.orig_w = self.img.shape[:2]

        # ---- 加载或初始化 mask ----
        self.mode = "manual"          # "manual" | "assisted"
        self.mask_path_input = mask_path

        if mask_path and os.path.exists(mask_path):
            self.mode = "assisted"
            raw = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if raw is None:
                print(f"[WARN] 无法读取 mask: {mask_path}，使用空白 mask")
                self.mask = np.zeros((self.orig_h, self.orig_w), dtype=np.uint8)
                self.mode = "manual"
            else:
                if raw.shape[:2] != (self.orig_h, self.orig_w):
                    raw = cv2.resize(raw, (self.orig_w, self.orig_h))
                self.mask = ((raw > 128).astype(np.uint8)) * 255
        else:
            self.mask = np.zeros((self.orig_h, self.orig_w), dtype=np.uint8)

        # ---- 撤销/重做栈 ----
        self.undo_stack: deque[np.ndarray] = deque(maxlen=MAX_UNDO_STEPS)
        self.redo_stack: deque[np.ndarray] = deque(maxlen=MAX_UNDO_STEPS)
        self._push_undo()  # 保存初始状态

        # ---- 画笔状态 ----
        self.drawing = False
        self.erasing = False
        self.brush_size = brush_size
        self.mask_alpha = mask_alpha

        # ---- 时间与统计 ----
        self.t_start: float = 0.0
        self.t_paused_total: float = 0.0
        self.t_pause_start: float = 0.0
        self.is_paused: bool = False
        self.timer_running: bool = False

        # 操作计数
        self.brush_strokes: int = 0      # 左键涂抹笔画数
        self.eraser_strokes: int = 0     # 右键擦除笔画数
        self.undo_count: int = 0         # 撤销次数
        self.redo_count: int = 0         # 重做次数
        self.brush_pixels: int = 0       # 涂抹总像素数
        self.eraser_pixels: int = 0      # 擦除总像素数
        self.saved = False

        # ---- 显示缩放 ----
        self.zoom = 1.0
        self.pan_x, self.pan_y = 0, 0
        self.dragging = False
        self.drag_start = (0, 0)

        # ---- 窗口 ----
        self.window_name = window_name
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
        if self.orig_h > 800:
            cv2.resizeWindow(window_name, int(self.orig_w * 0.7), int(self.orig_h * 0.7))
        else:
            cv2.resizeWindow(window_name, self.orig_w, self.orig_h)
        cv2.setMouseCallback(window_name, self._on_mouse)

        self.image_name = os.path.basename(image_path)

    # ===================== 撤销/重做 =====================

    def _push_undo(self):
        """将当前 mask 推入撤销栈 (去重: 与栈顶相同则跳过)"""
        if len(self.undo_stack) > 0 and np.array_equal(self.undo_stack[-1], self.mask):
            return
        self.undo_stack.append(self.mask.copy())
        self.redo_stack.clear()  # 新操作清空重做栈

    def undo(self) -> bool:
        """
        撤销: 恢复到上一个快照

        逻辑:
          _push_undo 在笔刷开始前保存当前状态
          撤销时, 弹出栈顶状态 (上笔之前的 mask), 将当前 mask 存入重做栈
        """
        if len(self.undo_stack) == 0:
            return False
        # 弹出上一个快照
        prev_state = self.undo_stack.pop()
        # 将当前 mask 存入重做栈 (以便 redo)
        self.redo_stack.append(self.mask.copy())
        # 恢复到上一个快照
        self.mask = prev_state
        self.undo_count += 1
        return True

    def redo(self) -> bool:
        """
        重做: 恢复到撤销前的状态
        """
        if len(self.redo_stack) == 0:
            return False
        # 弹出重做栈顶 (撤销前的 mask)
        next_state = self.redo_stack.pop()
        # 将当前 mask 存入撤销栈
        self.undo_stack.append(self.mask.copy())
        # 恢复到撤销前的状态
        self.mask = next_state
        self.redo_count += 1
        return True

    # ===================== 鼠标回调 =====================

    def _on_mouse(self, event, x, y, flags, param):
        # ---- 中键拖拽平移 ----
        if event == cv2.EVENT_MBUTTONDOWN:
            self.dragging = True
            self.drag_start = (x, y)
        elif event == cv2.EVENT_MBUTTONUP:
            self.dragging = False
        elif event == cv2.EVENT_MOUSEMOVE and self.dragging:
            dx = x - self.drag_start[0]
            dy = y - self.drag_start[1]
            self.pan_x += dx
            self.pan_y += dy
            self.drag_start = (x, y)

        # ---- 画笔 ----
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self._push_undo()
            self.brush_strokes += 1
            self._apply_brush(x, y, add=True)
        elif event == cv2.EVENT_LBUTTONUP:
            self.drawing = False

        # ---- 橡皮擦 ----
        if event == cv2.EVENT_RBUTTONDOWN:
            self.erasing = True
            self._push_undo()
            self.eraser_strokes += 1
            self._apply_brush(x, y, add=False)
        elif event == cv2.EVENT_RBUTTONUP:
            self.erasing = False

        # ---- 拖拽中持续绘制 ----
        if event == cv2.EVENT_MOUSEMOVE:
            if self.drawing:
                self._apply_brush(x, y, add=True)
            elif self.erasing:
                self._apply_brush(x, y, add=False)

    def _apply_brush(self, x: int, y: int, add: bool):
        """在指定位置绘制/擦除"""
        val = 255 if add else 0
        before = self.mask.sum()
        cv2.circle(self.mask, (x, y), self.brush_size, val, -1)
        after = self.mask.sum()
        delta = abs(int(after) - int(before))
        if add:
            self.brush_pixels += delta
        else:
            self.eraser_pixels += delta

    # ===================== 计时 =====================

    def _start_timer(self):
        if not self.timer_running:
            self.t_start = time.time()
            self.t_paused_total = 0.0
            self.timer_running = True
            self.is_paused = False

    def _pause_timer(self):
        if self.timer_running and not self.is_paused:
            self.t_pause_start = time.time()
            self.is_paused = True

    def _resume_timer(self):
        if self.timer_running and self.is_paused:
            self.t_paused_total += time.time() - self.t_pause_start
            self.is_paused = False

    def _elapsed(self) -> float:
        """获取有效耗时（排除暂停）"""
        if not self.timer_running:
            return 0.0
        if self.is_paused:
            return time.time() - self.t_start - self.t_paused_total - (time.time() - self.t_pause_start)
        return time.time() - self.t_start - self.t_paused_total

    # ===================== 显示 =====================

    def _render(self) -> np.ndarray:
        """渲染叠加视图"""
        overlay = self.img.copy()
        overlay[self.mask > 0] = MASK_COLOR
        display = cv2.addWeighted(self.img, 1 - self.mask_alpha, overlay, self.mask_alpha, 0)
        return display

    def _status_text(self) -> List[str]:
        """生成状态栏信息"""
        elapsed = self._elapsed()
        mins, secs = divmod(int(elapsed), 60)
        mode_str = "AI-Assisted" if self.mode == "assisted" else "Pure Manual"
        pause_str = " [PAUSED]" if self.is_paused else ""

        lines = [
            f"Mode: {mode_str}{pause_str}",
            f"Time: {mins:02d}:{secs:02d}",
            f"Brush: {self.brush_size} | Undo: {self.undo_count} | Redo: {self.redo_count}",
            f"Strokes: +{self.brush_strokes} / -{self.eraser_strokes}",
            f"Pixels: +{self.brush_pixels} / -{self.eraser_pixels}",
        ]
        return lines

    def _draw_status_bar(self, display: np.ndarray):
        """在画面底部绘制半透明状态栏"""
        bar_h = 110
        h, w = display.shape[:2]
        bar = np.zeros((bar_h, w, 3), dtype=np.uint8)
        bar[:] = (40, 40, 40)

        lines = self._status_text()
        for i, line in enumerate(lines):
            cv2.putText(bar, line, (12, 20 + i * 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)

        # 图例示意
        cv2.rectangle(bar, (w - 280, 10), (w - 270, 26), MASK_COLOR, -1)
        cv2.putText(bar, "= mask", (w - 260, 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

        return np.vstack([display, bar])

    # ===================== 主循环 =====================

    def run(self, output_mask_path: Optional[str] = None) -> dict:
        """
        启动标注主循环

        参数:
            output_mask_path: 修正后 mask 的保存路径（None 则自动生成）

        返回:
            dict: 包含 time_seconds, brush_strokes, eraser_strokes,
                  undo_count, redo_count, brush_pixels, eraser_pixels, saved
        """
        # 打印操作指南
        self._print_help()

        self._start_timer()

        while True:
            display = self._render()
            display = self._draw_status_bar(display)
            cv2.imshow(self.window_name, display)

            # 使用 waitKeyEx 获取原始 keycode (支持 Ctrl 组合键)
            raw = cv2.waitKeyEx(10)
            key = raw & 0xFF
            keycode = raw & 0xFFFF  # 完整 keycode

            # ---- 数字键 调整笔刷大小 ----
            if ord('1') <= key <= ord('9'):
                self.brush_size = min(BRUSH_MAX, key - ord('0'))
            elif key == ord('0'):
                self.brush_size = 10

            # ---- +/- 微调笔刷 ----
            elif key == ord('+') or key == ord('='):
                self.brush_size = min(BRUSH_MAX, self.brush_size + 1)
            elif key == ord('-') or key == ord('_'):
                self.brush_size = max(BRUSH_MIN, self.brush_size - 1)

            # ---- [ ] 调整透明度 ----
            elif key == ord('['):
                self.mask_alpha = max(0.1, self.mask_alpha - 0.1)
            elif key == ord(']'):
                self.mask_alpha = min(1.0, self.mask_alpha + 0.1)

            # ---- 撤销 Ctrl+Z (keycode 26 = 0x1A) ----
            elif keycode == 26 or key == 26:
                self.undo()

            # ---- 重做 Ctrl+Y / Ctrl+Shift+Z ----
            elif keycode == 25 or key == 25:
                self.redo()

            # ---- 空格 暂停/恢复 ----
            elif key == ord(' '):
                if self.is_paused:
                    self._resume_timer()
                else:
                    self._pause_timer()

            # ---- Enter / s 保存并退出 ----
            elif key == 13 or key == ord('s'):  # Enter or s
                self.saved = True
                break

            # ---- q / Esc 退出 (不保存) ----
            elif key == ord('q') or key == 27:
                self.saved = False
                break

            # ---- r 重置 mask ----
            elif key == ord('r'):
                self._push_undo()
                if self.mode == "assisted" and self.mask_path_input:
                    raw = cv2.imread(self.mask_path_input, cv2.IMREAD_GRAYSCALE)
                    if raw is not None:
                        if raw.shape[:2] != (self.orig_h, self.orig_w):
                            raw = cv2.resize(raw, (self.orig_w, self.orig_h))
                        self.mask = ((raw > 128).astype(np.uint8)) * 255
                else:
                    self.mask = np.zeros((self.orig_h, self.orig_w), dtype=np.uint8)

            # ---- w/a/x/d 键盘平移 (备用手势) ----
            elif key == ord('w'):   # 上
                self.pan_y += 50
            elif key == ord('a'):   # 左
                self.pan_x += 50
            elif key == ord('d'):   # 右
                self.pan_x -= 50
            elif key == ord('x'):   # 下
                self.pan_y -= 50

        cv2.destroyAllWindows()

        elapsed = self._elapsed()

        # 保存 mask
        if self.saved:
            if output_mask_path is None:
                base, _ = os.path.splitext(self.image_name)
                output_mask_path = f"{base}_corrected_mask.png"
            cv2.imwrite(output_mask_path, self.mask)
            print(f"[SAVED] {output_mask_path}")

        # 构建结果
        result = {
            "image_name": self.image_name,
            "mode": self.mode,
            "time_seconds": round(elapsed, 1),
            "brush_strokes": self.brush_strokes,
            "eraser_strokes": self.eraser_strokes,
            "total_strokes": self.brush_strokes + self.eraser_strokes,
            "undo_count": self.undo_count,
            "redo_count": self.redo_count,
            "brush_pixels": self.brush_pixels,
            "eraser_pixels": self.eraser_pixels,
            "mask_pixel_count": int((self.mask > 0).sum()),
            "saved": self.saved,
            "finished": True,
        }
        return result

    def _print_help(self):
        """打印操作帮助"""
        mode_label = "AI辅助 (预分割 mask 已加载)" if self.mode == "assisted" else "纯人工 (空白, 从零开始)"
        print("=" * 55)
        print(f"  Image: {self.image_name}")
        print(f"  Mode:  {mode_label}")
        print(f"  Size:  {self.orig_w} x {self.orig_h}")
        print("-" * 55)
        print("  左键拖拽  = 涂抹添加区域")
        print("  右键拖拽  = 擦除区域")
        print("  中键拖拽  = 平移画面")
        print("  1-9       = 快速设置笔刷大小")
        print("  +/-       = 调整笔刷大小")
        print("  [ ]       = 调整 mask 透明度")
        print("  Ctrl+Z    = 撤销")
        print("  Ctrl+Y    = 重做")
        print("  R         = 重置 mask 到初始状态")
        print("  空格      = 暂停/恢复计时")
        print("  S / Enter = 保存并完成")
        print("  Q / Esc   = 退出 (不保存)")
        print("=" * 55)


# ============================================================
# 命令行入口
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="2D 医学图像分割标注工具 — 人工 vs AI辅助时间效率对比"
    )
    parser.add_argument("--image", required=True,
                        help="原始图像路径")
    parser.add_argument("--mask", default=None,
                        help="预分割 mask 路径 (不提供则为纯人工模式)")
    parser.add_argument("--output-mask", default=None,
                        help="修正后 mask 保存路径 (默认: {image_name}_corrected_mask.png)")
    parser.add_argument("--output-log", default=None,
                        help="日志文件路径 (CSV, 追加写入; 不提供则仅打印)")
    parser.add_argument("--brush-size", type=int, default=DEFAULT_BRUSH_SIZE,
                        help=f"初始笔刷大小 (默认 {DEFAULT_BRUSH_SIZE})")
    parser.add_argument("--annotator", default="unknown",
                        help="标注者 ID (用于日志记录)")

    args = parser.parse_args()

    # 启动标注器
    editor = MaskAnnotator(
        image_path=args.image,
        mask_path=args.mask,
        brush_size=args.brush_size,
    )
    result = editor.run(output_mask_path=args.output_mask)

    # 追加元信息
    result["annotator"] = args.annotator
    result["timestamp"] = datetime.now().isoformat()
    result["image_path"] = os.path.abspath(args.image)
    result["mask_path_input"] = os.path.abspath(args.mask) if args.mask else ""

    # 输出结果
    print("\n" + "=" * 55)
    print("  ANNOTATION RESULT")
    print("-" * 55)
    print(f"  Mode:         {result['mode']}")
    print(f"  Time:         {result['time_seconds']:.1f}s")
    print(f"  Brush strokes:{result['brush_strokes']}")
    print(f"  Eraser strokes:{result['eraser_strokes']}")
    print(f"  Undo:         {result['undo_count']}")
    print(f"  Redo:         {result['redo_count']}")
    print(f"  Mask pixels:  {result['mask_pixel_count']}")
    print(f"  Saved:        {result['saved']}")
    print("=" * 55)

    # 写入日志
    if args.output_log:
        _write_log_csv(args.output_log, result)
    else:
        # 默认写入当前目录
        default_log = "segmentation_timing_log.csv"
        _write_log_csv(default_log, result)


def _write_log_csv(logpath: str, result: dict):
    """追加写入 CSV 日志"""
    fieldnames = [
        "timestamp", "annotator", "image_name", "image_path",
        "mode", "mask_path_input",
        "time_seconds", "brush_strokes", "eraser_strokes", "total_strokes",
        "undo_count", "redo_count",
        "brush_pixels", "eraser_pixels", "mask_pixel_count",
        "saved", "finished",
    ]
    file_exists = os.path.exists(logpath)
    with open(logpath, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow({k: result.get(k, "") for k in fieldnames})
    print(f"[LOG] 写入 {logpath}")


if __name__ == "__main__":
    main()
