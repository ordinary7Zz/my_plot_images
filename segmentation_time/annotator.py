#!/usr/bin/env python3
"""
2D 医学图像分割标注工具 — 人工 vs AI辅助时间效率对比实验
================================================================
功能:
  - 多边形轮廓模式 (默认): 点击描点 → Enter闭合 → 填充区域
    * 左键点击: 添加顶点 / 拖拽已有顶点
    * 右键点击: 删除最近顶点
    * Enter: 闭合多边形
    * Shift+点击: 在边上插入顶点
  - 画笔模式 (Tab切换): 涂抹添加/擦除区域
  - 两种子模式: 纯人工(空白) / AI辅助(加载预分割mask轮廓)
  - 撤销 Ctrl+Z / 重做 Ctrl+Y
  - 自动计时 + 操作次数统计
  - 日志自动保存为 CSV

依赖: pip install opencv-python numpy
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
from typing import Optional, List, Tuple
import math


# ============================================================
# 配置常量
# ============================================================
MAX_UNDO_STEPS = 200
DEFAULT_BRUSH_SIZE = 3
BRUSH_MIN, BRUSH_MAX = 1, 50
MASK_ALPHA = 0.45
MASK_COLOR = (0, 0, 255)          # mask 区域红色
CONTOUR_COLOR = (0, 255, 0)       # 多边形轮廓绿色
VERTEX_COLOR = (0, 255, 255)      # 顶点黄色
SELECTED_COLOR = (0, 165, 255)    # 选中顶点橙色
VERTEX_RADIUS = 5
SELECTED_RADIUS = 7
SNAP_RADIUS = 8                   # 点击捕捉顶点距离

# 状态栏配置
BAR_HEIGHT = 130
BAR_LINE_SPACING = 25
BAR_FONT = cv2.FONT_HERSHEY_SIMPLEX
BAR_FONT_SCALE = 0.45
BAR_FONT_THICKNESS = 1
BAR_BG = (40, 40, 40)
BAR_FG = (210, 210, 210)

# 轮廓简化参数
CONTOUR_EPSILON = 0.003           # approxPolyDP 精度 (相对周长)
MAX_CONTOUR_VERTICES = 30         # 从mask提取的最多顶点数


# ============================================================
# 主类
# ============================================================

class MaskAnnotator:
    """
    2D 分割标注器

    多边形轮廓模式 (默认):
      左键点击空白处 → 添加顶点
      左键拖拽顶点   → 移动顶点
      右键点击顶点   → 删除顶点
      Shift+点击边   → 插入顶点
      Enter          → 闭合多边形并填充

    画笔模式 (Tab 切换):
      左键拖拽 → 涂抹添加
      右键拖拽 → 擦除

    全局:
      Ctrl+Z / Ctrl+Y → 撤销/重做
      空格 → 暂停/恢复计时
      S / Enter → 保存并完成
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
        self.image_name = os.path.basename(image_path)

        # ---- 模式 ----
        self.annotation_mode = "manual"     # "manual" | "assisted"
        self.input_mode = "polygon"          # "polygon" | "brush"
        self.mask_path_input = mask_path

        # ---- 加载/初始化 mask ----
        if mask_path and os.path.exists(mask_path):
            self.annotation_mode = "assisted"
            raw = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if raw is None:
                self._init_blank_mask()
            else:
                if raw.shape[:2] != (self.orig_h, self.orig_w):
                    raw = cv2.resize(raw, (self.orig_w, self.orig_h))
                self.mask = ((raw > 128).astype(np.uint8)) * 255
        else:
            self._init_blank_mask()

        # ---- 多边形态 ----
        self.vertices: List[Tuple[int, int]] = []   # 当前多边形顶点列表
        self.selected_idx: int = -1                  # 被拖拽的顶点索引
        self.hover_idx: int = -1                     # 鼠标下方的顶点
        self.polygon_closed: bool = False             # 多边形是否已闭合
        self.vertex_drags: int = 0
        self.vertex_adds: int = 0
        self.vertex_deletes: int = 0
        self.polygon_closes: int = 0

        # 从预分割 mask 提取初始轮廓 (assisted 模式)
        if self.annotation_mode == "assisted":
            self._extract_contour_from_mask()

        # ---- 画笔状态 ----
        self.drawing = False
        self.erasing = False
        self.brush_size = brush_size
        self.mask_alpha = mask_alpha
        self.brush_strokes: int = 0
        self.eraser_strokes: int = 0
        self.brush_pixels: int = 0
        self.eraser_pixels: int = 0

        # ---- 撤销/重做 ----
        # 每个状态存储 (mask_copy, vertices_copy, polygon_closed)
        self.undo_stack: deque = deque(maxlen=MAX_UNDO_STEPS)
        self.redo_stack: deque = deque(maxlen=MAX_UNDO_STEPS)
        self._push_undo()
        self.undo_count: int = 0
        self.redo_count: int = 0

        # ---- 计时 ----
        self.t_start: float = 0.0
        self.t_paused_total: float = 0.0
        self.t_pause_start: float = 0.0
        self.is_paused: bool = False
        self.timer_running: bool = False

        self.saved = False

        # ---- 窗口 ----
        self.window_name = window_name
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
        if self.orig_h > 800:
            cv2.resizeWindow(window_name, int(self.orig_w * 0.7), int(self.orig_h * 0.7))
        else:
            cv2.resizeWindow(window_name, self.orig_w, self.orig_h)
        cv2.setMouseCallback(window_name, self._on_mouse)

    # ===================== 初始化辅助 =====================

    def _init_blank_mask(self):
        self.annotation_mode = "manual"
        self.mask = np.zeros((self.orig_h, self.orig_w), dtype=np.uint8)

    def _extract_contour_from_mask(self):
        """从预分割 mask 提取多边形轮廓"""
        contours, _ = cv2.findContours(self.mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            self.vertices = []
            self.polygon_closed = False
            return
        # 取最大轮廓
        largest = max(contours, key=cv2.contourArea)
        if len(largest) < 3:
            self.vertices = []
            return
        # 简化轮廓
        peri = cv2.arcLength(largest, True)
        epsilon = CONTOUR_EPSILON * peri
        approx = cv2.approxPolyDP(largest, epsilon, True)
        # 限制顶点数
        if len(approx) > MAX_CONTOUR_VERTICES:
            epsilon = peri / MAX_CONTOUR_VERTICES
            approx = cv2.approxPolyDP(largest, epsilon, True)
        self.vertices = [(int(p[0][0]), int(p[0][1])) for p in approx]
        self.polygon_closed = True
        self._fill_polygon()

    def _fill_polygon(self):
        """将当前多边形顶点填充到 mask"""
        if len(self.vertices) < 3:
            return
        pts = np.array(self.vertices, dtype=np.int32).reshape((-1, 1, 2))
        self.mask = np.zeros((self.orig_h, self.orig_w), dtype=np.uint8)
        cv2.fillPoly(self.mask, [pts], 255)

    # ===================== 撤销/重做 =====================

    def _get_state(self) -> tuple:
        """获取当前状态的快照"""
        return (self.mask.copy(), list(self.vertices), self.polygon_closed)

    def _restore_state(self, state: tuple):
        """恢复状态"""
        mask, verts, closed = state
        self.mask = mask
        self.vertices = verts
        self.polygon_closed = closed

    def _push_undo(self):
        state = self._get_state()
        if self.undo_stack:
            prev = self.undo_stack[-1]
            # 比较 mask 和 vertices
            if np.array_equal(prev[0], state[0]) and prev[1] == state[1]:
                return
        self.undo_stack.append(state)
        self.redo_stack.clear()

    def undo(self) -> bool:
        if len(self.undo_stack) <= 1:
            return False
        self.redo_stack.append(self.undo_stack.pop())
        self._restore_state(self.undo_stack[-1])
        self.undo_count += 1
        return True

    def redo(self) -> bool:
        if len(self.redo_stack) == 0:
            return False
        state = self.redo_stack.pop()
        self.undo_stack.append(state)
        self._restore_state(state)
        self.redo_count += 1
        return True

    # ===================== 鼠标事件路由 =====================

    def _on_mouse(self, event, x, y, flags, param):
        if self.input_mode == "polygon":
            self._on_mouse_polygon(event, x, y, flags)
        else:
            self._on_mouse_brush(event, x, y, flags)

    # ===================== 多边形模式鼠标 =====================

    def _find_nearest_vertex(self, x: int, y: int) -> int:
        """找到距离 (x,y) 最近的顶点索引, 在 SNAP_RADIUS 内返回, 否则 -1"""
        if not self.vertices:
            return -1
        best = -1
        best_dist = SNAP_RADIUS
        for i, (vx, vy) in enumerate(self.vertices):
            d = math.hypot(x - vx, y - vy)
            if d < best_dist:
                best_dist = d
                best = i
        return best

    def _find_nearest_edge(self, x: int, y: int) -> int:
        """找到距离 (x,y) 最近的边的起点索引, 否则 -1"""
        n = len(self.vertices)
        if n < 2:
            return -1
        # 闭合多边形也包括最后一条边
        indices = list(range(n)) if self.polygon_closed else list(range(n - 1))
        best = -1
        best_dist = float("inf")
        for i in indices:
            j = (i + 1) % n
            x1, y1 = self.vertices[i]
            x2, y2 = self.vertices[j]
            d = self._point_to_segment_dist(x, y, x1, y1, x2, y2)
            if d < best_dist:
                best_dist = d
                best = i
        if best_dist < SNAP_RADIUS * 2:
            return best
        return -1

    @staticmethod
    def _point_to_segment_dist(px, py, x1, y1, x2, y2) -> float:
        """点到线段的最短距离"""
        dx, dy = x2 - x1, y2 - y1
        if dx == 0 and dy == 0:
            return math.hypot(px - x1, py - y1)
        t = max(0, min(1, ((px - x1) * dx + (py - y1) * dy) / (dx * dx + dy * dy)))
        return math.hypot(px - (x1 + t * dx), py - (y1 + t * dy))

    def _on_mouse_polygon(self, event, x, y, flags):
        shift_held = bool(flags & cv2.EVENT_FLAG_SHIFTKEY)

        # ---- 鼠标移动: 更新 hover ----
        if event == cv2.EVENT_MOUSEMOVE:
            self.hover_idx = self._find_nearest_vertex(x, y)
            # 拖拽顶点
            if self.selected_idx >= 0 and self.selected_idx < len(self.vertices):
                self._push_undo()
                self.vertices[self.selected_idx] = (x, y)
                self.vertex_drags += 1
                if self.polygon_closed:
                    self._fill_polygon()

        # ---- 左键按下 ----
        elif event == cv2.EVENT_LBUTTONDOWN:
            near_v = self._find_nearest_vertex(x, y)

            if shift_held:
                # Shift+点击: 在边上插入顶点
                near_e = self._find_nearest_edge(x, y)
                if near_e >= 0:
                    self._push_undo()
                    n = len(self.vertices)
                    insert_at = (near_e + 1) % n if self.polygon_closed else near_e + 1
                    self.vertices.insert(insert_at, (x, y))
                    self.vertex_adds += 1
                    if self.polygon_closed:
                        self._fill_polygon()
                    self.selected_idx = insert_at
            elif near_v >= 0:
                # 拖拽已有顶点
                self.selected_idx = near_v
            else:
                # 添加新顶点
                self._push_undo()
                self.vertices.append((x, y))
                self.vertex_adds += 1
                self.polygon_closed = False
                self.selected_idx = len(self.vertices) - 1

        # ---- 左键松开 ----
        elif event == cv2.EVENT_LBUTTONUP:
            self.selected_idx = -1

        # ---- 右键: 删除顶点 ----
        elif event == cv2.EVENT_RBUTTONDOWN:
            near_v = self._find_nearest_vertex(x, y)
            if near_v >= 0 and len(self.vertices) > 3:
                self._push_undo()
                del self.vertices[near_v]
                self.vertex_deletes += 1
                if self.polygon_closed and len(self.vertices) >= 3:
                    self._fill_polygon()
                elif len(self.vertices) < 3:
                    self.polygon_closed = False

    # ===================== 画笔模式鼠标 (保留) =====================

    def _on_mouse_brush(self, event, x, y, flags):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self._push_undo()
            self.brush_strokes += 1
            self._apply_brush(x, y, add=True)
        elif event == cv2.EVENT_LBUTTONUP:
            self.drawing = False
        elif event == cv2.EVENT_RBUTTONDOWN:
            self.erasing = True
            self._push_undo()
            self.eraser_strokes += 1
            self._apply_brush(x, y, add=False)
        elif event == cv2.EVENT_RBUTTONUP:
            self.erasing = False
        elif event == cv2.EVENT_MOUSEMOVE:
            if self.drawing:
                self._apply_brush(x, y, add=True)
            elif self.erasing:
                self._apply_brush(x, y, add=False)

    def _apply_brush(self, x: int, y: int, add: bool):
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
        if not self.timer_running:
            return 0.0
        if self.is_paused:
            return time.time() - self.t_start - self.t_paused_total - (time.time() - self.t_pause_start)
        return time.time() - self.t_start - self.t_paused_total

    # ===================== 显示 =====================

    def _render(self) -> np.ndarray:
        """渲染图像 + mask + 多边形叠加"""
        overlay = self.img.copy()

        # 绘制 mask (半透明红色)
        overlay[self.mask > 0] = MASK_COLOR
        display = cv2.addWeighted(self.img, 1 - self.mask_alpha, overlay, self.mask_alpha, 0)

        # 多边形模式下绘制轮廓和顶点
        if self.input_mode == "polygon" and self.vertices:
            pts = np.array(self.vertices, dtype=np.int32).reshape((-1, 1, 2))

            # 绘制边
            if self.polygon_closed:
                cv2.polylines(display, [pts], True, CONTOUR_COLOR, 2)
            elif len(self.vertices) >= 2:
                cv2.polylines(display, [pts], False, CONTOUR_COLOR, 2)
                # 绘制最后一点到鼠标的虚线（在run中处理）

            # 绘制顶点
            for i, (vx, vy) in enumerate(self.vertices):
                color = SELECTED_COLOR if i == self.selected_idx else VERTEX_COLOR
                radius = SELECTED_RADIUS if i == self.selected_idx else VERTEX_RADIUS
                cv2.circle(display, (vx, vy), radius, color, -1)
                cv2.circle(display, (vx, vy), radius, (0, 0, 0), 1)

            # 高亮 hover 顶点
            if self.hover_idx >= 0 and self.hover_idx != self.selected_idx:
                vx, vy = self.vertices[self.hover_idx]
                cv2.circle(display, (vx, vy), VERTEX_RADIUS + 2, (255, 255, 255), 2)

        return display

    def _status_text(self) -> List[str]:
        """状态栏信息"""
        elapsed = self._elapsed()
        mins, secs = divmod(int(elapsed), 60)
        mode_str = "AI-Assisted" if self.annotation_mode == "assisted" else "Pure Manual"
        input_str = "Polygon" if self.input_mode == "polygon" else "Brush"
        tags = []
        if self.is_paused:
            tags.append("PAUSED")
        if self.input_mode == "polygon" and self.polygon_closed:
            tags.append("CLOSED")
        tag_str = " [" + " | ".join(tags) + "]" if tags else ""

        lines = [
            f"Image: {self.image_name}",
            f"Mode: {mode_str} | Input: {input_str}{tag_str}",
            f"Time: {mins:02d}:{secs:02d}",
        ]

        if self.input_mode == "polygon":
            lines.append(
                f"Verts: {len(self.vertices)} | +{self.vertex_adds} / -{self.vertex_deletes}"
                f" / ~{self.vertex_drags} | Undo: {self.undo_count}"
            )
        else:
            lines.append(
                f"Brush: {self.brush_size} | +{self.brush_strokes} / -{self.eraser_strokes}"
                f" | Undo: {self.undo_count}"
            )

        lines.append(f"Mask: {int((self.mask > 0).sum())} px")
        return lines

    def _draw_status_bar(self, display: np.ndarray) -> np.ndarray:
        bw = display.shape[1]
        bar = np.zeros((BAR_HEIGHT, bw, 3), dtype=np.uint8)
        bar[:] = BAR_BG

        lines = self._status_text()
        for i, line in enumerate(lines):
            y = int(18 + i * BAR_LINE_SPACING)
            cv2.putText(bar, line, (12, y),
                        BAR_FONT, BAR_FONT_SCALE, BAR_FG, BAR_FONT_THICKNESS,
                        cv2.LINE_AA)

        # 图例 (右下角)
        lx = bw - 290
        ly = BAR_HEIGHT - 30
        cv2.line(bar, (lx, ly), (lx + 32, ly), CONTOUR_COLOR, 2)
        cv2.putText(bar, "contour", (lx + 38, ly + 5),
                    BAR_FONT, 0.35, BAR_FG, 1, cv2.LINE_AA)
        cv2.circle(bar, (lx + 118, ly), 4, VERTEX_COLOR, -1)
        cv2.putText(bar, "vertex", (lx + 128, ly + 5),
                    BAR_FONT, 0.35, BAR_FG, 1, cv2.LINE_AA)
        cv2.rectangle(bar, (lx + 188, ly - 6), (lx + 206, ly + 6), MASK_COLOR, -1)
        cv2.putText(bar, "mask", (lx + 212, ly + 5),
                    BAR_FONT, 0.35, BAR_FG, 1, cv2.LINE_AA)

        return np.vstack([display, bar])

    # ===================== 主循环 =====================

    def run(self, output_mask_path: Optional[str] = None) -> dict:
        """启动标注主循环"""
        self._print_help()
        self._start_timer()

        while True:
            display = self._render()

            # 多边形模式: 绘制虚线到鼠标位置
            if self.input_mode == "polygon" and self.vertices and not self.polygon_closed:
                # 获取鼠标位置（通过 cv2.getWindowImageRect 做不到，用最后一帧的已知位置）
                pass  # draw_line_to_cursor 需要 setMouseCallback 配合，这里跳过虚线

            display = self._draw_status_bar(display)
            cv2.imshow(self.window_name, display)

            raw_key = cv2.waitKeyEx(10)
            key = raw_key & 0xFF

            # ==== 通用键 ====

            # 数字键: 笔刷大小
            if ord('1') <= key <= ord('9'):
                self.brush_size = min(BRUSH_MAX, key - ord('0'))
            elif key == ord('0'):
                self.brush_size = 10

            # +/-: 笔刷微调
            elif key == ord('+') or key == ord('='):
                self.brush_size = min(BRUSH_MAX, self.brush_size + 1)
            elif key == ord('-') or key == ord('_'):
                self.brush_size = max(BRUSH_MIN, self.brush_size - 1)

            # [ ]: 透明度
            elif key == ord('['):
                self.mask_alpha = max(0.1, self.mask_alpha - 0.1)
            elif key == ord(']'):
                self.mask_alpha = min(1.0, self.mask_alpha + 0.1)

            # Tab: 切换输入模式
            elif key == 9:
                self.input_mode = "brush" if self.input_mode == "polygon" else "polygon"
                if self.input_mode == "brush" and not self.polygon_closed and self.vertices:
                    # 切换到画笔时, 先闭合当前多边形
                    self._push_undo()
                    self._fill_polygon()
                    self.polygon_closed = True
                    self.polygon_closes += 1
                print(f"[INFO] Switched to {self.input_mode} mode")

            # Ctrl+Z: 撤销
            elif key in (26,):
                self.undo()

            # Ctrl+Y: 重做
            elif key in (25,):
                self.redo()

            # 空格: 暂停
            elif key == ord(' '):
                if self.is_paused:
                    self._resume_timer()
                else:
                    self._pause_timer()

            # Enter: 闭合多边形 (仅多边形模式)
            elif key == 13:
                if self.input_mode == "polygon" and not self.polygon_closed and len(self.vertices) >= 3:
                    self._push_undo()
                    self._fill_polygon()
                    self.polygon_closed = True
                    self.polygon_closes += 1
                else:
                    # 画笔模式或已闭合 → 保存退出
                    self.saved = True
                    break

            # s: 保存并退出
            elif key == ord('s'):
                # 如果多边形未闭合, 尝试闭合
                if self.input_mode == "polygon" and not self.polygon_closed and len(self.vertices) >= 3:
                    self._push_undo()
                    self._fill_polygon()
                    self.polygon_closed = True
                    self.polygon_closes += 1
                self.saved = True
                break

            # q / Esc: 退出不保存
            elif key == ord('q') or key == 27:
                self.saved = False
                break

            # r: 重置
            elif key == ord('r'):
                self._push_undo()
                if self.annotation_mode == "assisted" and self.mask_path_input:
                    raw = cv2.imread(self.mask_path_input, cv2.IMREAD_GRAYSCALE)
                    if raw is not None:
                        if raw.shape[:2] != (self.orig_h, self.orig_w):
                            raw = cv2.resize(raw, (self.orig_w, self.orig_h))
                        self.mask = ((raw > 128).astype(np.uint8)) * 255
                        self._extract_contour_from_mask()
                else:
                    self.mask = np.zeros((self.orig_h, self.orig_w), dtype=np.uint8)
                    self.vertices = []
                    self.polygon_closed = False
                self.vertex_adds = 0
                self.vertex_deletes = 0
                self.vertex_drags = 0
                self.polygon_closes = 0

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
            "mode": self.annotation_mode,
            "input_mode": self.input_mode,
            "time_seconds": round(elapsed, 1),
            "brush_strokes": self.brush_strokes,
            "eraser_strokes": self.eraser_strokes,
            "total_strokes": self.brush_strokes + self.eraser_strokes,
            "undo_count": self.undo_count,
            "redo_count": self.redo_count,
            "brush_pixels": self.brush_pixels,
            "eraser_pixels": self.eraser_pixels,
            "vertex_adds": self.vertex_adds,
            "vertex_deletes": self.vertex_deletes,
            "vertex_drags": self.vertex_drags,
            "polygon_closes": self.polygon_closes,
            "final_vertex_count": len(self.vertices),
            "mask_pixel_count": int((self.mask > 0).sum()),
            "saved": self.saved,
            "finished": True,
        }
        return result

    def _print_help(self):
        mode_label = "AI辅助 (预分割轮廓已加载)" if self.annotation_mode == "assisted" else "纯人工 (空白, 从零开始)"
        print("=" * 58)
        print(f"  Image: {self.image_name}")
        print(f"  Mode:  {mode_label}")
        print(f"  Size:  {self.orig_w} x {self.orig_h}")
        print(f"  Input: 多边形轮廓 (默认)")
        print("-" * 58)
        print("  === 多边形模式 (默认) ===")
        print("  左键点击        = 添加顶点")
        print("  左键拖拽顶点    = 移动顶点")
        print("  右键点击顶点    = 删除顶点")
        print("  Shift+点击边    = 插入顶点")
        print("  Enter           = 闭合多边形并填充")
        print("  === 画笔模式 (Tab 切换) ===")
        print("  左键拖拽        = 涂抹添加区域")
        print("  右键拖拽        = 擦除区域")
        print("  === 全局 ===")
        print("  Tab             = 切换多边形/画笔模式")
        print("  Ctrl+Z / Ctrl+Y = 撤销 / 重做")
        print("  1-9             = 笔刷大小")
        print("  +/-             = 微调笔刷")
        print("  [ ]             = 调整透明度")
        print("  R               = 重置")
        print("  空格            = 暂停/恢复计时")
        print("  S / Enter(闭合) = 保存并完成")
        print("  Q / Esc         = 退出 (不保存)")
        print("=" * 58)


# ============================================================
# CLI
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="2D 医学图像分割标注工具 — 多边形轮廓 + 画笔模式"
    )
    parser.add_argument("--image", required=True,
                        help="原始图像路径")
    parser.add_argument("--mask", default=None,
                        help="预分割 mask 路径 (不提供则为纯人工模式)")
    parser.add_argument("--output-mask", default=None,
                        help="修正后 mask 保存路径")
    parser.add_argument("--output-log", default=None,
                        help="日志 CSV 路径 (追加写入)")
    parser.add_argument("--brush-size", type=int, default=DEFAULT_BRUSH_SIZE,
                        help=f"笔刷大小 (默认 {DEFAULT_BRUSH_SIZE})")
    parser.add_argument("--annotator", default="unknown",
                        help="标注者 ID")

    args = parser.parse_args()

    editor = MaskAnnotator(
        image_path=args.image,
        mask_path=args.mask,
        brush_size=args.brush_size,
    )
    result = editor.run(output_mask_path=args.output_mask)

    result["annotator"] = args.annotator
    result["timestamp"] = datetime.now().isoformat()
    result["image_path"] = os.path.abspath(args.image)
    result["mask_path_input"] = os.path.abspath(args.mask) if args.mask else ""

    print("\n" + "=" * 58)
    print("  ANNOTATION RESULT")
    print("-" * 58)
    print(f"  Mode:         {result['mode']}")
    print(f"  Input:        {result['input_mode']}")
    print(f"  Time:         {result['time_seconds']:.1f}s")
    if result["vertex_adds"] > 0 or result["vertex_drags"] > 0:
        print(f"  Vertices:     +{result['vertex_adds']} / -{result['vertex_deletes']} / ~{result['vertex_drags']}")
        print(f"  Final verts:  {result['final_vertex_count']}")
    if result["brush_strokes"] > 0:
        print(f"  Brush:        +{result['brush_strokes']} / -{result['eraser_strokes']}")
    print(f"  Undo:         {result['undo_count']}")
    print(f"  Mask pixels:  {result['mask_pixel_count']}")
    print(f"  Saved:        {result['saved']}")
    print("=" * 58)

    log_path = args.output_log or "segmentation_timing_log.csv"
    _write_log_csv(log_path, result)


def _write_log_csv(logpath: str, result: dict):
    fieldnames = [
        "timestamp", "annotator", "image_name", "image_path",
        "mode", "input_mode", "mask_path_input",
        "time_seconds", "brush_strokes", "eraser_strokes", "total_strokes",
        "undo_count", "redo_count",
        "brush_pixels", "eraser_pixels",
        "vertex_adds", "vertex_deletes", "vertex_drags", "polygon_closes",
        "final_vertex_count", "mask_pixel_count",
        "saved", "finished",
    ]
    file_exists = os.path.exists(logpath)
    with open(logpath, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow({k: result.get(k, "") for k in fieldnames})
    print(f"[LOG] {logpath}")


if __name__ == "__main__":
    main()
