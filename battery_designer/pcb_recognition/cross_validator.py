"""正反面交叉校验模块

功能：
  1. 背面轮廓水平镜像（像翻书一样，左↔右互换）
  2. 质心对齐 + IoU 微调
  3. 交集 mask 去除单面毛刺
  4. 生成 diff 可视化图（绿=共识 / 红=仅正面 / 蓝=仅背面）

设计原则：
  - 简化：去掉直边匹配、网格搜索、缩放补偿等复杂逻辑
  - 可靠：质心对齐对小偏差鲁棒，IoU 微调仅做小范围精修
  - 可视：diff 图帮助用户理解校验结果
"""

from __future__ import annotations

import base64
import logging
from typing import Any

import cv2
import numpy as np

from ..logger import get_logger

_log = get_logger(__name__)

# 共享画布缩放因子（与原 _merge_front_back_outlines 一致）
SCALE = 10.0


class CrossValidator:
    """正反面轮廓交叉校验器

    用正反面两张 PCB 轮廓的交集消除单面检测的毛刺/阴影。
    """

    @staticmethod
    def validate(
        front_result: dict[str, Any],
        back_result: dict[str, Any],
    ) -> dict[str, Any]:
        """执行正反面交叉校验

        Args:
            front_result: 正面 pipeline.run() 结果，含 outline/pixels_per_mm/width_mm/height_mm
            back_result: 背面 pipeline.run() 结果

        Returns:
            {
                "outline": list[dict],       # 共识轮廓 (mm)
                "front_area_mm2": float,     # 正面面积 (mm²)
                "back_area_mm2": float,      # 背面面积 (mm²)
                "consensus_area_mm2": float, # 共识面积 (mm²)
                "diff_image_b64": str,       # diff 可视化图 (base64 PNG)
                "transparent_pcb_b64": str,  # 从共识轮廓生成的透明 PNG
            }
        """
        _log.info("Step 7: 正反面交叉校验 - 开始")

        # 提取参数
        front_outline = front_result.get("outline", [])
        back_outline = back_result.get("outline", [])
        front_ppm = front_result.get("pixels_per_mm", 1.0)
        back_ppm = back_result.get("pixels_per_mm", 1.0)
        w_mm = max(front_result.get("width_mm", 40.0), back_result.get("width_mm", 40.0))
        h_mm = max(front_result.get("height_mm", 25.0), back_result.get("height_mm", 25.0))

        if len(front_outline) < 3 or len(back_outline) < 3:
            _log.warning("轮廓点数不足，跳过交叉校验")
            return {
                "outline": front_outline,
                "front_area_mm2": 0.0,
                "back_area_mm2": 0.0,
                "consensus_area_mm2": 0.0,
                "diff_image_b64": "",
                "transparent_pcb_b64": front_result.get("transparent_pcb_b64", ""),
            }

        # ── Step 7.1: 转换到共享画布坐标 ──
        canvas_w = int(w_mm * SCALE)
        canvas_h = int(h_mm * SCALE)

        front_pts = CrossValidator._outline_to_canvas(front_outline, canvas_w, canvas_h)
        back_pts = CrossValidator._outline_to_canvas(back_outline, canvas_w, canvas_h)

        # ── Step 7.2: 背面水平镜像 ──
        mirrored_back = CrossValidator._mirror_horizontal(back_pts, canvas_w)

        # ── Step 7.3: 质心对齐 ──
        front_centroid = CrossValidator._centroid(front_pts)
        back_centroid = CrossValidator._centroid(mirrored_back)
        offset = (front_centroid[0] - back_centroid[0], front_centroid[1] - back_centroid[1])
        aligned_back = CrossValidator._translate(mirrored_back, offset)

        # ── Step 7.4: IoU 微调（小范围滑动窗口） ──
        refined_back, best_iou = CrossValidator._iou_refine(front_pts, aligned_back, canvas_w, canvas_h)
        _log.info("IoU 微调: 最佳 IoU=%.3f", best_iou)

        # ── Step 7.5: 生成 mask 并计算交集 ──
        front_mask = np.zeros((canvas_h, canvas_w), dtype=np.uint8)
        back_mask = np.zeros((canvas_h, canvas_w), dtype=np.uint8)
        cv2.fillPoly(front_mask, [front_pts.astype(np.int32)], 255)
        cv2.fillPoly(back_mask, [refined_back.astype(np.int32)], 255)

        consensus_mask = cv2.bitwise_and(front_mask, back_mask)

        # 形态学 OPEN 修剪细小突起
        kernel = np.ones((3, 3), np.uint8)
        consensus_mask = cv2.morphologyEx(consensus_mask, cv2.MORPH_OPEN, kernel)

        # ── Step 7.6: 从交集 mask 提取轮廓 ──
        contours, _ = cv2.findContours(consensus_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            # 取最大轮廓
            consensus_contour = max(contours, key=cv2.contourArea)
            consensus_pts = consensus_contour.reshape(-1, 2)
        else:
            consensus_pts = front_pts.astype(np.int32)

        # ── Step 7.7: 转换回 mm 坐标 ──
        consensus_outline = CrossValidator._canvas_to_outline(consensus_pts, canvas_w, canvas_h, w_mm, h_mm)

        # ── Step 7.8: 计算面积 ──
        front_area = cv2.contourArea(front_pts) / (SCALE ** 2)
        back_area = cv2.contourArea(refined_back) / (SCALE ** 2)
        consensus_area = cv2.contourArea(consensus_pts) / (SCALE ** 2)

        # ── Step 7.9: 生成 diff 可视化图 ──
        diff_image = CrossValidator._generate_diff_image(
            front_mask, back_mask, consensus_mask, canvas_w, canvas_h
        )
        diff_b64 = CrossValidator._encode_image(diff_image)

        # ── Step 7.10: 从共识 mask 生成透明 PNG ──
        transparent_png = CrossValidator._generate_transparent_png(
            front_result.get("rectified_png_b64", ""),
            consensus_outline,
            w_mm, h_mm
        )

        _log.info(
            "Step 7: 正反面交叉校验 - 完成 (正面=%.0fmm², 背面=%.0fmm², 共识=%.0fmm²)",
            front_area, back_area, consensus_area
        )

        return {
            "outline": consensus_outline,
            "front_area_mm2": round(front_area, 1),
            "back_area_mm2": round(back_area, 1),
            "consensus_area_mm2": round(consensus_area, 1),
            "diff_image_b64": diff_b64,
            "transparent_pcb_b64": transparent_png,
        }

    # ─────────────────────────────────────────────────────────────────
    #  辅助方法
    # ─────────────────────────────────────────────────────────────────

    @staticmethod
    def _outline_to_canvas(outline_mm: list[dict], canvas_w: int, canvas_h: int) -> np.ndarray:
        """将 mm 轮廓转换为画布像素坐标"""
        pts = []
        for p in outline_mm:
            x = round(p["x_mm"] / canvas_w * canvas_w) if canvas_w > 0 else 0
            y = round(p["y_mm"] / canvas_h * canvas_h) if canvas_h > 0 else 0
            pts.append([x, y])
        return np.array(pts, dtype=np.float32)

    @staticmethod
    def _canvas_to_outline(pts: np.ndarray, canvas_w: int, canvas_h: int, w_mm: float, h_mm: float) -> list[dict]:
        """将画布像素坐标转换回 mm 轮廓"""
        outline = []
        for pt in pts:
            x_mm = round(pt[0] / SCALE, 3)
            y_mm = round(pt[1] / SCALE, 3)
            outline.append({"x_mm": x_mm, "y_mm": y_mm})
        return outline

    @staticmethod
    def _mirror_horizontal(pts: np.ndarray, canvas_w: int) -> np.ndarray:
        """水平镜像（以画布中心为轴，左↔右互换）"""
        center_x = canvas_w / 2.0
        mirrored = pts.copy()
        mirrored[:, 0] = 2 * center_x - pts[:, 0]
        return mirrored

    @staticmethod
    def _centroid(pts: np.ndarray) -> tuple[float, float]:
        """计算质心"""
        if len(pts) == 0:
            return (0.0, 0.0)
        cx = np.mean(pts[:, 0])
        cy = np.mean(pts[:, 1])
        return (float(cx), float(cy))

    @staticmethod
    def _translate(pts: np.ndarray, offset: tuple[float, float]) -> np.ndarray:
        """平移轮廓"""
        result = pts.copy()
        result[:, 0] += offset[0]
        result[:, 1] += offset[1]
        return result

    @staticmethod
    def _iou_refine(front_pts: np.ndarray, back_pts: np.ndarray, canvas_w: int, canvas_h: int) -> tuple[np.ndarray, float]:
        """IoU 微调：小范围滑动窗口找最佳位置

        搜索范围：±1% 画布尺寸（约 10 像素）
        步长：1 像素
        """
        front_mask = np.zeros((canvas_h, canvas_w), dtype=np.uint8)
        cv2.fillPoly(front_mask, [front_pts.astype(np.int32)], 255)

        search_range = max(5, int(min(canvas_w, canvas_h) * 0.01))
        best_iou = 0.0
        best_offset = (0, 0)

        for dx in range(-search_range, search_range + 1):
            for dy in range(-search_range, search_range + 1):
                shifted = back_pts.copy()
                shifted[:, 0] += dx
                shifted[:, 1] += dy

                back_mask = np.zeros((canvas_h, canvas_w), dtype=np.uint8)
                cv2.fillPoly(back_mask, [shifted.astype(np.int32)], 255)

                intersection = np.sum(cv2.bitwise_and(front_mask, back_mask) > 0)
                union = np.sum(cv2.bitwise_or(front_mask, back_mask) > 0)
                iou = intersection / union if union > 0 else 0.0

                if iou > best_iou:
                    best_iou = iou
                    best_offset = (dx, dy)

        refined = back_pts.copy()
        refined[:, 0] += best_offset[0]
        refined[:, 1] += best_offset[1]
        return refined, best_iou

    @staticmethod
    def _generate_diff_image(front_mask: np.ndarray, back_mask: np.ndarray, consensus_mask: np.ndarray, w: int, h: int) -> np.ndarray:
        """生成 diff 可视化图

        颜色编码：
          - 绿色 (0,255,0) = 交集（共识可靠）
          - 红色 (255,0,0) = 仅正面有（正面毛刺）
          - 蓝色 (0,0,255) = 仅背面有（背面毛刺）
        """
        diff = np.zeros((h, w, 3), dtype=np.uint8)

        # 绿色：交集
        intersection = consensus_mask > 0
        diff[intersection] = [0, 255, 0]

        # 红色：仅正面
        only_front = (front_mask > 0) & (consensus_mask == 0)
        diff[only_front] = [255, 0, 0]

        # 蓝色：仅背面
        only_back = (back_mask > 0) & (consensus_mask == 0)
        diff[only_back] = [0, 0, 255]

        return diff

    @staticmethod
    def _encode_image(img: np.ndarray) -> str:
        """编码图像为 base64"""
        _, buf = cv2.imencode(".png", img)
        return base64.b64encode(buf).decode("ascii")

    @staticmethod
    def _generate_transparent_png(rectified_b64: str, outline: list[dict], w_mm: float, h_mm: float) -> str:
        """从共识轮廓生成透明 PNG"""
        if not rectified_b64:
            return ""

        try:
            # 解码原图
            img_bytes = base64.b64decode(rectified_b64)
            nparr = np.frombuffer(img_bytes, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if img is None:
                return ""

            h_px, w_px = img.shape[:2]

            # 生成 mask
            mask = np.zeros((h_px, w_px), dtype=np.uint8)
            if len(outline) >= 3:
                pts = np.array([
                    [round(p["x_mm"] / w_mm * w_px), round(p["y_mm"] / h_mm * h_px)]
                    for p in outline
                ], dtype=np.int32)
                cv2.fillPoly(mask, [pts], 255)

            # 生成透明 PNG
            from ..vision import _make_transparent
            transparent_bytes = _make_transparent(img, mask)
            return base64.b64encode(transparent_bytes).decode("ascii")

        except Exception as e:
            _log.warning("生成透明 PNG 失败: %s", e)
            return rectified_b64