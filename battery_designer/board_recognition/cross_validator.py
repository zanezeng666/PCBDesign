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

from ..core.logger import get_logger

_log = get_logger(__name__)

# 共享画布缩放因子（mm → canvas pixel）
# SCALE=50 给凹槽足够分辨率（20px深的凹槽在画布上有100px深）
SCALE = 50.0


class CrossValidator:
    """正反面轮廓交叉校验器

    用正反面两张 PCB 轮廓的交集消除单面检测的毛刺/阴影。
    """

    @staticmethod
    def process_images(
        front_image_bytes: bytes,
        back_image_bytes: bytes,
        width_mm: float,
        height_mm: float,
    ) -> dict[str, Any]:
        """输入两张图片，输出正反面交叉校验结果（高级接口）

        自动对两张图片运行 pipeline 并执行交叉校验。

        Args:
            front_image_bytes: 正面图片字节
            back_image_bytes: 背面图片字节
            width_mm: PCB 宽度 (mm)
            height_mm: PCB 高度 (mm)

        Returns:
            {
                "front_result": 正面 pipeline.run() 结果,
                "back_result": 背面 pipeline.run() 结果,
                "consensus_outline": 共识轮廓 (mm),
                "consensus_area_mm2": 共识面积 (mm²),
                "diff_image_b64": diff 可视化图 (base64),
                "transparent_pcb_b64": 共识轮廓的透明 PNG,
            }
        """
        # 延迟导入避免循环依赖
        from .pipeline import PCBRecognitionPipeline
        
        _log.info("CrossValidator.process_images: 开始处理正反面图片")
        
        # 创建 pipeline 实例
        pipeline = PCBRecognitionPipeline()
        
        # 处理正面图片
        _log.info("处理正面图片...")
        front_result = pipeline.run(front_image_bytes, width_mm, height_mm)
        
        # 处理背面图片
        _log.info("处理背面图片...")
        back_result = pipeline.run(back_image_bytes, width_mm, height_mm)
        
        # 执行交叉校验
        _log.info("执行交叉校验...")
        consensus_result = CrossValidator.validate(
            front_result, back_result, width_mm, height_mm
        )
        
        _log.info("CrossValidator.process_images: 完成")
        
        return {
            "front_result": front_result,
            "back_result": back_result,
            "consensus_outline": consensus_result["outline"],
            "consensus_area_mm2": consensus_result["consensus_area_mm2"],
            "front_area_mm2": consensus_result["front_area_mm2"],
            "back_area_mm2": consensus_result["back_area_mm2"],
            "diff_image_b64": consensus_result["diff_image_b64"],
            "transparent_pcb_b64": consensus_result["transparent_pcb_b64"],
            "transparent_pcb_back_b64": consensus_result["transparent_pcb_back_b64"],
        }

    @staticmethod
    def validate(
        front_result: dict[str, Any],
        back_result: dict[str, Any],
        width_mm: float | None = None,
        height_mm: float | None = None,
    ) -> dict[str, Any]:
        """执行正反面交叉校验

        Args:
            front_result: 正面 pipeline.run() 结果，含 outline/pixels_per_mm/width_mm/height_mm
            back_result: 背面 pipeline.run() 结果
            width_mm: PCB 宽度 (mm)，若未提供则从 result 提取或使用默认值
            height_mm: PCB 高度 (mm)，若未提供则从 result 提取或使用默认值

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
        
        # 尺寸参数优先级：用户传入 > result 提取 > 默认值
        if width_mm is None:
            width_mm = max(front_result.get("width_mm", 40.0), back_result.get("width_mm", 40.0))
        if height_mm is None:
            height_mm = max(front_result.get("height_mm", 25.0), back_result.get("height_mm", 25.0))
        
        # 类型断言：确保不为 None（已在上面赋值或从 result 提取）
        assert width_mm is not None and height_mm is not None
        
        w_mm = width_mm
        h_mm = height_mm

        if len(front_outline) < 3 or len(back_outline) < 3:
            _log.warning("轮廓点数不足，跳过交叉校验")
            return {
                "outline": front_outline,
                "front_area_mm2": 0.0,
                "back_area_mm2": 0.0,
                "consensus_area_mm2": 0.0,
                "diff_image_b64": "",
                "transparent_pcb_b64": front_result.get("transparent_pcb_b64", ""),
                "transparent_pcb_back_b64": back_result.get("transparent_pcb_b64", ""),
            }

        # ── Step 7.1: 转换到共享画布坐标 ──
        canvas_w = int(w_mm * SCALE)
        canvas_h = int(h_mm * SCALE)

        front_pts = CrossValidator._outline_to_canvas(front_outline, canvas_w, canvas_h, w_mm, h_mm)
        back_pts = CrossValidator._outline_to_canvas(back_outline, canvas_w, canvas_h, w_mm, h_mm)

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

        # ── Step 7.5: 生成 mask ──
        front_mask = np.zeros((canvas_h, canvas_w), dtype=np.uint8)
        back_mask = np.zeros((canvas_h, canvas_w), dtype=np.uint8)
        cv2.fillPoly(front_mask, [front_pts.astype(np.int32)], 255)
        cv2.fillPoly(back_mask, [refined_back.astype(np.int32)], 255)

        # ── Step 7.6: 共识策略 ──
        # 合并正反面：对每列取最凹陷的上下边界，保留任一面检测到的凹槽。
        # 正面检测到凹槽但背面漏检（或反之）时，凹槽仍保留在共识轮廓中。
        consensus_mask = CrossValidator._merge_boundary_masks(front_mask, back_mask)

        from .pcb_rectangular_contour_refiner import PCBRectangularContourRefiner
        refiner = PCBRectangularContourRefiner()
        _, ortho_contour = refiner.refine(consensus_mask)
        consensus_pts = ortho_contour.reshape(-1, 2)
        _log.info(
            "共识策略: 合并正反面凹槽（取最深边界）, IoU=%.3f, %d点",
            best_iou, len(consensus_pts),
        )

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

        # ── Step 7.10: 从共识轮廓分别生成正反面透明 PNG ──
        transparent_png_front = CrossValidator._generate_transparent_png(
            front_result.get("rectified_png_b64", ""),
            consensus_outline,
            w_mm, h_mm
        )
        transparent_png_back = CrossValidator._generate_transparent_png(
            back_result.get("rectified_png_b64", ""),
            consensus_outline,
            w_mm, h_mm
        )

        _log.info(
            "Step 7: 正反面交叉校验 - 完成 (正面=%.0fmm2, 背面=%.0fmm2, 共识=%.0fmm2)",
            front_area, back_area, consensus_area
        )

        return {
            "outline": consensus_outline,
            "front_area_mm2": round(front_area, 1),
            "back_area_mm2": round(back_area, 1),
            "consensus_area_mm2": round(consensus_area, 1),
            "diff_image_b64": diff_b64,
            "transparent_pcb_b64": transparent_png_front,
            "transparent_pcb_back_b64": transparent_png_back,
        }

    # ─────────────────────────────────────────────────────────────────
    #  辅助方法
    # ─────────────────────────────────────────────────────────────────

    @staticmethod
    def _merge_boundary_masks(front_mask: np.ndarray, back_mask: np.ndarray) -> np.ndarray:
        """合并正反面 mask：对每列取最凹陷的上下边界。

        对每列 x：
          merged_top[x]    = max(front_top[x], back_top[x])     ← 更远离上边缘
          merged_bottom[x] = min(front_bottom[x], back_bottom[x]) ← 更远离下边缘

        如果任一面在某位置检测到凹槽（边界更靠内），共识结果保留该凹槽。
        同时做行方向合并（左右边界），处理纵向凹槽。
        """
        h, w = front_mask.shape

        # ── 列方向合并（top/bottom 边界）──
        col_mask = np.zeros((h, w), dtype=np.uint8)
        for x in range(w):
            f_col = np.where(front_mask[:, x] > 0)[0]
            b_col = np.where(back_mask[:, x] > 0)[0]
            has_f, has_b = len(f_col) > 0, len(b_col) > 0
            if not has_f and not has_b:
                continue
            if has_f and has_b:
                top = max(f_col[0], b_col[0])
                bottom = min(f_col[-1], b_col[-1])
            elif has_f:
                top, bottom = int(f_col[0]), int(f_col[-1])
            else:
                top, bottom = int(b_col[0]), int(b_col[-1])
            if top <= bottom:
                col_mask[top:bottom + 1, x] = 255

        # ── 行方向合并（left/right 边界）──
        row_mask = np.zeros((h, w), dtype=np.uint8)
        for y in range(h):
            f_row = np.where(front_mask[y, :] > 0)[0]
            b_row = np.where(back_mask[y, :] > 0)[0]
            has_f, has_b = len(f_row) > 0, len(b_row) > 0
            if not has_f and not has_b:
                continue
            if has_f and has_b:
                left = max(f_row[0], b_row[0])
                right = min(f_row[-1], b_row[-1])
            elif has_f:
                left, right = int(f_row[0]), int(f_row[-1])
            else:
                left, right = int(b_row[0]), int(b_row[-1])
            if left <= right:
                row_mask[y, left:right + 1] = 255

        # 取两个方向的交集（列合并 ∩ 行合并）
        consensus = cv2.bitwise_and(col_mask, row_mask)

        # 形态学闭合：填补边界合并可能产生的小缝隙
        kernel = np.ones((3, 3), np.uint8)
        consensus = cv2.morphologyEx(consensus, cv2.MORPH_CLOSE, kernel)

        _log.info(
            "边界合并: 列方向面积=%d, 行方向面积=%d, 合并后面积=%d",
            int(np.sum(col_mask > 0)), int(np.sum(row_mask > 0)), int(np.sum(consensus > 0)),
        )
        return consensus

    @staticmethod
    def _outline_to_canvas(
        outline_mm: list[dict],
        canvas_w: int,
        canvas_h: int,
        w_mm: float,
        h_mm: float,
    ) -> np.ndarray:
        """将 mm 轮廓转换为画布像素坐标

        Args:
            outline_mm: mm单位的轮廓点列表
            canvas_w: 画布宽度（像素）
            canvas_h: 画布高度（像素）
            w_mm: 实际宽度（毫米）
            h_mm: 实际高度（毫米）
        """
        pts = []
        for p in outline_mm:
            x = round(p["x_mm"] / w_mm * canvas_w) if w_mm > 0 else 0
            y = round(p["y_mm"] / h_mm * canvas_h) if h_mm > 0 else 0
            pts.append([x, y])
        return np.array(pts, dtype=np.float32)

    @staticmethod
    def _canvas_to_outline(
        pts: np.ndarray,
        canvas_w: int,
        canvas_h: int,
        w_mm: float,
        h_mm: float,
    ) -> list[dict]:
        """将画布像素坐标转换回 mm 轮廓

        Args:
            pts: 像素坐标点数组
            canvas_w: 画布宽度（像素）
            canvas_h: 画布高度（像素）
            w_mm: 实际宽度（毫米）
            h_mm: 实际高度（毫米）
        """
        outline = []
        for pt in pts:
            x_mm = round(pt[0] / canvas_w * w_mm, 3) if canvas_w > 0 else 0.0
            y_mm = round(pt[1] / canvas_h * h_mm, 3) if canvas_h > 0 else 0.0
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
        """计算质心（使用 bounding box 中心，避免顶点密度不均匀导致偏移）"""
        if len(pts) == 0:
            return (0.0, 0.0)
        x_min, x_max = np.min(pts[:, 0]), np.max(pts[:, 0])
        y_min, y_max = np.min(pts[:, 1]), np.max(pts[:, 1])
        return ((x_min + x_max) / 2.0, (y_min + y_max) / 2.0)

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

        search_range = max(5, int(min(canvas_w, canvas_h) * 0.03))
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

            # 生成 mask 并直接创建 RGBA（与 TransparentPNGGenerator 一致：硬alpha，无腐蚀无去污）
            if len(outline) >= 3:
                pts = np.array([
                    [round(p["x_mm"] / w_mm * w_px), round(p["y_mm"] / h_mm * h_px)]
                    for p in outline
                ], dtype=np.int32)
                mask = np.zeros((h_px, w_px), dtype=np.uint8)
                cv2.fillPoly(mask, [pts], 255)
            else:
                mask = np.zeros((h_px, w_px), dtype=np.uint8)

            img_rgba = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
            img_rgba[:, :, 3] = mask

            success, encoded = cv2.imencode('.png', img_rgba)
            if not success:
                return ""
            return base64.b64encode(encoded.tobytes()).decode("ascii")

        except Exception as e:
            _log.warning("生成透明 PNG 失败: %s", e)
            return rectified_b64