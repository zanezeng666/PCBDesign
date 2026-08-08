"""正反面交叉校验模块（Shapely 几何运算版）

功能：
  1. 背面轮廓水平镜像（像翻书一样，左↔右互换）
  2. 质心对齐 + IoU 微调
  3. 正反面多边形交集求共识轮廓
  4. 正交化精修（矩形+矩形凹槽+可选圆角）
  5. 生成 diff 可视化图（绿=共识 / 红=仅正面 / 蓝=仅背面）

设计原则：
  - 使用 Shapely 几何运算代替光栅化，核心运算（镜像/对齐/IoU/交集）全在 mm 坐标下完成
  - 彻底消除中间画布；仅在正交化精修和 diff 图渲染时使用最小画布（SCALE=10, 一次性）
  - IoU 搜索从 ~450 万像素画布的滑动窗口变为 O(n) 多边形运算
"""

from __future__ import annotations

import base64
import logging
from typing import Any

import cv2
import numpy as np
from shapely.geometry import Polygon, MultiPolygon
from shapely import affinity

from ..core.logger import get_logger

_log = get_logger(__name__)

# 仅用于正交化精修（refiner 需要像素 mask）和 diff 图渲染的画布缩放
# SCALE=10 → 60mm = 600px；refiner 的像素阈值（min_width=30px 等）在此尺度下有意义
SCALE = 10.0


class CrossValidator:
    """正反面轮廓交叉校验器

    用正反面两张 PCB 轮廓的交集消除单面检测的毛刺/阴影。
    所有核心运算使用 Shapely 几何运算，无需中间光栅化画布。
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
        """执行正反面交叉校验（Shapely 几何运算版）

        Args:
            front_result: 正面 pipeline.run() 结果，含 outline/width_mm/height_mm
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

        # 提取轮廓
        front_outline = front_result.get("outline", [])
        back_outline = back_result.get("outline", [])

        # 尺寸参数优先级：用户传入 > result 提取 > 默认值
        if width_mm is None:
            width_mm = max(front_result.get("width_mm", 40.0), back_result.get("width_mm", 40.0))
        if height_mm is None:
            height_mm = max(front_result.get("height_mm", 25.0), back_result.get("height_mm", 25.0))

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

        # ── Step 7.1: 转为 Shapely 多边形（mm 坐标） ──
        front_poly = CrossValidator._make_polygon(front_outline)
        back_poly = CrossValidator._make_polygon(back_outline)

        if front_poly is None or back_poly is None:
            _log.warning("无法构建多边形，跳过交叉校验")
            return {
                "outline": front_outline,
                "front_area_mm2": 0.0,
                "back_area_mm2": 0.0,
                "consensus_area_mm2": 0.0,
                "diff_image_b64": "",
                "transparent_pcb_b64": front_result.get("transparent_pcb_b64", ""),
                "transparent_pcb_back_b64": back_result.get("transparent_pcb_b64", ""),
            }

        # ── Step 7.2: 背面水平镜像（绕 w_mm/2 翻转 X 轴） ──
        mirrored_back = affinity.scale(back_poly, xfact=-1, yfact=1.0, origin=(w_mm / 2, h_mm / 2))

        # ── Step 7.3: 质心对齐 ──
        f_cx, f_cy = CrossValidator._bounds_center(front_poly)
        b_cx, b_cy = CrossValidator._bounds_center(mirrored_back)
        aligned_back = affinity.translate(mirrored_back, f_cx - b_cx, f_cy - b_cy)

        # ── Step 7.4: IoU 微调（纯几何搜索，无光栅化） ──
        refined_back, best_iou = CrossValidator._iou_refine(front_poly, aligned_back, w_mm, h_mm)
        _log.info("IoU 微调: 最佳 IoU=%.3f", best_iou)

        # ── Step 7.5: 共识 = 正反面多边形交集 ──
        consensus = front_poly.intersection(refined_back)
        if consensus.is_empty or consensus.area < 1.0:
            _log.warning("交集为空或过小，使用正面轮廓")
            consensus = front_poly
        if isinstance(consensus, MultiPolygon):
            consensus = max(consensus.geoms, key=lambda g: g.area)
            _log.info("交集为 MultiPolygon，取最大面片")
        _log.info(
            "共识策略: 正反面交集, IoU=%.3f, 共识面积=%.1fmm^2",
            best_iou, consensus.area,
        )

        # ── Step 7.6: 直接使用共识多边形（无需重新正交化精修） ──
        #
        # 正反面已在 pipeline 中各自完成正交化精修（原始高分辨率），
        # 共识交集 = 两个正交多边形的 intersection，结果本身就是正交多边形。
        # 不再在 SCALE=10 低分辨率画布上重复精修——那会过滤掉凹槽。
        #
        # 用 simplify() 消除 intersection 可能产生的微小共线碎片
        consensus_simplified = consensus.simplify(0.01, preserve_topology=True)
        if consensus_simplified.is_empty or consensus_simplified.area < 1.0:
            consensus_simplified = consensus
        consensus_outline = CrossValidator._polygon_to_outline(consensus_simplified)
        _log.info("共识轮廓: %d 点 (直接从 Shapely 几何转换)", len(consensus_outline))

        canvas_w = int(w_mm * SCALE)
        canvas_h = int(h_mm * SCALE)
        consensus_mask = CrossValidator._polygon_to_mask(
            consensus_simplified, canvas_w, canvas_h, w_mm, h_mm
        )

        # ── Step 7.7: 面积（Shapely 直接在 mm² 计算） ──
        front_area = front_poly.area
        back_area = refined_back.area
        consensus_area = consensus.area

        # ── Step 7.8: diff 可视化图（最小画布渲染） ──
        front_mask_img = CrossValidator._polygon_to_mask(
            front_poly, canvas_w, canvas_h, w_mm, h_mm
        )
        back_mask_img = CrossValidator._polygon_to_mask(
            refined_back, canvas_w, canvas_h, w_mm, h_mm
        )
        diff_image = CrossValidator._generate_diff_image(
            front_mask_img, back_mask_img, consensus_mask, canvas_w, canvas_h
        )
        diff_b64 = CrossValidator._encode_image(diff_image)

        # ── Step 7.9: 从共识轮廓分别生成正反面透明 PNG ──
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
            "Step 7: 正反面交叉校验 - 完成 (正面=%.0fmm^2, 背面=%.0fmm^2, 共识=%.0fmm^2)",
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
    #  Shapely 几何运算辅助方法
    # ─────────────────────────────────────────────────────────────────

    @staticmethod
    def _make_polygon(outline_mm: list[dict]) -> Polygon | None:
        """从 mm 轮廓点列表构建 Shapely 多边形

        自动修复无效多边形（自相交等），取最大面片。
        """
        if len(outline_mm) < 3:
            return None
        coords = [(p["x_mm"], p["y_mm"]) for p in outline_mm]
        poly = Polygon(coords)
        if not poly.is_valid:
            poly = poly.buffer(0)
            if poly.is_empty:
                return None
            if isinstance(poly, MultiPolygon):
                poly = max(poly.geoms, key=lambda g: g.area)
        return poly

    @staticmethod
    def _bounds_center(poly: Polygon) -> tuple[float, float]:
        """包围盒中心（避免顶点密度不均匀导致偏移）"""
        minx, miny, maxx, maxy = poly.bounds
        return ((minx + maxx) / 2.0, (miny + maxy) / 2.0)

    @staticmethod
    def _iou_refine(
        front_poly: Polygon, back_poly: Polygon, w_mm: float, h_mm: float
    ) -> tuple[Polygon, float]:
        """IoU 微调：小范围搜索最佳偏移（纯几何运算，无光栅化）

        在 ±search_mm 范围内以 step_mm 步长搜索最佳 (dx, dy) 偏移，
        使正反面多边形 IoU 最大化。
        """
        search_mm = max(0.5, min(w_mm, h_mm) * 0.03)
        step_mm = 0.1
        n_steps = int(search_mm / step_mm)

        best_iou = CrossValidator._compute_iou(front_poly, back_poly)
        best_dx, best_dy = 0.0, 0.0

        for i in range(-n_steps, n_steps + 1):
            dx = i * step_mm
            for j in range(-n_steps, n_steps + 1):
                dy = j * step_mm
                if dx == 0.0 and dy == 0.0:
                    continue
                shifted = affinity.translate(back_poly, dx, dy)
                iou = CrossValidator._compute_iou(front_poly, shifted)
                if iou > best_iou:
                    best_iou = iou
                    best_dx, best_dy = dx, dy

        refined = affinity.translate(back_poly, best_dx, best_dy)
        _log.info(
            "IoU 搜索: 范围=±%.1fmm, 步长=%.1fmm, %d次迭代, 最优偏移=(%.1f,%.1f)",
            search_mm, step_mm, (2 * n_steps + 1) ** 2, best_dx, best_dy,
        )
        return refined, best_iou

    @staticmethod
    def _compute_iou(poly1: Polygon, poly2: Polygon) -> float:
        """计算两个多边形的 IoU"""
        inter = poly1.intersection(poly2)
        union_area = poly1.area + poly2.area - inter.area
        if union_area <= 0:
            return 0.0
        return inter.area / union_area

    # ─────────────────────────────────────────────────────────────────
    #  最小画布栅格化辅助方法（仅用于 refiner 输入和 diff 图）
    # ─────────────────────────────────────────────────────────────────

    @staticmethod
    def _polygon_to_mask(
        poly: Polygon, canvas_w: int, canvas_h: int, w_mm: float, h_mm: float
    ) -> np.ndarray:
        """将 Shapely 多边形栅格化为二值 mask（一次性 fillPoly）"""
        mask = np.zeros((canvas_h, canvas_w), dtype=np.uint8)
        if poly.is_empty:
            return mask
        ext_coords = list(poly.exterior.coords)
        pts = np.array([
            (int(x / w_mm * canvas_w), int(y / h_mm * canvas_h))
            for x, y in ext_coords
        ], dtype=np.int32)
        cv2.fillPoly(mask, [pts], 255)
        return mask

    @staticmethod
    def _px_to_mm_outline(
        pts: np.ndarray, canvas_w: int, canvas_h: int, w_mm: float, h_mm: float
    ) -> list[dict]:
        """画布像素坐标转 mm 轮廓"""
        outline = []
        for pt in pts:
            x_mm = round(float(pt[0]) / canvas_w * w_mm, 3) if canvas_w > 0 else 0.0
            y_mm = round(float(pt[1]) / canvas_h * h_mm, 3) if canvas_h > 0 else 0.0
            outline.append({"x_mm": x_mm, "y_mm": y_mm})
        return outline

    @staticmethod
    def _polygon_to_outline(poly: Polygon) -> list[dict]:
        """Shapely 多边形转 mm 轮廓"""
        outline = []
        for x, y in poly.exterior.coords:
            outline.append({"x_mm": round(x, 3), "y_mm": round(y, 3)})
        if len(outline) > 1 and outline[0] == outline[-1]:
            outline.pop()
        return outline

    # ─────────────────────────────────────────────────────────────────
    #  可视化辅助方法
    # ─────────────────────────────────────────────────────────────────

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
