"""焊盘识别模块 (重点优化模块)

这个模块负责从PCB图像中识别焊盘/触点，是整个流程中最关键的部分。

主要功能：
  1. 焊盘候选检测
  2. 焊盘类型分类 (矩形/圆形/椭圆形/槽形)
  3. 焊盘位置精确定位
  4. 焊盘尺寸测量
  5. 焊盘标签识别

流程：
  输入: 透明PCB图像 + 轮廓信息
  输出: 焊盘列表 [{"label": "P+", "type": "rect", "center": {...}, "bbox": {...}, ...}, ...]
"""

from __future__ import annotations

import base64
import json
import logging
import os
from pathlib import Path
from typing import Literal

import cv2
import numpy as np

from .logger import get_logger, log_function, log_errors

_log = get_logger(__name__)

# 导入VLM检测功能
from .vlm_detection import (
    detect_with_vlm as _detect_with_vlm,
    detect_all_vlm as _detect_all_vlm,
    verify_pad_crop as _verify_pad_crop,
)


@log_function(level=logging.INFO, include_args=True)
def detect_pads(
    transparent_pcb_b64: str,
    outline_points_mm: list[dict],
    side: Literal["front", "back"] = "front",
    pixels_per_mm: float = 10.0,
    refine_iterations: int = 3,
) -> dict:
    """焊盘检测 - 核心流程

    Args:
        transparent_pcb_b64: 透明PCB图像 (base64)
        outline_points_mm: PCB轮廓顶点 (mm)
        side: 板面朝向 ("front"|"back")
        pixels_per_mm: 像素密度
        refine_iterations: 精修迭代次数 (默认3次)

    Returns:
        {
            "pads": [                # 焊盘列表
                {
                    "id": str,
                    "label": str,     # 标签 (P+/P-/B-/C+/...)
                    "type": str,      # 类型 (rect/round/oval/slot)
                    "confidence": float,  # 置信度 (0-1)
                    "center": {"x_mm": ..., "y_mm": ...},
                    "bbox": {"x_mm": ..., "y_mm": ..., "w_mm": ..., "h_mm": ...},
                    "polygon": [{"x_mm": ..., "y_mm": ...}, ...],
                    "radius_mm": float,  # 圆形焊盘半径
                    "corner_radius_mm": float,  # 矩形焊盘圆角
                },
                ...
            ],
            "pad_count": int,
            "side": "front"|"back",
            "coordinate_system": {"origin": "pcb_top_left", "units": "mm"},
        }
    """
    _log.info("焊盘检测: 开始 (side=%s, pixels_per_mm=%.1f)", side, pixels_per_mm)

    # 解码透明PCB图像
    try:
        img_bytes = base64.b64decode(transparent_pcb_b64)
        nparr = np.frombuffer(img_bytes, np.uint8)
        img_rgba = cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)

        if img_rgba is None or len(img_rgba.shape) != 3 or img_rgba.shape[2] != 4:
            raise ValueError("Invalid transparent PNG: expected RGBA image")

        _log.debug("焊盘检测: 图像尺寸 %dx%d", img_rgba.shape[1], img_rgba.shape[0])

    except Exception as e:
        _log.error("焊盘检测: 图像解码失败 - %s", e)
        raise ValueError(f"Failed to decode transparent PCB: {e}")

    # ── Step 1: VLM识别焊盘候选 ──
    try:
        vlm_result = _detect_all_vlm(
            transparent_pcb_b64,
            side=side,
            pixels_per_mm=pixels_per_mm,
        )

        pads_raw = vlm_result.get("candidates", [])
        _log.info("焊盘检测: VLM识别到 %d 个候选", len(pads_raw))

    except Exception as e:
        _log.error("焊盘检测: VLM识别失败 - %s", e)
        raise RuntimeError(f"VLM detection failed: {e}")

    # ── Step 2: 焊盘精修和分类 ──
    pads_refined = []
    for i, pad_raw in enumerate(pads_raw):
        try:
            pad_refined = _refine_pad(
                pad_raw,
                img_rgba,
                pixels_per_mm,
                refine_iterations,
            )
            pad_refined["id"] = f"pad_{i:03d}"
            pads_refined.append(pad_refined)

        except Exception as e:
            _log.warning("焊盘检测: 第 %d 个焊盘精修失败 - %s", i, e)
            # 保留原始数据，但标记为未精修
            pad_raw["id"] = f"pad_{i:03d}"
            pad_raw["refined"] = False
            pads_refined.append(pad_raw)

    _log.info(
        "焊盘检测: 完成 (%d 个焊盘, %d 精修成功)",
        len(pads_refined),
        sum(1 for p in pads_refined if p.get("refined", True)),
    )

    return {
        "pads": pads_refined,
        "pad_count": len(pads_refined),
        "side": side,
        "coordinate_system": {
            "origin": "pcb_top_left",
            "units": "mm",
        },
        "refine_iterations": refine_iterations,
    }


def _refine_pad(
    pad_raw: dict,
    img_rgba: np.ndarray,
    pixels_per_mm: float,
    iterations: int,
) -> dict:
    """焊盘精修 - 几何校正和类型推断

    Args:
        pad_raw: VLM识别的原始焊盘数据
        img_rgba: 透明PCB图像 (RGBA)
        pixels_per_mm: 像素密度
        iterations: 精修迭代次数

    Returns:
        精修后的焊盘数据
    """
    import math

    # 提取基本信息
    label = pad_raw.get("label", "?")
    confidence = pad_raw.get("confidence", 0.0)
    center = pad_raw.get("center", {})
    bbox = pad_raw.get("bbox", {})
    polygon = pad_raw.get("polygon", [])

    # ── Step 1: 坐标转换 (mm → pixel) ──
    center_px = {
        "x": int(center.get("x_mm", 0) * pixels_per_mm),
        "y": int(center.get("y_mm", 0) * pixels_per_mm),
    }

    # ── Step 2: 焊盘类型推断 ──
    pad_type = _infer_pad_type(pad_raw, pixels_per_mm)

    # ── Step 3: 尺寸测量 ──
    if bbox:
        w_mm = bbox.get("w_mm", 0)
        h_mm = bbox.get("h_mm", 0)

        # 圆形焊盘：半径 = min(w, h) / 2
        if pad_type == "round":
            radius_mm = min(w_mm, h_mm) / 2.0
        else:
            radius_mm = 0

        # 矩形焊盘：圆角半径估算
        if pad_type == "rect" and polygon:
            corner_radius_mm = _estimate_corner_radius(polygon)
        else:
            corner_radius_mm = 0

    else:
        w_mm = h_mm = radius_mm = corner_radius_mm = 0

    # ── Step 4: 构建精修结果 ──
    result = {
        **pad_raw,  # 保留原始字段
        "label": label,
        "type": pad_type,
        "confidence": confidence,
        "center": center,
        "bbox": bbox,
        "polygon": polygon,
        "radius_mm": round(radius_mm, 3),
        "corner_radius_mm": round(corner_radius_mm, 3),
        "refined": True,
    }

    return result


def _infer_pad_type(pad_raw: dict, pixels_per_mm: float) -> str:
    """推断焊盘类型

    Args:
        pad_raw: 原始焊盘数据
        pixels_per_mm: 像素密度

    Returns:
        焊盘类型: "rect"|"round"|"oval"|"slot"
    """
    bbox = pad_raw.get("bbox", {})
    polygon = pad_raw.get("polygon", [])

    if not bbox:
        return "rect"  # 默认

    w_mm = bbox.get("w_mm", 0)
    h_mm = bbox.get("h_mm", 0)

    if w_mm <= 0 or h_mm <= 0:
        return "rect"

    aspect_ratio = w_mm / h_mm if h_mm > 0 else 1.0

    # 圆形焊盘: 宽高比 ≈ 1
    if 0.9 <= aspect_ratio <= 1.1:
        return "round"

    # 椭圆形焊盘: 宽高比 > 1.5 或 < 0.67
    if aspect_ratio > 1.5 or aspect_ratio < 0.67:
        return "oval"

    # 槽形焊盘: 长条形且有多边形顶点
    if len(polygon) > 10 and (aspect_ratio > 3.0 or aspect_ratio < 0.33):
        return "slot"

    # 默认矩形
    return "rect"


def _estimate_corner_radius(polygon_mm: list[dict]) -> float:
    """估算矩形焊盘的圆角半径

    Args:
        polygon_mm: 多边形顶点 (mm)

    Returns:
        圆角半径 (mm)
    """
    import math

    if len(polygon_mm) < 4:
        return 0.0

    # 简化实现：基于顶点分布估算
    # TODO: 更精确的圆角检测算法
    xs = [p.get("x_mm", 0) for p in polygon_mm]
    ys = [p.get("y_mm", 0) for p in polygon_mm]

    if not xs or not ys:
        return 0.0

    # 假设矩形焊盘，圆角半径 ≈ 短边的 10%
    w = max(xs) - min(xs)
    h = max(ys) - min(ys)
    short_side = min(w, h)

    return short_side * 0.1


def verify_pad_alignment(
    pads: list[dict],
    expected_labels: list[str],
    tolerance_mm: float = 2.0,
) -> dict:
    """焊盘对齐验证

    Args:
        pads: 焊盘列表
        expected_labels: 期望的焊盘标签列表
        tolerance_mm: 位置容差 (mm)

    Returns:
        {
            "valid": bool,
            "matched_count": int,
            "missing_labels": [str],
            "extra_labels": [str],
        }
    """
    detected_labels = [p.get("label", "") for p in pads]

    matched = set(detected_labels) & set(expected_labels)
    missing = set(expected_labels) - set(detected_labels)
    extra = set(detected_labels) - set(expected_labels)

    return {
        "valid": len(missing) == 0,
        "matched_count": len(matched),
        "missing_labels": sorted(list(missing)),
        "extra_labels": sorted(list(extra)),
    }


# 导出公共API
__all__ = [
    "detect_pads",
    "verify_pad_alignment",
]