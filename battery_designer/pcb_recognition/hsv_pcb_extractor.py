"""HSV PCB提取

功能：使用HSV颜色空间从校正后的图片中提取PCB区域。

输入：校正后图片字节
输出：PCB轮廓多边形 + 提取的PCB图像
"""

from __future__ import annotations

import logging
from typing import List, Tuple

import cv2
import numpy as np

from ..logger import get_logger

_log = get_logger(__name__)


class HSVPCBExtractor:
    """HSV PCB提取器

    基于HSV颜色空间的PCB区域提取，支持多种PCB颜色（绿色、蓝色、黑色等）。
    """

    def __init__(self):
        """初始化提取器"""
        # PCB常见颜色的HSV范围
        self.color_ranges = [
            # 绿色PCB (最常见)
            {
                "name": "green",
                "lower": np.array([35, 40, 40]),
                "upper": np.array([85, 255, 255]),
            },
            # 蓝色PCB
            {
                "name": "blue",
                "lower": np.array([90, 40, 40]),
                "upper": np.array([130, 255, 255]),
            },
            # 黑色/深色PCB
            {
                "name": "dark",
                "lower": np.array([0, 0, 0]),
                "upper": np.array([180, 255, 80]),
            },
        ]

    def extract(self, image_bytes: bytes) -> dict:
        """提取PCB区域

        Args:
            image_bytes: 校正后图片字节

        Returns:
            {
                "pcb_mask": np.ndarray,       # PCB掩码
                "pcb_contour": np.ndarray,    # PCB轮廓
                "color_name": str,            # 检测到的PCB颜色
                "pcb_area_ratio": float,      # PCB面积占比
            }
        """
        _log.info("HSV PCB提取: 开始")

        # ── Step 1: 解码图片 ──
        try:
            nparr = np.frombuffer(image_bytes, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            if img is None:
                raise ValueError("Failed to decode image")

            h, w = img.shape[:2]
            _log.debug("图片尺寸: %dx%d", w, h)

        except Exception as e:
            _log.error("HSV PCB提取: 图片解码失败 - %s", e)
            raise ValueError(f"Failed to decode image: {e}")

        # ── Step 2: 转换到HSV空间 ──
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        # ── Step 3: 尝试不同颜色范围 ──
        best_mask = None
        best_area = 0
        best_color = ""

        for color_range in self.color_ranges:
            mask = cv2.inRange(
                hsv,
                color_range["lower"],
                color_range["upper"],
            )

            area = cv2.countNonZero(mask)

            if area > best_area:
                best_area = area
                best_mask = mask
                best_color = color_range["name"]

        if best_mask is None:
            _log.error("HSV PCB提取: 未找到PCB区域")
            raise ValueError("No PCB region found")

        # ── Step 4: 形态学处理 ──
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        best_mask = cv2.morphologyEx(best_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        best_mask = cv2.morphologyEx(best_mask, cv2.MORPH_OPEN, kernel, iterations=1)

        # ── Step 5: 提取轮廓 ──
        contours, _ = cv2.findContours(
            best_mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )

        if not contours:
            raise ValueError("No PCB contour found")

        # 选择最大轮廓
        pcb_contour = max(contours, key=cv2.contourArea)

        # 计算面积占比
        pcb_area_ratio = cv2.contourArea(pcb_contour) / (w * h)

        _log.info(
            "HSV PCB提取: 完成 (颜色=%s, 面积占比=%.1f%%)",
            best_color,
            pcb_area_ratio * 100,
        )

        return {
            "pcb_mask": best_mask,
            "pcb_contour": pcb_contour,
            "color_name": best_color,
            "pcb_area_ratio": pcb_area_ratio,
        }


# ── 测试代码 ──
if __name__ == "__main__":
    extractor = HSVPCBExtractor()
    print("HSV PCB提取器初始化完成")