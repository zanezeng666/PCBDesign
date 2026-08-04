"""纸色模型构建

功能：基于PCB轮廓构建纸色模型，用于验证和修正轮廓。

输入：PCB轮廓多边形 + 原图
输出：优化后的PCB轮廓
"""

from __future__ import annotations

import logging

import cv2
import numpy as np

from ..logger import get_logger

_log = get_logger(__name__)


class PaperModelBuilder:
    """纸色模型构建器

    基于物理约束（矩形、对称性等）优化PCB轮廓。
    """

    def __init__(self):
        """初始化构建器"""
        pass

    def build(
        self,
        image_bytes: bytes,
        pcb_contour: np.ndarray,
        pixels_per_mm: float,
    ) -> dict:
        """构建纸色模型并优化轮廓

        Args:
            image_bytes: 校正后图片字节
            pcb_contour: HSV提取的PCB轮廓
            pixels_per_mm: 像素密度

        Returns:
            {
                "refined_contour": np.ndarray,  # 优化后的轮廓
                "paper_model": dict,            # 纸色模型参数
                "is_rectangular": bool,         # 是否为矩形
            }
        """
        _log.info("纸色模型构建: 开始")

        # ── Step 1: 解码图片 ──
        try:
            nparr = np.frombuffer(image_bytes, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            if img is None:
                raise ValueError("Failed to decode image")

        except Exception as e:
            _log.error("纸色模型构建: 图片解码失败 - %s", e)
            raise ValueError(f"Failed to decode image: {e}")

        # ── Step 2: 多边形近似 ──
        epsilon = 0.01 * cv2.arcLength(pcb_contour, True)
        approx = cv2.approxPolyDP(pcb_contour, epsilon, True)

        # ── Step 3: 判断是否为矩形 ──
        is_rectangular = len(approx) == 4 or self._is_approx_rectangular(approx)

        # ── Step 4: 构建纸色模型 ──
        if is_rectangular:
            # 矩形PCB：使用最小外接矩形
            rect = cv2.minAreaRect(pcb_contour)
            box = cv2.boxPoints(rect)
            refined_contour = np.int0(box).reshape(-1, 1, 2)

            paper_model = {
                "type": "rectangular",
                "width_mm": rect[1][0] / pixels_per_mm,
                "height_mm": rect[1][1] / pixels_per_mm,
                "rotation_deg": rect[2],
            }

        else:
            # 异形PCB：保持原轮廓
            refined_contour = pcb_contour

            paper_model = {
                "type": "irregular",
                "vertex_count": len(approx),
            }

        _log.info(
            "纸色模型构建: 完成 (类型=%s, 顶点数=%d)",
            paper_model["type"],
            len(refined_contour),
        )

        return {
            "refined_contour": refined_contour,
            "paper_model": paper_model,
            "is_rectangular": is_rectangular,
        }

    def _is_approx_rectangular(self, contour: np.ndarray, threshold: float = 0.1) -> bool:
        """判断轮廓是否近似为矩形

        Args:
            contour: 轮廓点集
            threshold: 凸度阈值

        Returns:
            是否近似为矩形
        """
        # 计算凸包
        hull = cv2.convexHull(contour)

        # 计算轮廓面积与凸包面积的比值
        contour_area = cv2.contourArea(contour)
        hull_area = cv2.contourArea(hull)

        if hull_area == 0:
            return False

        ratio = contour_area / hull_area

        return ratio > (1 - threshold)


# ── 测试代码 ──
if __name__ == "__main__":
    builder = PaperModelBuilder()
    print("纸色模型构建器初始化完成")