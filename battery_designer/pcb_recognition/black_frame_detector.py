"""黑色方框检测

功能：在图片中检测黑色校准方框，返回方框位置和像素密度。

输入：图片字节 + 方框尺寸 (mm)
输出：方框四角坐标 (pixel) + 像素密度 (pixels/mm)
"""

from __future__ import annotations

import logging
import uuid
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np

from ..logger import get_logger

_log = get_logger(__name__)


class BlackFrameDetector:
    """黑色方框检测器

    检测图片中的黑色矩形方框，用于后续的透视校正和像素密度计算。
    """

    def __init__(
        self,
        min_area_ratio: float = 0.3,  # 最小面积占比 (相对于图片)
        max_area_ratio: float = 0.95,  # 最大面积占比
        black_threshold: int = 50,      # 黑色阈值 (0-255)
    ):
        """初始化检测器

        Args:
            min_area_ratio: 方框最小面积占比
            max_area_ratio: 方框最大面积占比
            black_threshold: 黑色像素阈值
        """
        self.min_area_ratio = min_area_ratio
        self.max_area_ratio = max_area_ratio
        self.black_threshold = black_threshold

    def detect(
        self,
        image_bytes: bytes,
        frame_width_mm: float,
        frame_height_mm: float,
    ) -> dict:
        """检测黑色方框

        Args:
            image_bytes: 图片字节 (JPEG/PNG)
            frame_width_mm: 方框宽度 (mm)
            frame_height_mm: 方框高度 (mm)

        Returns:
            {
                "corners": [(x1,y1), (x2,y2), (x3,y3), (x4,y4)],  # 方框四角 (pixel)
                "pixels_per_mm": float,  # 像素密度
                "frame_width_px": int,   # 方框宽度 (pixel)
                "frame_height_px": int,  # 方框高度 (pixel)
                "detection_id": str,     # 检测ID
                "debug_image_b64": str,  # 调试图片 (base64)
            }
        """
        _log.info(
            "黑色方框检测: 开始 (方框尺寸 %.1f×%.1f mm)",
            frame_width_mm,
            frame_height_mm,
        )

        # ── Step 1: 解码图片 ──
        try:
            nparr = np.frombuffer(image_bytes, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            if img is None:
                raise ValueError("Failed to decode image")

            h, w = img.shape[:2]
            _log.debug("图片尺寸: %dx%d", w, h)

        except Exception as e:
            _log.error("黑色方框检测: 图片解码失败 - %s", e)
            raise ValueError(f"Failed to decode image: {e}")

        # ── Step 2: 灰度化 + 二值化 ──
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(
            gray,
            self.black_threshold,
            255,
            cv2.THRESH_BINARY_INV,
        )

        # ── Step 3: 查找轮廓 ──
        contours, _ = cv2.findContours(
            binary,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )

        if not contours:
            _log.error("黑色方框检测: 未找到任何轮廓")
            raise ValueError("No contours found in image")

        # ── Step 4: 筛选黑色方框 ──
        frame_contour = self._find_frame_contour(contours, w, h)

        if frame_contour is None:
            _log.error("黑色方框检测: 未找到符合条件的方框")
            raise ValueError("No valid frame contour found")

        # ── Step 5: 提取四角坐标 ──
        epsilon = 0.02 * cv2.arcLength(frame_contour, True)
        approx = cv2.approxPolyDP(frame_contour, epsilon, True)

        if len(approx) != 4:
            _log.warning("黑色方框检测: 轮廓不是四边形 (%d 个顶点)，使用最小外接矩形", len(approx))
            rect = cv2.minAreaRect(frame_contour)
            box = cv2.boxPoints(rect)
            corners = [tuple(map(int, pt)) for pt in box]
        else:
            corners = [tuple(pt[0]) for pt in approx]

        # 排序四角：左上、右上、右下、左下
        corners = self._order_corners(corners)

        # ── Step 6: 计算像素密度 ──
        # 计算方框的像素尺寸（取上下边平均值）
        top_width = np.linalg.norm(np.array(corners[0]) - np.array(corners[1]))
        bottom_width = np.linalg.norm(np.array(corners[3]) - np.array(corners[2]))
        frame_width_px = (top_width + bottom_width) / 2

        # 计算像素密度
        pixels_per_mm = frame_width_px / frame_width_mm

        # 计算高度（取左右边平均值）
        left_height = np.linalg.norm(np.array(corners[0]) - np.array(corners[3]))
        right_height = np.linalg.norm(np.array(corners[1]) - np.array(corners[2]))
        frame_height_px = (left_height + right_height) / 2

        _log.info(
            "黑色方框检测: 完成 (方框尺寸 %.0f×%.0f px, 密度 %.2f px/mm)",
            frame_width_px,
            frame_height_px,
            pixels_per_mm,
        )

        # ── Step 7: 生成调试图片 ──
        debug_img = img.copy()
        cv2.drawContours(debug_img, [frame_contour], -1, (0, 255, 0), 3)

        for i, (x, y) in enumerate(corners):
            cv2.circle(debug_img, (x, y), 10, (0, 0, 255), -1)
            cv2.putText(debug_img, str(i), (x + 15, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 2)

        # 编码为base64
        import base64
        _, buffer = cv2.imencode('.jpg', debug_img)
        debug_image_b64 = base64.b64encode(buffer).decode('utf-8')

        return {
            "corners": corners,
            "pixels_per_mm": pixels_per_mm,
            "frame_width_px": int(frame_width_px),
            "frame_height_px": int(frame_height_px),
            "detection_id": str(uuid.uuid4()).replace('-', ''),
            "debug_image_b64": debug_image_b64,
        }

    def _find_frame_contour(
        self,
        contours: list,
        img_width: int,
        img_height: int,
    ) -> np.ndarray | None:
        """从轮廓列表中找到黑色方框轮廓

        Args:
            contours: 轮廓列表
            img_width: 图片宽度
            img_height: 图片高度

        Returns:
            方框轮廓，如果未找到则返回None
        """
        img_area = img_width * img_height
        min_area = img_area * self.min_area_ratio
        max_area = img_area * self.max_area_ratio

        # 按面积降序排序
        contours = sorted(contours, key=cv2.contourArea, reverse=True)

        for contour in contours:
            area = cv2.contourArea(contour)

            # 检查面积范围
            if area < min_area or area > max_area:
                continue

            # 检查凸性
            if cv2.isContourConvex(contour):
                return contour

        return None

    def _order_corners(self, corners: list) -> list:
        """对四角进行排序：左上、右上、右下、左下

        Args:
            corners: 四角坐标列表

        Returns:
            排序后的四角坐标
        """
        # 转换为numpy数组
        pts = np.array(corners, dtype=np.float32)

        # 计算中心点
        center = np.mean(pts, axis=0)

        # 计算每个点相对于中心的角度
        angles = np.arctan2(pts[:, 1] - center[1], pts[:, 0] - center[0])

        # 按角度排序
        sorted_indices = np.argsort(angles)
        sorted_pts = pts[sorted_indices]

        # 确保第一个点是左上角（y最小）
        if sorted_pts[0, 1] > sorted_pts[1, 1]:
            sorted_pts = sorted_pts[[1, 0, 3, 2]]

        return [tuple(map(int, pt)) for pt in sorted_pts]


# ── 测试代码 ──
if __name__ == "__main__":
    # 测试用例
    detector = BlackFrameDetector()

    test_img_path = Path(__file__).parents[3] / "input" / "test.jpg"
    if test_img_path.exists():
        test_bytes = test_img_path.read_bytes()
        result = detector.detect(test_bytes, 85.0, 60.0)
        print(f"检测成功: 密度={result['pixels_per_mm']:.2f} px/mm")
    else:
        print("测试图片不存在，跳过测试")