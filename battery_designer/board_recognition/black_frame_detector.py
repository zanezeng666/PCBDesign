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

from ..core.logger import get_logger

_log = get_logger(__name__)


class BlackFrameDetector:
    """黑色方框检测器

    检测图片中的黑色矩形方框，用于后续的透视校正和像素密度计算。
    """

    def __init__(
        self,
        min_area_ratio: float = 0.3,  # 最小面积占比 (相对于图片)
        max_area_ratio: float = 0.95,  # 最大面积占比
        black_threshold: int = 70,      # 黑色阈值 (0-255)，提高到70以更好检测黑色方框
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

        # ── Step 2: Otsu + 自适应阈值混合 (借鉴 vision.py 的验证方法) ──
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Otsu 全局阈值
        _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # 自适应局部阈值
        adaptive = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 101, 20
        )
        
        # 混合两种阈值结果
        binary = cv2.bitwise_or(otsu, adaptive)
        
        # 形态学闭运算 - 连接断开的线条（对于线条组成的方框很重要）
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)

        # ── Step 3: 查找轮廓 ──
        contours, _ = cv2.findContours(
            binary,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )

        if not contours:
            _log.error("黑色方框检测: 未找到任何轮廓")
            raise ValueError("No contours found in image")

        # ── Step 4: 筛选黑色方框（使用长宽比辅助筛选） ──
        target_aspect = frame_width_mm / frame_height_mm
        frame_contour = self._find_frame_contour(contours, w, h, target_aspect)

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

        # ── Step 6.5: 计算黑框线条宽度 ──
        # 方法：分析黑框角点附近的线条宽度
        # 使用形态学骨架距离估计
        frame_border_mm = self._estimate_line_width(
            binary, frame_contour, corners, pixels_per_mm
        )

        _log.info(
            "黑框线条宽度: %.2f mm",
            frame_border_mm
        )

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
            "frame_border_mm": frame_border_mm,  # 黑框线条宽度
            "detection_id": str(uuid.uuid4()).replace('-', ''),
            "debug_image_b64": debug_image_b64,
        }

    def _estimate_line_width(
        self,
        binary: np.ndarray,
        frame_contour: np.ndarray,
        corners: list,
        pixels_per_mm: float,
    ) -> float:
        """从图像中动态测量黑框线条宽度

        方法：沿着每条边的外缘，垂直向内扫描，
        第一段连续黑色像素的宽度即为线条宽度。
        在每条边上取多个采样点，取中位数以提高鲁棒性。

        Args:
            binary: 二值图像（BINARY_INV，黑色=255）
            frame_contour: 黑框轮廓
            corners: 黑框四角坐标 [左上, 右上, 右下, 左下]
            pixels_per_mm: 像素密度

        Returns:
            线条宽度 (mm)
        """
        corners_arr = np.array(corners, dtype=np.float64)
        center = np.mean(corners_arr, axis=0)

        # 4条边: top(L→R), right(T→B), bottom(R→L), left(B→T)
        edge_pairs = [(0, 1), (1, 2), (2, 3), (3, 0)]

        widths_px = []
        scan_range_mm = 8  # 向内扫描范围
        scan_steps = max(int(scan_range_mm * pixels_per_mm), 30)
        start_outside = -3  # 从外缘外侧3px开始，确保捕获完整线宽

        for si, ei in edge_pairs:
            p_start = corners_arr[si]
            p_end = corners_arr[ei]
            edge_vec = p_end - p_start
            edge_len = np.linalg.norm(edge_vec)
            if edge_len < 1:
                continue
            edge_dir = edge_vec / edge_len

            # 垂直方向（指向内侧）
            perp = np.array([-edge_dir[1], edge_dir[0]])
            mid = (p_start + p_end) / 2
            if np.dot(perp, center - mid) < 0:
                perp = -perp

            # 沿边取多个采样点（避开角点）
            num_samples = 10
            for i in range(1, num_samples + 1):
                t = i / (num_samples + 1)
                point = p_start + t * edge_vec

                # 垂直向内扫描，找第一段连续黑色像素
                first_black_run = None
                current_run = 0
                in_black = False
                found_first = False

                for step in range(start_outside, scan_steps):
                    pos = point + step * perp
                    px, py = int(round(pos[0])), int(round(pos[1]))

                    if 0 <= px < binary.shape[1] and 0 <= py < binary.shape[0]:
                        is_dark = binary[py, px] > 127
                    else:
                        is_dark = False

                    if is_dark:
                        current_run += 1
                        in_black = True
                    else:
                        if in_black and not found_first:
                            first_black_run = current_run
                            found_first = True
                        current_run = 0
                        in_black = False

                # 扫描结束时仍在黑色区域
                if in_black and not found_first:
                    first_black_run = current_run

                if first_black_run is not None and first_black_run >= 1:
                    widths_px.append(first_black_run)

        if not widths_px:
            _log.warning("黑框线条宽度: 无法从图像中测量，使用默认值 1.5mm")
            return 1.5

        median_px = float(np.median(widths_px))
        mean_px = float(np.mean(widths_px))
        std_px = float(np.std(widths_px))
        line_width_mm = median_px / pixels_per_mm

        # 限制在合理范围 [0.3, 3.0] mm
        line_width_mm = max(0.3, min(line_width_mm, 3.0))

        _log.info(
            "黑框线条宽度: %.2f mm (%.1f px, 中位数), "
            "均值=%.1fpx, std=%.1f, 采样点=%d",
            line_width_mm, median_px, mean_px, std_px, len(widths_px),
        )

        return line_width_mm

    def _find_frame_contour(
        self,
        contours: list,
        img_width: int,
        img_height: int,
        target_aspect: float | None = None,
    ) -> np.ndarray | None:
        """从轮廓列表中找到黑色方框轮廓

        Args:
            contours: 轮廓列表
            img_width: 图片宽度
            img_height: 图片高度
            target_aspect: 目标长宽比 (width/height)，用于辅助筛选

        Returns:
            方框轮廓，如果未找到则返回None
        """
        img_area = img_width * img_height
        min_area = img_area * self.min_area_ratio
        max_area = img_area * self.max_area_ratio

        # 如果有目标长宽比，使用评分系统（借鉴 vision.py 的方法）
        ASPECT_TOLERANCE = 0.50  # 允许最多50%的长宽比误差

        if target_aspect is not None and target_aspect > 0:
            scored = []
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area < 100:
                    continue

                # 使用凸包来计算最小外接矩形
                hull = cv2.convexHull(cnt)
                rect = cv2.minAreaRect(hull)
                rw, rh = rect[1]

                if min(rw, rh) <= 0:
                    continue

                # 计算长宽比
                contour_aspect = max(rw, rh) / min(rw, rh)
                aspect_err = abs(contour_aspect - target_aspect) / target_aspect

                # 面积范围检查
                if area < min_area or area > max_area:
                    continue

                # 长宽比误差评分：误差越小分数越高
                aspect_score = 1.0 / (1.0 + aspect_err * 3.0)
                score = area * aspect_score

                scored.append((score, cnt, hull, aspect_err, contour_aspect))

            if scored:
                # 按分数降序排序
                scored.sort(key=lambda x: x[0], reverse=True)
                best_score, best_cnt, best_hull, aspect_err, best_aspect = scored[0]
                _log.info(
                    "选中轮廓: 分数=%.0f 面积=%.0f 长宽比误差=%.1f%% (共%d个候选)",
                    best_score,
                    cv2.contourArea(best_cnt),
                    aspect_err * 100,
                    len(scored),
                )
                return best_hull

        # 没有目标长宽比时，使用原来的逻辑
        contours = sorted(contours, key=cv2.contourArea, reverse=True)

        for contour in contours:
            area = cv2.contourArea(contour)

            # 检查面积范围
            if area < min_area or area > max_area:
                continue

            # 使用凸包
            hull = cv2.convexHull(contour)

            # 尝试近似为四边形
            epsilon = 0.02 * cv2.arcLength(hull, True)
            approx = cv2.approxPolyDP(hull, epsilon, True)

            # 如果能近似为四边形，就认为有效
            if len(approx) == 4:
                _log.debug("找到四边形轮廓（面积 %.0f px2）", area)
                return hull

        return None

    def _order_corners(self, corners: list) -> list:
        """对四角进行排序：左上、右上、右下、左下

        使用 sum(x+y) 和 diff(x-y) 方法，比角度排序更稳健，
        不会因边框倾斜导致水平/垂直镜像。

        Args:
            corners: 四角坐标列表

        Returns:
            排序后的四角坐标 [TL, TR, BR, BL]
        """
        pts = np.array(corners, dtype=np.float32)

        # sum = x + y: TL 最小, BR 最大
        s = pts[:, 0] + pts[:, 1]
        # diff = x - y: BL 最小, TR 最大
        d = pts[:, 0] - pts[:, 1]

        tl = pts[np.argmin(s)]
        br = pts[np.argmax(s)]
        tr = pts[np.argmax(d)]
        bl = pts[np.argmin(d)]

        return [tuple(map(int, pt)) for pt in [tl, tr, br, bl]]


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