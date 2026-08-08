"""透视校正

功能：根据黑色方框四角坐标，对图片进行透视变换，校正为正视图。

输入：图片字节 + 方框四角坐标 + 像素密度
输出：校正后的图片字节
"""

from __future__ import annotations

import base64
import logging
import os
import uuid
from pathlib import Path

import cv2
import numpy as np

from ..core.logger import get_logger

_log = get_logger(__name__)


class PerspectiveCalibrator:
    """透视校正器

    将倾斜拍摄的图片校正为正视图，基于黑色方框四角进行透视变换。
    """

    def __init__(self, interpolation: int = cv2.INTER_LANCZOS4):
        """初始化校正器

        Args:
            interpolation: OpenCV插值方法
        """
        self.interpolation = interpolation

    def calibrate(
        self,
        image_bytes: bytes,
        corners: list,
        pixels_per_mm: float,
        frame_width_mm: float,
        frame_height_mm: float,
    ) -> dict:
        """透视校正

        Args:
            image_bytes: 图片字节
            corners: 方框四角坐标 [(x1,y1), (x2,y2), (x3,y3), (x4,y4)]
            pixels_per_mm: 像素密度
            frame_width_mm: 方框宽度 (mm)
            frame_height_mm: 方框高度 (mm)

        Returns:
            {
                "calibration_id": str,        # 校准ID
                "rectified_bytes": bytes,     # 校正后图片字节
                "rectified_width_px": int,    # 校正后宽度
                "rectified_height_px": int,   # 校正后高度
                "pixels_per_mm": float,       # 像素密度（验证）
            }
        """
        _log.info("透视校正: 开始")

        # ── Step 1: 解码图片 ──
        try:
            nparr = np.frombuffer(image_bytes, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            if img is None:
                raise ValueError("Failed to decode image")

            _log.debug("原始图片尺寸: %dx%d", img.shape[1], img.shape[0])

        except Exception as e:
            _log.error("透视校正: 图片解码失败 - %s", e)
            raise ValueError(f"Failed to decode image: {e}")

        # ── Step 2: 计算目标尺寸 ──
        target_width_px = int(frame_width_mm * pixels_per_mm)
        target_height_px = int(frame_height_mm * pixels_per_mm)

        # ── Step 3: 构建透视变换矩阵 ──
        src_pts = np.array(corners, dtype=np.float32)
        dst_pts = np.array([
            [0, 0],
            [target_width_px - 1, 0],
            [target_width_px - 1, target_height_px - 1],
            [0, target_height_px - 1],
        ], dtype=np.float32)

        # 计算透视变换矩阵
        M = cv2.getPerspectiveTransform(src_pts, dst_pts)

        # ── Step 4: 应用透视变换 ──
        rectified = cv2.warpPerspective(
            img,
            M,
            (target_width_px, target_height_px),
            flags=self.interpolation,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(255, 255, 255),  # 白色背景
        )

        _log.info(
            "透视校正: 完成 (校正后尺寸 %dx%d px)",
            rectified.shape[1],
            rectified.shape[0],
        )

        # ── Step 5: 编码输出 ──
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 95]
        success, encoded = cv2.imencode('.jpg', rectified, encode_param)

        if not success:
            raise ValueError("Failed to encode rectified image")

        rectified_bytes = encoded.tobytes()

        # ── Step 6: 生成校准ID ──
        calibration_id = str(uuid.uuid4()).replace('-', '')

        # ── Step 7: 保存到工作目录 ──
        ROOT = Path(__file__).resolve().parents[3]
        WORK_ROOT = Path(os.getenv("BATTERY_DESIGN_WORKDIR", ROOT / "work"))
        cal_dir = WORK_ROOT / "calibrations" / calibration_id
        cal_dir.mkdir(parents=True, exist_ok=True)

        # 保存校正后图片
        (cal_dir / "rectified.png").write_bytes(
            cv2.imencode('.png', rectified)[1].tobytes()
        )

        # 保存元数据
        import json
        metadata = {
            "calibration_id": calibration_id,
            "pixels_per_mm": pixels_per_mm,
            "frame_width_mm": frame_width_mm,
            "frame_height_mm": frame_height_mm,
        }
        (cal_dir / "calibration.json").write_text(
            json.dumps(metadata, indent=2),
            encoding='utf-8',
        )

        return {
            "calibration_id": calibration_id,
            "rectified_bytes": rectified_bytes,
            "rectified_width_px": rectified.shape[1],
            "rectified_height_px": rectified.shape[0],
            "pixels_per_mm": pixels_per_mm,
            "calibration_dir": str(cal_dir),
        }


# ── 测试代码 ──
if __name__ == "__main__":
    # 测试用例（需要配合BlackFrameDetector）
    calibrator = PerspectiveCalibrator()
    print("透视校正器初始化完成")