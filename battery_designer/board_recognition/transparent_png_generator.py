"""透明PNG生成

功能：基于PCB轮廓生成透明背景的PNG图片。

输入：校正后图片 + PCB轮廓
输出：透明PNG图片（RGBA）
"""

from __future__ import annotations

import base64
import logging
from pathlib import Path

import cv2
import numpy as np

from ..core.logger import get_logger

_log = get_logger(__name__)


class TransparentPNGGenerator:
    """透明PNG生成器

    生成背景透明的PCB图片，用于后续的焊盘识别等流程。
    """

    def __init__(self, background_color: tuple = (0, 0, 0, 0)):
        """初始化生成器

        Args:
            background_color: 背景颜色 (R, G, B, A)，默认透明
        """
        self.background_color = background_color

    def generate(
        self,
        image_bytes: bytes,
        pcb_contour: np.ndarray,
        calibration_id: str,
    ) -> dict:
        """生成透明PNG

        Args:
            image_bytes: 校正后图片字节
            pcb_contour: PCB轮廓
            calibration_id: 校准ID

        Returns:
            {
                "transparent_bytes": bytes,  # 透明PNG字节
                "transparent_b64": str,      # 透明PNG (base64)
                "bbox": (x, y, w, h),        # PCB边界框
            }
        """
        _log.info("透明PNG生成: 开始")

        # ── Step 1: 解码图片 ──
        try:
            nparr = np.frombuffer(image_bytes, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            if img is None:
                raise ValueError("Failed to decode image")

            h, w = img.shape[:2]
            _log.debug("图片尺寸: %dx%d", w, h)

        except Exception as e:
            _log.error("透明PNG生成: 图片解码失败 - %s", e)
            raise ValueError(f"Failed to decode image: {e}")

        # ── Step 2: 创建掩码 ──
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.drawContours(mask, [pcb_contour], -1, 255, -1)

        # ── Step 3: 转换为RGBA ──
        img_rgba = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)

        # 设置透明通道
        img_rgba[:, :, 3] = mask

        # ── Step 4: 计算边界框 ──
        x, y, w_box, h_box = cv2.boundingRect(pcb_contour)
        bbox = (int(x), int(y), int(w_box), int(h_box))

        # ── Step 5: 编码为PNG ──
        success, encoded = cv2.imencode('.png', img_rgba)

        if not success:
            raise ValueError("Failed to encode transparent PNG")

        transparent_bytes = encoded.tobytes()
        transparent_b64 = base64.b64encode(transparent_bytes).decode('utf-8')

        # ── Step 6: 保存到文件 ──
        import os
        ROOT = Path(__file__).resolve().parents[3]
        WORK_ROOT = Path(os.getenv("BATTERY_DESIGN_WORKDIR", ROOT / "work"))
        transparent_path = WORK_ROOT / "calibrations" / calibration_id / "transparent.png"

        transparent_path.parent.mkdir(parents=True, exist_ok=True)
        transparent_path.write_bytes(transparent_bytes)

        _log.info(
            "透明PNG生成: 完成 (PCB边界框 %dx%d px)",
            w_box,
            h_box,
        )

        return {
            "transparent_bytes": transparent_bytes,
            "transparent_b64": transparent_b64,
            "bbox": bbox,
            "save_path": str(transparent_path),
        }


# ── 测试代码 ──
if __name__ == "__main__":
    generator = TransparentPNGGenerator()
    print("透明PNG生成器初始化完成")