from __future__ import annotations

import base64
import logging

import cv2
import numpy as np

logger = logging.getLogger(__name__)


def _make_overlay_image(rectified_b64: str, outline_mm: list[dict],
                        ppm: float, frame_w_mm: float,
                        frame_h_mm: float) -> str:
    """在透视校正图上叠加 PCB 轮廓，生成校验图 (base64 PNG)。"""
    if not rectified_b64 or not outline_mm or len(outline_mm) < 3:
        return ""
    try:
        img_bytes = base64.b64decode(rectified_b64)
        img = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            return ""
        h, w = img.shape[:2]

        # mm → pixel（校正图坐标系与 frame 尺寸对应）
        outline_px = np.array([
            [round(p["x_mm"] / frame_w_mm * w), round(p["y_mm"] / frame_h_mm * h)]
            for p in outline_mm
        ], dtype=np.int32)

        overlay = img.copy()
        cv2.polylines(overlay, [outline_px], True, (0, 255, 0), 3)
        cv2.fillPoly(overlay, [outline_px], (0, 255, 0))
        blended = cv2.addWeighted(overlay, 0.25, img, 0.75, 0)

        # 在每个顶点画圆点
        for pt in outline_px:
            cv2.circle(blended, tuple(pt), 5, (0, 0, 255), -1)

        _, buf = cv2.imencode(".png", blended)
        return base64.b64encode(buf).decode("ascii")
    except Exception:
        logger.warning("recognize-pcb: overlay generation failed", exc_info=True)
        return ""
