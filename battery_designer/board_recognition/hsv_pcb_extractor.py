"""PCB提取器（绿色环边界检测）

功能：从校正后的图片中，利用PCB边缘的绿色阻焊层（绿色环）直接检测PCB边界。

核心思路：
  PCB边缘一定是绿色阻焊层，形成连续的绿色环，覆盖率97-99.7%。
  1. HSV色彩空间检测绿色像素
  2. 取最大连通域（排除噪声）
  3. 形态学闭运算填补缝隙，形成完整闭环
  4. 填充外轮廓内部 → PCB mask

输入：校正后图片字节
输出：PCB轮廓多边形 + PCB mask
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np

from ..core.logger import get_logger

_log = get_logger(__name__)


class HSVPCBExtractor:
    """PCB提取器（绿色环边界检测）

    利用PCB边缘的绿色阻焊层直接确定PCB边界。
    """

    def __init__(self):
        """初始化提取器"""
        pass

    # ──────────────────────────────────────────────
    #  公开接口
    # ──────────────────────────────────────────────

    def extract(
        self,
        image_bytes: bytes,
        pixels_per_mm: float | None = None,
        frame_border_mm: float | None = None,
        debug_dir: Path | None = None,
    ) -> dict[str, Any]:
        """提取 PCB 区域（绿色环边界检测）

        Args:
            image_bytes: 校正后图片字节
            pixels_per_mm: 像素密度（保留接口兼容，当前未使用）
            frame_border_mm: 黑框线条宽度（保留接口兼容，当前未使用）
            debug_dir: 调试图片输出目录 (Path 或 None)

        Returns:
            {
                "pcb_mask": np.ndarray,
                "pcb_contour": np.ndarray,
                "pcb_mask_with_shadow": np.ndarray,
                "pcb_contour_with_shadow": np.ndarray,
                "color_name": str,
                "pcb_area_ratio": float,
            }
        """
        _log.info("PCB提取（绿色环边界检测）: 开始")

        # ── Step 1: 解码图片 ──
        nparr = np.frombuffer(image_bytes, np.uint8)
        img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img_bgr is None:
            raise ValueError("Failed to decode image")
        h, w = img_bgr.shape[:2]
        total_area = h * w
        _log.debug("原始尺寸: %dx%d", w, h)

        if debug_dir:
            debug_dir = Path(debug_dir)
            debug_dir.mkdir(parents=True, exist_ok=True)

        # ── Step 2: 绿色环边界检测 ──
        pcb_mask = self._detect_green_ring_boundary(img_bgr, debug_dir)
        if pcb_mask is None:
            raise ValueError("绿色环检测失败：未检测到足够的绿色像素，无法确定PCB边界")

        # ── Step 3: 形态学清理 + 最大连通域 + 轮廓 ──
        k_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        pcb_mask = cv2.morphologyEx(pcb_mask, cv2.MORPH_OPEN, k_open, iterations=1)
        k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        pcb_mask = cv2.morphologyEx(pcb_mask, cv2.MORPH_CLOSE, k_close, iterations=1)

        num_lbl, labels, stats_arr, _ = cv2.connectedComponentsWithStats(pcb_mask, 8)
        if num_lbl <= 1:
            raise ValueError("No PCB region found")
        best = 1 + int(np.argmax(stats_arr[1:, cv2.CC_STAT_AREA]))
        pcb_mask_clean = np.zeros_like(pcb_mask)
        pcb_mask_clean[labels == best] = 255

        contours, _ = cv2.findContours(pcb_mask_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            raise ValueError("No PCB contour found")
        raw_ct = max(contours, key=cv2.contourArea)
        peri = cv2.arcLength(raw_ct, True)
        pcb_contour = cv2.approxPolyDP(raw_ct, 0.002 * peri, True)

        pcb_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.drawContours(pcb_mask, [pcb_contour], -1, 255, -1)

        # ── Step 4: 颜色估计 ──
        color_name = self._estimate_pcb_color(img_bgr, pcb_mask)
        pcb_area_ratio = cv2.contourArea(pcb_contour) / total_area

        _log.info(
            "PCB提取（绿色环边界检测）: 完成 (颜色=%s, 面积占比=%.1f%%)",
            color_name, pcb_area_ratio * 100,
        )

        return {
            "pcb_mask": pcb_mask,
            "pcb_contour": pcb_contour,
            "pcb_mask_with_shadow": pcb_mask,
            "pcb_contour_with_shadow": pcb_contour,
            "color_name": color_name,
            "pcb_area_ratio": pcb_area_ratio,
        }

    # ──────────────────────────────────────────────
    #  私有方法: 绿色环边界检测
    # ──────────────────────────────────────────────

    def _detect_green_ring_boundary(
        self,
        img_bgr: np.ndarray,
        debug_dir: Path | None = None,
    ) -> np.ndarray | None:
        """用绿色环直接检测PCB边界

        核心原理：PCB边缘一定是绿色阻焊层，形成连续的绿色环。
        绿色环覆盖率97-99.7%，经过形态学闭运算后形成完整闭环。

        步骤：
        1. HSV检测绿色像素
        2. 取最大连通域（排除噪声）
        3. 形态学闭运算填补缝隙
        4. 找外轮廓并填充内部 → PCB mask

        Args:
            img_bgr: BGR图像
            debug_dir: 调试输出目录

        Returns:
            PCB mask (uint8, 0/255)，如果不满足绿色条件返回None
        """
        h, w = img_bgr.shape[:2]

        # HSV + Lab 双重过滤
        # HSV: 色相35-90(绿), S>=25, V>=15
        # Lab a通道: a<128表示绿色倾向, 与亮度无关, 暗绿也能检测
        #   (OpenCV Lab中a=128为中性, 绿色a<128, 红/暖色a>128)
        #   比BGR的G-B差值更鲁棒: 暗绿色G-B可能<15但Lab a仍<128
        hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
        h_mask = (hsv[:, :, 0] >= 35) & (hsv[:, :, 0] <= 90)
        s_mask = hsv[:, :, 1] >= 25
        v_mask = hsv[:, :, 2] >= 15
        lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
        lab_green = lab[:, :, 1] < 128
        green_mask = (h_mask & s_mask & v_mask & lab_green).astype(np.uint8) * 255

        green_pct = cv2.countNonZero(green_mask) / (h * w) * 100

        # 绿色覆盖率不足，无法检测
        if green_pct < 5.0:
            _log.warning(
                "绿色环检测: 绿色仅%.1f%%，不满足条件",
                green_pct,
            )
            return None

        # 取最大连通域
        n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(green_mask)
        if n_labels <= 1:
            return None

        max_idx = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        green_main = np.where(labels == max_idx, 255, 0).astype(np.uint8)
        main_pct = cv2.countNonZero(green_main) / (h * w) * 100

        _log.debug(
            "绿色环检测: 总绿色=%.1f%% 最大连通域=%.1f%% (%d个连通域)",
            green_pct, main_pct, n_labels - 1,
        )

        # 形态学闭运算填补缝隙（绿色环中可能有细小断裂）
        k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        green_closed = cv2.morphologyEx(green_main, cv2.MORPH_CLOSE, k_close, iterations=3)

        # 找外轮廓
        contours, _ = cv2.findContours(
            green_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE,
        )
        if not contours:
            return None

        main_contour = max(contours, key=cv2.contourArea)
        contour_area_pct = cv2.contourArea(main_contour) / (h * w) * 100

        # 面积合理性检查（PCB通常占10%-80%）
        if contour_area_pct < 5.0 or contour_area_pct > 80.0:
            _log.warning(
                "绿色环检测: 轮廓面积%.1f%%不合理",
                contour_area_pct,
            )
            return None

        # 填充轮廓内部 → PCB mask
        pcb_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.drawContours(pcb_mask, [main_contour], -1, 255, -1)

        filled_pct = cv2.countNonZero(pcb_mask) / (h * w) * 100
        _log.info(
            "绿色环边界: 绿色=%.1f%% 填充PCB=%.1f%% 轮廓面积=%.1f%%",
            green_pct, filled_pct, contour_area_pct,
        )

        # 调试可视化
        if debug_dir:
            vis = img_bgr.copy()
            # 半透明叠加: 检测到的绿色区域叠加红色高亮, 保留原始颜色
            vis_f = vis.astype(np.float64)
            vis_f[green_mask > 0] = vis_f[green_mask > 0] * 0.4 + np.array([0, 0, 255], dtype=np.float64) * 0.6
            vis = vis_f.astype(np.uint8)
            cv2.imwrite(str(debug_dir / "step4_green_ring.jpg"), vis)
            overlay = img_bgr.copy()
            overlay[pcb_mask > 0] = (
                overlay[pcb_mask > 0] * 0.5
                + np.array([0, 100, 0], dtype=np.float64) * 0.5
            ).astype(np.uint8)
            cv2.imwrite(str(debug_dir / "step4_green_pcb_overlay.jpg"), overlay)

        return pcb_mask

    # ──────────────────────────────────────────────
    #  辅助: 估计 PCB 颜色
    # ──────────────────────────────────────────────

    def _estimate_pcb_color(
        self,
        img_bgr: np.ndarray,
        pcb_mask: np.ndarray,
    ) -> str:
        """根据 PCB 区域像素估计颜色名称"""
        mask_bool = pcb_mask > 127
        if not np.any(mask_bool):
            return "unknown"

        region = img_bgr[mask_bool]
        avg_bgr = region.mean(axis=0)

        # 转 HSV 判断主色调
        avg_bgr_uint8 = np.uint8([[avg_bgr]])
        avg_hsv = cv2.cvtColor(avg_bgr_uint8, cv2.COLOR_BGR2HSV)[0, 0]
        h = avg_hsv[0]
        s = avg_hsv[1]
        v = avg_hsv[2]

        if v < 70:
            return "black"
        elif s < 30:
            return "gray"
        elif 40 <= h <= 90:
            return "green"
        elif 100 <= h <= 130:
            return "blue"
        elif 20 <= h <= 35:
            return "yellow"
        else:
            return "green"  # 默认绿板


# ── 测试代码 ──
if __name__ == "__main__":
    extractor = HSVPCBExtractor()
    print("PCB提取器（绿色环边界检测）初始化完成")
