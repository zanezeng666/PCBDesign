from __future__ import annotations

import logging

import numpy as np

from .pipeline import PCBRecognitionPipeline

logger = logging.getLogger(__name__)


def _resample_polygon_perimeter(pts, n):
    """Resample polygon to exactly n evenly-spaced points along its perimeter."""
    if len(pts) < 2:
        return pts
    # Compute cumulative edge lengths
    seg_lens = []
    for i in range(len(pts)):
        j = (i + 1) % len(pts)
        seg_lens.append(np.hypot(pts[j][0] - pts[i][0], pts[j][1] - pts[i][1]))
    total_len = sum(seg_lens)
    if total_len < 1e-9:
        return np.array([pts[0]] * n)

    cum = [0.0]
    for sl in seg_lens[:-1]:  # exclude closing segment
        cum.append(cum[-1] + sl)

    result = []
    for k in range(n):
        t = k / n * total_len
        # Find segment index
        idx = 0
        for i in range(len(cum)):
            if cum[i] <= t:
                idx = i
        seg_t = (t - cum[idx]) / seg_lens[idx] if seg_lens[idx] > 0 else 0
        j = (idx + 1) % len(pts)
        x = pts[idx][0] + seg_t * (pts[j][0] - pts[idx][0])
        y = pts[idx][1] + seg_t * (pts[j][1] - pts[idx][1])
        result.append([x, y])
    return np.array(result, dtype=np.float32)


def _compare_front_back_contours(steps):
    """Compare front and back PCB outlines — they must be identical (same physical board).

    Since back is just the front PCB flipped, we merge both into a refined contour.
    Returns a dict with: ok, message, mismatch_area_pct, merged_outline_mm (optional).
    """
    outlines = {}
    for s in steps:
        if s.get("calibration_success") and s.get("transparent_pcb_outline_mm"):
            outlines[s["side"]] = s["transparent_pcb_outline_mm"]
    if len(outlines) < 2:
        return {"ok": False, "message": "前后两面标定数据不完整，无法对比轮廓", "mismatch_area_pct": 0}

    front_ol = outlines["front"]
    back_ol = outlines["back"]
    if not isinstance(front_ol, list) or len(front_ol) == 0:
        return {"ok": False, "message": "正面轮廓数据为空", "mismatch_area_pct": 0}
    if not isinstance(back_ol, list) or len(back_ol) == 0:
        return {"ok": False, "message": "背面轮廓数据为空", "mismatch_area_pct": 0}

    # Extract per-side PPM calibration for inter-camera scale normalization.
    # Different cameras can yield slightly different px/mm values (e.g.
    # front=47.6, back=48.4 → 1.6% systematic error).  Passing both PPM
    # values lets the merge function unify the coordinate systems.
    front_ppm = 0.0
    back_ppm = 0.0
    for s in steps:
        if s.get("side") == "front":
            front_ppm = s.get("pixels_per_mm", 0.0)
        elif s.get("side") == "back":
            back_ppm = s.get("pixels_per_mm", 0.0)

    AREA_THRESHOLD_PCT = 10.0  # 10% tolerance

    # ── Merge front + back into refined consensus outline ──
    w_mm = steps[0].get("frame_w_mm", 40.0) if steps else 40.0
    h_mm = steps[0].get("frame_h_mm", 25.0) if steps else 25.0

    front_result = {
        "outline": front_ol,
        "pixels_per_mm": front_ppm,
        "width_mm": w_mm,
        "height_mm": h_mm,
    }
    back_result = {
        "outline": back_ol,
        "pixels_per_mm": back_ppm,
        "width_mm": w_mm,
        "height_mm": h_mm,
    }
    cross_result = PCBRecognitionPipeline.cross_validate_front_back(
        front_result, back_result
    )
    merged_outline = cross_result["outline"]
    f_area = cross_result["front_area_mm2"]
    b_area = cross_result["back_area_mm2"]
    merged_area = cross_result["consensus_area_mm2"]

    if f_area <= 0 or b_area <= 0:
        return {"ok": False, "message": "无法计算PCB面积", "mismatch_area_pct": 0}

    # ── Consensus-based area validation ──
    # All three areas (f_area, b_area, merged_area) come from the same
    # mask-drawing pipeline at the same SCALE.  This eliminates the bias
    # introduced by comparing raw contour areas against mask-derived
    # consensus areas.
    consensus_dev_front = (abs(f_area - merged_area) / merged_area * 100.0
                           if merged_area > 0 else 999)
    consensus_dev_back = (abs(b_area - merged_area) / merged_area * 100.0
                          if merged_area > 0 else 999)
    consensus_dev = max(consensus_dev_front, consensus_dev_back)

    logger.info(
        "Front/back merge: front=%dpts(mask=%.0fmm2) back=%dpts(mask=%.0fmm2) "
        "-> merged=%dpts(%.0fmm2), consensus_dev=%.1f%% (f=%.1f%% b=%.1f%%)",
        len(front_ol), f_area, len(back_ol), b_area,
        len(merged_outline), merged_area,
        consensus_dev, consensus_dev_front, consensus_dev_back)

    result = {
        "mismatch_area_pct": round(consensus_dev, 1),
        "merged_outline_mm": merged_outline,
        "merged_area_mm2": round(merged_area, 1),
    }

    if consensus_dev > AREA_THRESHOLD_PCT:
        result["ok"] = False
        result["message"] = (f"正反面轮廓与共识偏差 {consensus_dev:.1f}%，"
                             "已合并生成参考轮廓")
    else:
        result["ok"] = True
        result["message"] = (f"正反面轮廓与共识匹配通过（偏差 {consensus_dev:.1f}%），"
                             "已合并为精确轮廓")

    return result
