"""CV-based pad position refinement using Canny edges + contour finding.

Given a VLM-identified pad region, uses edge detection to precisely locate
the actual metallic pad boundaries. Falls back to VLM coordinates if no clear
edge contour is found.

Architecture:
    VLM raw coords (approx) + rectified image
        │
        ▼
    _refine_pad_with_cv()  -- Canny edge → largest closed contour
        │
        ▼
    refined polygon (mm) or VLM original (fallback)
"""

from __future__ import annotations

import logging

import cv2
import numpy as np

_log = logging.getLogger(__name__)


def refine_pad_positions(
    rectified_png: bytes,
    vlm_candidates: list[dict],
    width_mm: float,
    height_mm: float,
) -> list[dict]:
    """Refine VLM-detected pad positions using CV edge-based boundary detection.

    For each candidate, crops a region around the VLM-reported position,
    applies Canny edge detection, finds the largest closed contour that
    matches expected pad characteristics, and produces a refined polygon.

    If CV refinement fails for a pad, the original VLM coordinates are kept.

    Returns the candidates list with refined visible_region data.
    """
    img_array = np.frombuffer(rectified_png, np.uint8)
    image = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    if image is None:
        _log.warning("refine_pad_positions: could not decode image, returning VLM original")
        return vlm_candidates

    h_img, w_img = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    refined = []
    for candidate in vlm_candidates:
        label = candidate["label"]
        region = candidate["visible_region"]
        bbox = region["bbox"]

        vlm_cx = region["center"]["x_mm"]
        vlm_cy = region["center"]["y_mm"]
        vlm_x = bbox["x_mm"]
        vlm_y = bbox["y_mm"]
        vlm_w = bbox["width_mm"]
        vlm_h = bbox["height_mm"]

        # Convert to pixel ROI with generous margin
        margin = 40  # px margin around VLM bbox
        roi_x1 = max(0, int(vlm_x / width_mm * w_img) - margin)
        roi_y1 = max(0, int(vlm_y / height_mm * h_img) - margin)
        roi_x2 = min(w_img, int((vlm_x + vlm_w) / width_mm * w_img) + margin)
        roi_y2 = min(h_img, int((vlm_y + vlm_h) / height_mm * h_img) + margin)

        # Ensure minimum ROI size
        if roi_x2 - roi_x1 < 30:
            cx = (roi_x1 + roi_x2) // 2
            roi_x1 = max(0, cx - 40)
            roi_x2 = min(w_img, cx + 40)
        if roi_y2 - roi_y1 < 30:
            cy = (roi_y1 + roi_y2) // 2
            roi_y1 = max(0, cy - 40)
            roi_y2 = min(h_img, cy + 40)

        refined_candidate = _refine_single_pad(
            image, gray, label, vlm_cx, vlm_cy, vlm_x, vlm_y, vlm_w, vlm_h,
            roi_x1, roi_y1, roi_x2, roi_y2,
            w_img, h_img, width_mm, height_mm, candidate,
        )
        refined.append(refined_candidate)

    return refined


def _refine_single_pad(
    image: np.ndarray,
    gray: np.ndarray,
    label: str,
    vlm_cx: float, vlm_cy: float,
    vlm_x: float, vlm_y: float, vlm_w: float, vlm_h: float,
    roi_x1: int, roi_y1: int, roi_x2: int, roi_y2: int,
    w_img: int, h_img: int,
    width_mm: float, height_mm: float,
    candidate: dict,
) -> dict:
    """Refine a single pad's position using CV edge analysis."""

    roi = image[roi_y1:roi_y2, roi_x1:roi_x2]
    roi_gray = gray[roi_y1:roi_y2, roi_x1:roi_x2]
    roi_h, roi_w = roi.shape[:2]

    # ── Strategy: use Canny edge detection to find pad boundaries ──
    # Thick metallic pad edges appear as strong, connected contours
    best_contour = None
    best_score = -1

    # Try multiple Canny thresholds for robustness
    for low_thresh in (20, 30, 40, 50, 60):
        edges = cv2.Canny(roi_gray, low_thresh, low_thresh * 3)

        # Close gaps in edges
        kernel = np.ones((3, 3), np.uint8)
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)

        contours, hierarchy = cv2.findContours(
            edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        for c in contours:
            area = cv2.contourArea(c)

            # Filter: pad should be reasonable size
            expected_vlm_area_px = (vlm_w / width_mm * w_img) * (vlm_h / height_mm * h_img)
            if area < max(30, expected_vlm_area_px * 0.05):
                continue
            if area > roi_w * roi_h * 0.9:  # ignore whole-ROI contours
                continue

            # Check rectangularity
            hull = cv2.convexHull(c)
            hull_area = cv2.contourArea(hull)
            if hull_area < 20:
                continue

            rect = cv2.minAreaRect(c)
            (_, _), (rw, rh), _ = rect
            if min(rw, rh) < 4:
                continue

            aspect = max(rw, rh) / max(min(rw, rh), 1)

            # Pads have aspect ratio within expected range
            vlm_aspect = vlm_w / max(vlm_h, 0.01)
            aspect_ok = abs(aspect - vlm_aspect) < vlm_aspect * 1.5

            # Fill ratio (how well does contour fill its bounding box)
            fill_ratio = area / max(rw * rh, 1) if aspect_ok else 0

            # Score: prefer contours with reasonable area, good fill, and
            # center near VLM center
            roi_center_x = roi_w / 2
            roi_center_y = roi_h / 2
            M = cv2.moments(c)
            if M["m00"] > 0:
                ccx = M["m10"] / M["m00"]
                ccy = M["m01"] / M["m00"]
            else:
                continue

        dist_from_center = np.sqrt((ccx - roi_center_x) ** 2 + (ccy - roi_center_y) ** 2)
        norm_dist = dist_from_center / max(roi_w, roi_h)

        # Size match: prefer contours with area close to VLM estimate
        area_ratio = area / max(expected_vlm_area_px, 1)
        size_match = 1.0 - min(abs(area_ratio - 1.0) / 2.0, 1.0)  # 0=worst, 1=best

        score = (0.35 * size_match                                   # prefer VLM-sized area
                 + 0.25 * fill_ratio                                 # prefer solid shapes
                 + 0.25 * (1.0 - norm_dist)                          # prefer center-positioned
                 + 0.15 * min(area / max(roi_w * roi_h * 0.3, 1), 1.0))  # prefer reasonably large

        if score > best_score:
            best_score = score
            best_contour = c

    # Fallback: try brightness thresholding
    if best_contour is None:
        best_contour = _brightness_fallback(roi, roi_gray, vlm_w, vlm_h, w_img, width_mm, height_mm)

    if best_contour is None:
        _log.debug("CV refinement failed for %s (no valid contour found), keeping VLM coords", label)
        return candidate

    # ── Convert best contour to mm coordinates ──
    hull = cv2.convexHull(best_contour)

    # Smooth polygon approximation
    epsilon = 0.03 * cv2.arcLength(hull, True)
    approx = cv2.approxPolyDP(hull, epsilon, True).reshape(-1, 2)

    if len(approx) < 3:
        # Fall back to bounding rect if polygon approximation fails
        bbox_px = cv2.boundingRect(hull)
        bx, by, bw, bh = bbox_px
        approx = np.array([
            [bx, by], [bx + bw, by], [bx + bw, by + bh], [bx, by + bh]
        ])

    # Map from ROI coords to full image coords → mm
    poly_mm = []
    for px, py in approx:
        full_x = px + roi_x1
        full_y = py + roi_y1
        poly_mm.append({
            "x_mm": round(full_x / w_img * width_mm, 3),
            "y_mm": round(full_y / h_img * height_mm, 3),
        })

    # Compute refined center and bbox
    xs_mm = [p["x_mm"] for p in poly_mm]
    ys_mm = [p["y_mm"] for p in poly_mm]
    new_cx = round(sum(xs_mm) / len(xs_mm), 3)
    new_cy = round(sum(ys_mm) / len(ys_mm), 3)
    new_x = round(min(xs_mm), 3)
    new_y = round(min(ys_mm), 3)
    new_w = round(max(xs_mm) - min(xs_mm), 3)
    new_h = round(max(ys_mm) - min(ys_mm), 3)

    # ── Validation: reject obviously wrong refinements ──
    # 1. Size sanity: CV contour should be within 0.4x-2.5x of VLM estimate
    if vlm_w > 0 and vlm_h > 0:
        size_ratio_w = new_w / vlm_w
        size_ratio_h = new_h / vlm_h
        if size_ratio_w < 0.4 or size_ratio_w > 2.5 or size_ratio_h < 0.4 or size_ratio_h > 2.5:
            _log.debug(
                "CV refinement for %s rejected: size mismatch (CV %.2fx%.2f vs VLM %.2fx%.2f, "
                "ratio_w=%.1f ratio_h=%.1f)",
                label, new_w, new_h, vlm_w, vlm_h, size_ratio_w, size_ratio_h,
            )
            return candidate

    # 2. Meaningful shift: require >2px center shift OR >0.5mm size change
    shift_px = np.sqrt(
        ((new_cx - vlm_cx) / width_mm * w_img) ** 2
        + ((new_cy - vlm_cy) / height_mm * h_img) ** 2
    )

    if shift_px < 2.0 and abs(new_w - vlm_w) < 0.5 and abs(new_h - vlm_h) < 0.5:
        _log.debug("CV refinement for %s: shift only %.1fpx, keeping VLM coords", label, shift_px)
        return candidate

    _log.info(
        "CV refined [%s]: VLM(%.2f,%.2f) %.2fx%.2f → CV(%.2f,%.2f) %.2fx%.2f (shift=%.1fpx)",
        label, vlm_cx, vlm_cy, vlm_w, vlm_h,
        new_cx, new_cy, new_w, new_h, shift_px,
    )

    # ── Build refined candidate ──
    n_vertices = len(poly_mm)
    if n_vertices == 4:
        shape = "rect"
    elif n_vertices <= 8:
        ratio = new_w / max(new_h, 0.01)
        shape = "rounded_rect" if 1.4 < ratio < 3.0 else ("circle" if 0.8 < ratio < 1.25 else "oval")
    else:
        shape = "custom"

    new_region = {
        "type": "solder_pad",
        "visual_class": "metallic",
        "shape": shape,
        "center": {"x_mm": new_cx, "y_mm": new_cy},
        "bbox": {
            "x_mm": new_x, "y_mm": new_y,
            "width_mm": new_w, "height_mm": new_h,
        },
        "polygon": poly_mm,
        "source": "vlm+cv_refine",
    }

    # Update candidate fields
    updated = dict(candidate)
    updated["visible_position"] = {"x_mm": new_cx, "y_mm": new_cy}
    updated["visible_region"] = new_region
    updated["matched_regions"] = [new_region]
    updated["shape"] = shape
    updated["width_mm"] = new_w
    updated["height_mm"] = new_h
    updated["method"] = "vlm+cv_refine"

    return updated


def _brightness_fallback(
    roi: np.ndarray,
    roi_gray: np.ndarray,
    vlm_w: float, vlm_h: float,
    w_img: int,
    width_mm: float, height_mm: float,
) -> np.ndarray | None:
    """CV fallback: use brightness/contrast enhancement to find metallic pads.

    When edge detection fails (e.g., pads blend into dark PCB), try enhancing
    the image and looking for the brightest connected region.
    """
    roi_h, roi_w = roi.shape[:2]
    expected_area = (vlm_w / width_mm * w_img) * (vlm_h / height_mm * (roi_h * w_img / roi_w))
    expected_area = max(30, expected_area)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
    enhanced = clahe.apply(roi_gray)

    # Try multiple threshold levels
    for thresh_percentile in (70, 75, 80, 85, 90):
        thresh_val = np.percentile(enhanced, thresh_percentile)
        _, binary = cv2.threshold(enhanced, thresh_val, 255, cv2.THRESH_BINARY)

        kernel = np.ones((3, 3), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)

        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in contours:
            area = cv2.contourArea(c)
            if area < expected_area * 0.1:
                continue
            if area > roi_w * roi_h * 0.8:
                continue
            return c

    return None
