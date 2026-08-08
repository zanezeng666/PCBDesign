from __future__ import annotations

import logging

import cv2
import numpy as np

logger = logging.getLogger(__name__)


def _cv_find_metallic_pads(transparent_png: bytes, width_mm: float, height_mm: float,
                           pixels_per_mm: float) -> list[dict]:
    """Find metallic solder pads on transparent PCB using CV thresholding.

    Metallic pads are bright (high V) and low saturation (silver/tin/gold).
    Returns list of dicts with center_mm, polygon_mm, area_mm2.
    """
    nparr = np.frombuffer(transparent_png, np.uint8)
    img_rgba = cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)
    if img_rgba is None or len(img_rgba.shape) < 3 or img_rgba.shape[2] != 4:
        return []

    h, w = img_rgba.shape[:2]
    alpha = img_rgba[:, :, 3]
    bgr = img_rgba[:, :, :3]

    # PCB content mask (where alpha > 0)
    pcb_mask = (alpha > 30).astype(np.uint8) * 255

    # Convert to HSV for metallic detection
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)

    # Metallic pads: bright (V > 120) and low saturation (S < 100)
    # This captures silver/tin colored solder pads on the PCB
    metallic_mask = cv2.inRange(hsv, np.array([0, 0, 120]), np.array([180, 100, 255]))

    # Only within PCB area
    metallic_mask = cv2.bitwise_and(metallic_mask, pcb_mask)

    # Morphological cleanup: close gaps within pads, remove tiny noise
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    metallic_mask = cv2.morphologyEx(metallic_mask, cv2.MORPH_CLOSE, kernel, iterations=3)
    metallic_mask = cv2.morphologyEx(metallic_mask, cv2.MORPH_OPEN, kernel, iterations=1)

    # Find contours of metallic regions
    contours, _ = cv2.findContours(metallic_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    pads = []
    min_area_px = (pixels_per_mm * 0.8) ** 2   # min ~0.64mm²
    max_area_px = (pixels_per_mm * 20.0) ** 2   # max 20mm × 20mm

    for cnt in contours:
        area_px = cv2.contourArea(cnt)
        if area_px < min_area_px or area_px > max_area_px:
            continue

        # No aspect ratio filter — battery protection board pads can be elongated strips

        # Compute center in mm
        M = cv2.moments(cnt)
        if M["m00"] == 0:
            continue
        cx_px = M["m10"] / M["m00"]
        cy_px = M["m01"] / M["m00"]
        cx_mm = cx_px / pixels_per_mm
        cy_mm = cy_px / pixels_per_mm

        # Use minAreaRect to get a clean rotated rectangle (pads are rounded rects)
        rect = cv2.minAreaRect(cnt)  # ((cx,cy), (w,h), angle)
        box_pts = cv2.boxPoints(rect)  # 4 corner points
        polygon_mm = [
            {"x_mm": round(float(pt[0]) / pixels_per_mm, 3),
             "y_mm": round(float(pt[1]) / pixels_per_mm, 3)}
            for pt in box_pts
        ]

        # Bounding box (axis-aligned) for display
        bx, by, bw, bh = cv2.boundingRect(cnt)
        pads.append({
            "center_mm": (round(cx_mm, 3), round(cy_mm, 3)),
            "polygon_mm": polygon_mm,
            "area_mm2": round(area_px / (pixels_per_mm ** 2), 2),
            "bbox_mm": {
                "x_mm": round(bx / pixels_per_mm, 3),
                "y_mm": round(by / pixels_per_mm, 3),
                "width_mm": round(bw / pixels_per_mm, 3),
                "height_mm": round(bh / pixels_per_mm, 3),
            },
        })

    # Sort by X position for consistent ordering
    pads.sort(key=lambda p: p["center_mm"][0])
    return pads


def _match_vlm_to_cv(candidates: list[dict], cv_pads: list[dict]):
    """Match VLM candidates to CV-detected metallic pads using optimal assignment.

    VLM tells us HOW MANY pads there are and their approximate positions.
    CV gives precise locations of metallic regions.
    We find the best 1-to-1 assignment: each VLM candidate → one unique CV pad.
    Uses greedy global-optimal: pick the best (closest) pair first, repeat.
    """
    if not candidates or not cv_pads:
        return

    # Build distance matrix: candidates × cv_pads
    n_cand = len(candidates)
    n_pads = len(cv_pads)

    # Compute adaptive match threshold from spatial extent of all points
    all_xs = [c.get("visible_region", {}).get("center", {}).get("x_mm", 0) for c in candidates]
    all_ys = [c.get("visible_region", {}).get("center", {}).get("y_mm", 0) for c in candidates]
    for pad in cv_pads:
        all_xs.append(pad["center_mm"][0])
        all_ys.append(pad["center_mm"][1])
    if all_xs and all_ys:
        spatial_spread = max(max(all_xs) - min(all_xs), max(all_ys) - min(all_ys), 1.0)
        max_match_dist = spatial_spread * 0.25  # 25% of spatial extent
    else:
        max_match_dist = 15.0

    # Compute all (distance, cand_idx, pad_idx) pairs
    pairs = []
    for ci, cand in enumerate(candidates):
        vr = cand.get("visible_region", {})
        vlm_center = vr.get("center", {})
        vlm_x = vlm_center.get("x_mm", 0)
        vlm_y = vlm_center.get("y_mm", 0)
        for pi, pad in enumerate(cv_pads):
            px, py = pad["center_mm"]
            dist = ((px - vlm_x) ** 2 + (py - vlm_y) ** 2) ** 0.5
            pairs.append((dist, ci, pi))

    # Sort by distance (best matches first)
    pairs.sort(key=lambda x: x[0])

    # Greedy assignment: pick best pair, mark both used, repeat
    used_cands = set()
    used_pads = set()
    assignments = {}  # cand_idx → pad_idx

    for dist, ci, pi in pairs:
        if ci in used_cands or pi in used_pads:
            continue
        if dist > max_match_dist:  # too far, not a valid match
            break
        assignments[ci] = pi
        used_cands.add(ci)
        used_pads.add(pi)
        if len(assignments) == n_cand:
            break

    # Apply assignments: replace VLM coords with CV-precise coords
    for ci, pi in assignments.items():
        cand = candidates[ci]
        pad = cv_pads[pi]
        vr = cand.get("visible_region", {})
        vr["center"] = {"x_mm": pad["center_mm"][0], "y_mm": pad["center_mm"][1]}
        vr["polygon"] = pad["polygon_mm"]
        vr["bbox"] = pad["bbox_mm"]
        vr["source"] = "cv_refined"
        if cand.get("visible_position"):
            cand["visible_position"] = {"x_mm": pad["center_mm"][0], "y_mm": pad["center_mm"][1]}
        if cand.get("matched_regions"):
            cand["matched_regions"] = [vr]
