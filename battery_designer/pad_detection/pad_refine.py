"""Pad position refinement using CV edge detection and geometric alignment.

Extracted from app.py to isolate the pad geometry logic from the API layer.
"""
from __future__ import annotations

import logging
import math
import statistics

import cv2
import numpy as np

logger = logging.getLogger(__name__)


def _refine_positions_cv(result: dict, img_rgba, pixels_per_mm: float) -> dict:
    """Refine VLM pad positions using CV edge detection on the original image.

    VLM is good at identifying WHICH pads exist and their labels, but its
    coordinate precision is limited (~0.002 fractional = ~0.1mm on a 41mm crop).
    This function uses image processing to find the actual metallic boundary
    near each VLM-indicated position and shifts the polygon to match.

    Only position is adjusted — VLM's count, labels, and shape are preserved.
    """
    h_img, w_img = img_rgba.shape[:2]
    has_alpha = len(img_rgba.shape) == 3 and img_rgba.shape[2] == 4

    invalid_indices: set[int] = set()  # candidates to reject after loop

    for idx, cand in enumerate(result.get("candidates", [])):
        regions = cand.get("matched_regions", [])
        if not regions:
            continue
        region = regions[0]
        poly = region.get("polygon", [])
        if len(poly) < 3:
            continue

        # Polygon bbox in pixel coords
        xs = [p["x_mm"] * pixels_per_mm for p in poly]
        ys = [p["y_mm"] * pixels_per_mm for p in poly]
        pad_w = max(xs) - min(xs)
        pad_h = max(ys) - min(ys)
        # ── Label-aware ROI padding ──
        # P+/P-/B+/B-: VLM often traces silkscreen text 1.0–1.5 mm away from
        # the actual metal pad → wide search (80% + 2.0mm).
        # ID/TH/T/NTC/N: moderate search (50% + 1.0mm) — VLM sometimes
        # misplaces them on nearby silkscreen text but the real metal pad
        # is within ~2mm.
        # Others: tight search (30% + 0.5mm).
        label = cand.get("label", "")
        label_up = label.upper()
        if label_up in ("P+", "P-", "B+", "B-"):
            pad_px = int(max(pad_w, pad_h) * 0.80) + int(2.0 * pixels_per_mm)
        elif label_up in ("ID", "TH", "T", "NTC", "N"):
            pad_px = int(max(pad_w, pad_h) * 0.50) + int(1.0 * pixels_per_mm)
        else:
            pad_px = int(max(pad_w, pad_h) * 0.30) + int(0.5 * pixels_per_mm)

        rx1 = max(0, int(min(xs)) - pad_px)
        ry1 = max(0, int(min(ys)) - pad_px)
        rx2 = min(w_img, int(max(xs)) + pad_px + 1)
        ry2 = min(h_img, int(max(ys)) + pad_px + 1)
        if rx2 - rx1 < 10 or ry2 - ry1 < 10:
            continue

        roi = img_rgba[ry1:ry2, rx1:rx2].copy()

        # PCB region mask (alpha >= 128 = solid board).  Non-PCB pixels are
        # excluded from metallic detection so grooves/notches don't pollute it.
        if has_alpha:
            a = roi[:, :, 3]
            pcb_region = (a >= 128).astype(np.uint8) * 255
        else:
            pcb_region = np.full(roi.shape[:2], 255, dtype=np.uint8)

        # Metallic pads are LOW-SATURATION (silver/tin/gold) whereas the green
        # solder mask is highly saturated.  A plain Otsu threshold on grayscale
        # fails here: the dark groove/notch pixels drag the Otsu threshold down
        # so the whole green board gets classified as "bright", the blob becomes
        # the entire PCB and its center collapses to the ROI center (≈ the VLM
        # estimate) — i.e. no refinement at all.  Saturation separates metal
        # (S≈28) from solder mask (S≈224) cleanly; the loose V floor only
        # rejects truly-black parts (component bodies, V<40).
        # Use tighter S threshold (80 vs 100) to reduce background bleed at
        # board edges where white paper has similarly low saturation.
        # Also cap V at 230 to exclude pure-white paper/polygon (V≈240-255,
        # S≈0-10), which otherwise passes the low-saturation check.
        hsv = cv2.cvtColor(roi[:, :, :3], cv2.COLOR_BGR2HSV)
        # V cap raised to 245 (was 230): some metallic pads exhibit V≈225-240
        # under certain lighting, especially tin/solder surfaces.  White paper
        # (V≈245-255) is still safely excluded.
        metallic = cv2.inRange(hsv, np.array([0, 0, 40]), np.array([180, 80, 245]))
        binary = cv2.bitwise_and(metallic, pcb_region)

        # Clean up noise
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)
        # Fill holes (text on pad creates dark holes that shift centroid)
        flood = binary.copy()
        h_f, w_f = flood.shape
        mask_ff = np.zeros((h_f + 2, w_f + 2), np.uint8)
        cv2.floodFill(flood, mask_ff, (0, 0), 255)
        holes = cv2.bitwise_not(flood)
        binary = binary | holes

        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            # ── Wider fallback search ──
            # VLM may have placed the polygon on silkscreen text adjacent to
            # the actual metal pad.  Expand the ROI by 1.5× and retry once.
            fallback_px = int(pad_px * 1.5)
            frx1 = max(0, int(min(xs)) - fallback_px)
            fry1 = max(0, int(min(ys)) - fallback_px)
            frx2 = min(w_img, int(max(xs)) + fallback_px + 1)
            fry2 = min(h_img, int(max(ys)) + fallback_px + 1)
            if frx2 - frx1 > 20 and fry2 - fry1 > 20:
                froi = img_rgba[fry1:fry2, frx1:frx2].copy()
                if has_alpha:
                    fa = froi[:, :, 3]
                    fpcb = (fa >= 128).astype(np.uint8) * 255
                else:
                    fpcb = np.full(froi.shape[:2], 255, dtype=np.uint8)
                fhsv = cv2.cvtColor(froi[:, :, :3], cv2.COLOR_BGR2HSV)
                fmetallic = cv2.inRange(fhsv, np.array([0, 0, 40]), np.array([180, 80, 245]))
                fbinary = cv2.bitwise_and(fmetallic, fpcb)
                fkernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
                fbinary = cv2.morphologyEx(fbinary, cv2.MORPH_OPEN, fkernel, iterations=1)
                fbinary = cv2.morphologyEx(fbinary, cv2.MORPH_CLOSE, fkernel, iterations=2)
                fcontours, _ = cv2.findContours(fbinary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if fcontours:
                    # Found metal in wider search — reuse the expanded ROI context
                    binary, contours = fbinary, fcontours
                    rx1, ry1, rx2, ry2 = frx1, fry1, frx2, fry2
                    roi_area = (rx2 - rx1) * (ry2 - ry1)
                    logger.info("CV refine: %s metal found in wider fallback (+%.0f%%)",
                                cand.get('label', '?'), (fallback_px / max(pad_px, 1) - 1) * 100)
            if not contours:
                logger.info("CV refine: %s NO metallic contour found in ROI — keeping VLM position as-is",
                            cand.get('label', '?'))
                continue

        # Find the largest contour (should be the metallic pad)
        best = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(best)
        poly_area = pad_w * pad_h
        roi_area = (rx2 - rx1) * (ry2 - ry1)

        # ── VLM hallucination / silkscreen pseudo-pad rejection ──
        # If the CV-detected metal contour is far smaller than the VLM-claimed
        # polygon, the region is NOT a real terminal pad.  This catches:
        #  • VLM detecting a silkscreen label outline (rectangular, pad-like)
        #    as a "pad" when only a tiny via dot sits inside
        #  • Pure hallucinations where no metal exists at all
        #
        # Geometric-inferred pads are exempt: they fill a known pattern gap and
        # may land outside the board where no metal is visible.
        is_geometric = any("geometric" in (r.get("source", "") or "")
                           for r in (cand.get("matched_regions") or []))
        area_ratio = area / max(poly_area, 0.01)
        label = cand.get("label", "")

        # ID/TH/T/NTC/N are real auxiliary pads — never reject them.
        # VLM often places their polygon on silkscreen text instead of metal,
        # so CV may find zero metal in the ROI.  Keep the VLM position as-is.
        _AUX_LABELS = {"ID", "TH", "T", "NTC", "N"}

        if not is_geometric and label.upper() not in _AUX_LABELS:
            # Two-tier rejection (only for main power pads):
            # Tier 1: <5%  → pure hallucination (no metal at all)
            # Tier 2: <15% AND polygon > 4 mm² → silkscreen pseudo-pad
            if area_ratio < 0.05:
                logger.warning("CV refine: %s area %.0f < 5%% of poly %.0f — rejecting VLM hallucination",
                               label, area, poly_area)
                invalid_indices.add(idx)
                continue
            if area_ratio < 0.15 and poly_area > 4.0:
                logger.warning(
                    "CV refine: %s metal/poly=%.1f%% (poly=%.1f mm²) — "
                    "silkscreen pseudo-pad (large poly but little metal), rejecting",
                    label, area_ratio * 100, poly_area,
                )
                invalid_indices.add(idx)
                continue

        # Safety: contour must be 25%–400% of VLM polygon area
        if area < poly_area * 0.25 or area > poly_area * 4.0:
            logger.info("CV refine: %s contour area %.0f outside range [%.0f, %.0f], skip",
                        cand.get('label','?'), area, poly_area*0.25, poly_area*4.0)
            continue

        # Safety: if metallic mask fills >80% of ROI, background is bleeding
        # into the detection (happens at board edges or near large metal planes),
        # so the contour center is unreliable — skip.
        metal_pct = float(cv2.countNonZero(binary)) / float(roi_area) if roi_area > 0 else 1.0
        if metal_pct > 0.80:
            logger.info("CV refine: %s metal_mask=%.0f%% of ROI (background bleed), skip",
                        cand.get('label','?'), metal_pct * 100)
            continue
        # If <10% of expected pad area is metallic, the pad is barely
        # visible in this ROI — skip CV refinement but keep the candidate.
        if area < poly_area * 0.10:
            logger.info("CV refine: %s contour area %.0f < 10%% of poly_area %.0f, skip",
                        cand.get('label','?'), area, poly_area)
            continue

        # Use minAreaRect for geometric center (more robust than centroid
        # because it's not affected by asymmetric brightness within the pad)
        rect = cv2.minAreaRect(best)
        cx_px = rect[0][0] + rx1
        cy_px = rect[0][1] + ry1

        # Estimate corner radius from contour vs minAreaRect corners
        # For a rounded rect, contour points near each box corner are inset
        # by r*(sqrt(2)-1) from the sharp corner.
        import math as _math
        box_pts = cv2.boxPoints(rect)  # 4 corners of minAreaRect (in ROI coords)
        contour_pts = best.reshape(-1, 2).astype(float)
        cr_radii = []
        for bp in box_pts:
            dists = np.sqrt((contour_pts[:, 0] - bp[0])**2 +
                            (contour_pts[:, 1] - bp[1])**2)
            min_d = float(dists.min())
            if min_d > 0.5:  # at least 0.5px inset (not a sharp corner)
                cr_radii.append(min_d / (_math.sqrt(2) - 1))
        if len(cr_radii) >= 3:
            cr_sorted = sorted(cr_radii)
            cr_med = cr_sorted[len(cr_sorted) // 2]
            cr_good = [r for r in cr_radii if 0.5 * cr_med <= r <= 1.5 * cr_med]
            cr_px = sum(cr_good) / len(cr_good) if cr_good else cr_med
            cand["_cv_corner_radius_px"] = cr_px

        # VLM polygon center
        cx_vlm = sum(xs) / len(xs)
        cy_vlm = sum(ys) / len(ys)

        # Shift = actual center - VLM center
        dx = cx_px - cx_vlm
        dy = cy_px - cy_vlm
        shift_px = (dx ** 2 + dy ** 2) ** 0.5

        # Safety: max shift is a fraction of pad diagonal.
        # CV should refine, not relocate — but when the CV finds a large,
        # high-confidence metal contour (area_ratio ≥ 0.30) we can trust it
        # more and allow a larger shift.  This is critical for P+/P- pads
        # where VLM often places the polygon on silkscreen text 1.0–1.5mm
        # away from the actual metallic pad.
        img_max_dim = max(h_img, w_img)
        base_factor = 0.60 if area_ratio >= 0.30 else 0.30
        max_shift = min((pad_w ** 2 + pad_h ** 2) ** 0.5 * base_factor,
                        img_max_dim * 0.05)
        logger.info("CV refine: %s VLM=(%.1f,%.1f) CV=(%.1f,%.1f) shift=%.1fpx max=%.1fpx",
                    cand.get('label','?'), cx_vlm, cy_vlm, cx_px, cy_px, shift_px, max_shift)
        if shift_px > max_shift:
            continue

        # Apply shift (position only, preserve VLM shape)
        dx_mm = round(dx / pixels_per_mm, 3)
        dy_mm = round(dy / pixels_per_mm, 3)
        shift_mm = round((dx_mm ** 2 + dy_mm ** 2) ** 0.5, 3)
        cand["_cv_shift_mm"] = shift_mm  # store BEFORE polygon shift for alignment
        for pt in poly:
            pt["x_mm"] = round(pt["x_mm"] + dx_mm, 3)
            pt["y_mm"] = round(pt["y_mm"] + dy_mm, 3)
        center = region.get("center", {})
        if center:
            center["x_mm"] = round(center.get("x_mm", 0) + dx_mm, 3)
            center["y_mm"] = round(center.get("y_mm", 0) + dy_mm, 3)
        bbox = region.get("bbox", {})
        if bbox:
            bbox["x_mm"] = round(bbox.get("x_mm", 0) + dx_mm, 3)
            bbox["y_mm"] = round(bbox.get("y_mm", 0) + dy_mm, 3)
        vp = cand.get("visible_position")
        if vp:
            vp["x_mm"] = round(vp.get("x_mm", 0) + dx_mm, 3)
            vp["y_mm"] = round(vp.get("y_mm", 0) + dy_mm, 3)

        logger.info("CV refine: %s shifted (%.2f, %.2f)mm [%.1fpx]",
                    cand.get("label", "?"), dx_mm, dy_mm, shift_px)

        # −− Store CV-detected tight metal bbox (world mm) for downstream crop−windows −−
        # VLM polygon may be over−sized; the CV contour is the true metal boundary.
        # minAreaRect gives an axis−aligned bounding rectangle of the actual metal.
        cv_w_mm = rect[1][0] / pixels_per_mm
        cv_h_mm = rect[1][1] / pixels_per_mm
        cand["_cv_metal_bbox"] = {
            "x_mm": round(cx_px / pixels_per_mm, 3),
            "y_mm": round(cy_px / pixels_per_mm, 3),
            "w_mm": round(cv_w_mm, 3),
            "h_mm": round(cv_h_mm, 3),
        }

    # ── Remove hallucinated candidates ──
    if invalid_indices:
        original = result.get("candidates", [])
        kept = [c for i, c in enumerate(original) if i not in invalid_indices]
        rejected = [original[i]["label"] for i in sorted(invalid_indices) if i < len(original)]
        logger.warning("CV refine: rejecting %d hallucinated pad(s): %s",
                       len(invalid_indices), rejected)
        result["candidates"] = kept

    return result


def _clamp_pads_to_board(result, pcb_w_mm: float, pcb_h_mm: float):
    """Clamp every candidate's polygon vertices to stay within PCB bounds.

    Geometrically-inferred pads (or pads adjusted by `_align_pad_groups`)
    may land outside the board.  This function truncates the polygon so no
    vertex extends beyond [0, pcb_w]×[0, pcb_h], then recomputes the
    effective centre.

    Pads whose polygon collapses to a degenerate shape (zero area) after
    clamping are removed entirely — they were entirely outside the PCB.
    """
    if pcb_w_mm <= 0 or pcb_h_mm <= 0:
        return

    candidates = result.get("candidates", [])
    rejected: list[int] = []

    for ci, cand in enumerate(candidates):
        regions = cand.get("matched_regions", [])
        region_indices_to_remove: list[int] = []
        for ri, region in enumerate(regions):
            poly = region.get("polygon") or []
            if len(poly) < 3:
                continue
            clamped_any = False
            for v in poly:
                ox = v.get("x_mm")
                oy = v.get("y_mm")
                if ox is not None:
                    cx = max(0.0, min(pcb_w_mm, float(ox)))
                    if abs(cx - float(ox)) > 1e-4:
                        clamped_any = True
                    v["x_mm"] = round(cx, 3)
                if oy is not None:
                    cy = max(0.0, min(pcb_h_mm, float(oy)))
                    if abs(cy - float(oy)) > 1e-4:
                        clamped_any = True
                    v["y_mm"] = round(cy, 3)
            if not clamped_any:
                continue

            # Recompute effective centre from clamped polygon
            xs = [v["x_mm"] for v in poly]
            ys = [v["y_mm"] for v in poly]
            cx = round(sum(xs) / len(xs), 3)
            cy = round(sum(ys) / len(ys), 3)

            # Check for degenerate polygon: zero area after clamping means
            # the entire pad was outside the PCB and got squashed to a line/point.
            w_clamped = round(max(xs) - min(xs), 3)
            h_clamped = round(max(ys) - min(ys), 3)
            if w_clamped < 0.05 or h_clamped < 0.05:
                region_indices_to_remove.append(ri)
                logger.warning(
                    "PCB clamp: %s DEGENERATE polygon (%.3f×%.3fmm) at (%.2f,%.2f) — "
                    "pad was entirely outside PCB, discarding",
                    cand.get("label", "?"), w_clamped, h_clamped, cx, cy,
                )
                continue

            region["center"] = {"x_mm": cx, "y_mm": cy}
            bbox = region.get("bbox", {})
            if bbox:
                bbox["x_mm"] = round(min(xs), 3)
                bbox["y_mm"] = round(min(ys), 3)
                bbox["width_mm"] = w_clamped
                bbox["height_mm"] = h_clamped
            vp = cand.get("visible_position", {})
            if vp:
                vp["x_mm"] = cx
                vp["y_mm"] = cy
            logger.info("PCB clamp: %s polygon truncated to board edge (clamped %.3f×%.3fmm)",
                        cand.get("label", "?"), w_clamped, h_clamped)

        # Remove degenerate regions
        if region_indices_to_remove:
            for ri in reversed(region_indices_to_remove):
                del regions[ri]
            cand["matched_regions"] = regions

        # If all regions were removed, mark candidate for rejection
        if not cand.get("matched_regions"):
            rejected.append(ci)

    # Remove candidates that lost all their regions
    if rejected:
        for ci in reversed(rejected):
            label = candidates[ci].get("label", "?")
            logger.warning("PCB clamp: removing %s — all polygon regions collapsed outside PCB",
                          label)
            del candidates[ci]
        result["candidates"] = candidates


def _align_pad_groups(result, pixels_per_mm=1.0, pcb_w_mm=0.0, pcb_h_mm=0.0):
    """Enforce alignment, uniform size, and even spacing among pad groups."""
    import statistics

    candidates = result.get("candidates", [])
    if len(candidates) < 2:
        return result

    pads: list[dict] = []
    for cand in candidates:
        regions = cand.get("matched_regions", [])
        region = regions[0] if regions else cand.get("visible_region", {})
        poly = region.get("polygon") or []
        center = region.get("center", {})
        if len(poly) < 3 or not center:
            cx = center.get("x_mm")
            cy = center.get("y_mm")
            cand_w = cand.get("width_mm") or cand.get("visible_region", {}).get("bbox", {}).get("width_mm")
            cand_h = cand.get("height_mm") or cand.get("visible_region", {}).get("bbox", {}).get("height_mm")
            if cx is not None and cy is not None and cand_w and cand_h:
                hw, hh = cand_w / 2, cand_h / 2
                poly = [
                    {"x_mm": cx - hw, "y_mm": cy - hh},
                    {"x_mm": cx + hw, "y_mm": cy - hh},
                    {"x_mm": cx + hw, "y_mm": cy + hh},
                    {"x_mm": cx - hw, "y_mm": cy + hh},
                ]
            else:
                continue
        xs = [p["x_mm"] for p in poly]
        ys = [p["y_mm"] for p in poly]
        w = max(xs) - min(xs)
        h = max(ys) - min(ys)
        pads.append({
            "cand": cand, "region": region, "poly": poly,
            "cx": center.get("x_mm", 0), "cy": center.get("y_mm", 0),
            "w": w, "h": h,
        })

    if len(pads) < 2:
        return result

    typical = statistics.median([p["w"] for p in pads] + [p["h"] for p in pads])
    if typical < 0.01:
        return result
    tol = typical * 0.6

    used: set[int] = set()
    groups: list[tuple[str, list[int]]] = []

    for i in range(len(pads)):
        if i in used:
            continue
        v_grp = [j for j in range(len(pads))
                 if j not in used and abs(pads[j]["cx"] - pads[i]["cx"]) < tol]
        if len(v_grp) >= 2:
            groups.append(("v", v_grp))
            used.update(v_grp)
            continue
        h_grp = [j for j in range(len(pads))
                 if j not in used and abs(pads[j]["cy"] - pads[i]["cy"]) < tol]
        if len(h_grp) >= 2:
            groups.append(("h", h_grp))
            used.update(h_grp)

    for axis, indices in groups:
        grp = [pads[i] for i in indices]
        n = len(grp)

        med_w = statistics.median([p["w"] for p in grp])
        med_h = statistics.median([p["h"] for p in grp])

        if axis == "v":
            med_align = statistics.median([p["cx"] for p in grp])
        else:
            med_align = statistics.median([p["cy"] for p in grp])

        target_positions = None
        if n >= 3:
            if axis == "v":
                ordered = sorted(grp, key=lambda p: p["cy"])
                coords = [p["cy"] for p in ordered]
            else:
                ordered = sorted(grp, key=lambda p: p["cx"])
                coords = [p["cx"] for p in ordered]

            gaps = [coords[i+1] - coords[i] for i in range(n - 1)]
            if len(gaps) == 0:
                continue
            med_gap = statistics.median(gaps)

            pad_size_along_free = med_h if axis == "v" else med_w
            if med_gap < pad_size_along_free:
                max_gap = max(gaps)
                if max_gap >= pad_size_along_free:
                    med_gap = max_gap
                else:
                    med_gap = pad_size_along_free
                logger.info(
                    "Align(%s): median gap %.3f < pad size %.3f → bumped to %.3f",
                    axis, statistics.median(gaps), pad_size_along_free, med_gap,
                )

            clusters = [ordered]
            if n >= 4:
                mid = (coords[0] + coords[-1]) / 2
                top_half = [p for p in ordered
                            if (p["cy"] if axis == "v" else p["cx"]) < mid]
                bot_half = [p for p in ordered
                            if (p["cy"] if axis == "v" else p["cx"]) >= mid]
                if len(top_half) >= 2 and len(bot_half) >= 2:
                    clusters = [top_half, bot_half]

            target_positions = {}
            for cluster in clusters:
                cn = len(cluster)
                if cn >= 2:
                    if axis == "v":
                        cl_ordered = sorted(cluster, key=lambda p: p["cy"])
                        cl_coords = [p["cy"] for p in cl_ordered]
                    else:
                        cl_ordered = sorted(cluster, key=lambda p: p["cx"])
                        cl_coords = [p["cx"] for p in cl_ordered]
                    if cn >= 3:
                        cl_gaps = [cl_coords[i+1] - cl_coords[i] for i in range(cn - 1)]
                        cl_med_gap = statistics.median(cl_gaps)

                        cl_pad_size = med_h if axis == "v" else med_w
                        cl_min_gap = cl_pad_size + 0.15
                        if cl_med_gap < cl_min_gap:
                            cl_max_gap = max(cl_gaps)
                            span = cl_coords[-1] - cl_coords[0]
                            max_fit = span / (cn - 1)
                            if cl_max_gap >= cl_min_gap and cl_max_gap <= max_fit:
                                cl_med_gap = cl_max_gap
                            elif max_fit >= cl_pad_size:
                                cl_med_gap = max_fit
                            else:
                                logger.warning(
                                    "Align cluster(%s): can't fit %d pads in span %.2f "
                                    "(pad=%.2f) — keeping original positions",
                                    axis, cn, span, cl_pad_size,
                                )
                                cl_med_gap = 0

                        if cl_med_gap > 0:
                            start = cl_coords[0]
                            for idx_p, p in enumerate(cl_ordered):
                                target_positions[id(p)] = start + idx_p * cl_med_gap
                    elif cn == 2:
                        actual_mid = (cl_coords[0] + cl_coords[1]) / 2
                        half_gap = abs(cl_coords[0] - cl_coords[1]) / 2
                        first = cl_ordered[0]
                        second = cl_ordered[1]
                        target_positions[id(first)] = actual_mid - half_gap
                        target_positions[id(second)] = actual_mid + half_gap

        if n == 2:
            if axis == "h":
                actual_mid = (grp[0]["cx"] + grp[1]["cx"]) / 2
                half_gap = abs(grp[0]["cx"] - grp[1]["cx"]) / 2
                left = grp[0] if grp[0]["cx"] < grp[1]["cx"] else grp[1]
                right = grp[1] if grp[0]["cx"] < grp[1]["cx"] else grp[0]
                target_positions = {
                    id(left): actual_mid - half_gap,
                    id(right): actual_mid + half_gap,
                }
                logger.info("Align(h): 2-pad symmetric about actual mid X=%.3f, gap=%.3f",
                            actual_mid, half_gap * 2)
            elif axis == "v":
                actual_mid = (grp[0]["cy"] + grp[1]["cy"]) / 2
                half_gap = abs(grp[0]["cy"] - grp[1]["cy"]) / 2
                top = grp[0] if grp[0]["cy"] < grp[1]["cy"] else grp[1]
                bottom = grp[1] if grp[0]["cy"] < grp[1]["cy"] else grp[0]
                target_positions = {
                    id(top): actual_mid - half_gap,
                    id(bottom): actual_mid + half_gap,
                }
                logger.info("Align(v): 2-pad symmetric about actual mid Y=%.3f, gap=%.3f",
                            actual_mid, half_gap * 2)

        corner_radii_mm = []
        for p in grp:
            cv_cr_px = p["cand"].get("_cv_corner_radius_px")
            if cv_cr_px is not None and cv_cr_px > 0.5:
                corner_radii_mm.append(cv_cr_px / pixels_per_mm)
                continue
            poly = p["poly"]
            if len(poly) < 4:
                continue
            xs_p = [pt["x_mm"] for pt in poly]
            ys_p = [pt["y_mm"] for pt in poly]
            bx1, by1 = min(xs_p), min(ys_p)
            bx2, by2 = max(xs_p), max(ys_p)
            short = min(bx2 - bx1, by2 - by1)
            if short < 0.1:
                continue
            search_r = short * 0.45
            corners = [(bx1, by1), (bx2, by1), (bx2, by2), (bx1, by2)]
            radii = []
            for ccx, ccy in corners:
                min_d = float('inf')
                for pt in poly:
                    if abs(pt["x_mm"] - ccx) > search_r or abs(pt["y_mm"] - ccy) > search_r:
                        continue
                    d = math.hypot(pt["x_mm"] - ccx, pt["y_mm"] - ccy)
                    if d < min_d:
                        min_d = d
                if 0 < min_d < float('inf'):
                    radii.append(min_d / (math.sqrt(2) - 1))
            if len(radii) >= 3:
                radii_s = sorted(radii)
                med_r = radii_s[len(radii_s) // 2]
                good = [r for r in radii if 0.6 * med_r <= r <= 1.4 * med_r]
                corner_radii_mm.append(sum(good) / len(good) if good else med_r)
            elif radii:
                corner_radii_mm.append(sum(radii) / len(radii))

        grp_radius_mm = round(statistics.median(corner_radii_mm), 3) if corner_radii_mm else round(min(med_w, med_h) * 0.1, 3)
        grp_radius_mm = max(0.05, min(grp_radius_mm, min(med_w, med_h) / 2))

        cv_refined_cxs: list[float] = []
        for p in grp:
            shift_mm = p["cand"].get("_cv_shift_mm", 0)
            if shift_mm >= 0.25:
                cv_refined_cxs.append(p["cx"])

        if cv_refined_cxs:
            cv_med = statistics.median(cv_refined_cxs)
            logger.info(
                "Align(%s): using CV-refined anchor X=%.3f (from %d pads) "
                "instead of group median %.3f",
                axis, cv_med, len(cv_refined_cxs), med_align,
            )
            med_align = cv_med

        hw = med_w / 2
        hh = med_h / 2
        for p in grp:
            if axis == "v":
                new_cx = med_align
                new_cy = target_positions.get(id(p), p["cy"]) if target_positions else p["cy"]
            else:
                new_cx = target_positions.get(id(p), p["cx"]) if target_positions else p["cx"]
                new_cy = med_align

            new_poly = [
                {"x_mm": round(new_cx - hw, 3), "y_mm": round(new_cy - hh, 3)},
                {"x_mm": round(new_cx + hw, 3), "y_mm": round(new_cy - hh, 3)},
                {"x_mm": round(new_cx + hw, 3), "y_mm": round(new_cy + hh, 3)},
                {"x_mm": round(new_cx - hw, 3), "y_mm": round(new_cy + hh, 3)},
            ]
            p["poly"].clear()
            p["poly"].extend(new_poly)

            center = p["region"].get("center", {})
            center["x_mm"] = round(new_cx, 3)
            center["y_mm"] = round(new_cy, 3)

            bbox = p["region"].get("bbox", {})
            if bbox:
                bbox["x_mm"] = round(new_cx - hw, 3)
                bbox["y_mm"] = round(new_cy - hh, 3)
                bbox["width_mm"] = round(med_w, 3)
                bbox["height_mm"] = round(med_h, 3)

            vp = p["cand"].get("visible_position")
            if vp:
                vp["x_mm"] = round(new_cx, 3)
                vp["y_mm"] = round(new_cy, 3)

            p["cand"]["width_mm"] = round(med_w, 3)
            p["cand"]["height_mm"] = round(med_h, 3)
            p["cand"]["corner_radius_mm"] = grp_radius_mm

            logger.info("Align(%s): %s → center=(%.3f,%.3f) size=%.3fx%.3f",
                        axis, p["cand"].get("label", "?"),
                        new_cx, new_cy, med_w, med_h)

    return result


def _align_pad_groups(result, pixels_per_mm=1.0, pcb_w_mm=0.0, pcb_h_mm=0.0):
    """Enforce alignment, uniform size, and even spacing among pad groups.

    PCB layout conventions:
      - Pads in a column (similar X) are vertically aligned → same X center.
      - Pads in a row (similar Y) are horizontally aligned → same Y center.
      - All pads in a group share the SAME width & height.
      - 3+ pads in a group: check for uniform spacing; if roughly even,
        snap to perfectly even spacing.
      - 2 pads: just align center axis + unify size.
      - Ungrouped pads: left untouched.

    The 'regular' majority determines the truth; outliers are corrected.
    """
    import statistics

    candidates = result.get("candidates", [])
    if len(candidates) < 2:
        return result

    # ── Gather pad info ──
    pads: list[dict] = []
    for cand in candidates:
        regions = cand.get("matched_regions", [])
        region = regions[0] if regions else cand.get("visible_region", {})
        poly = region.get("polygon") or []
        center = region.get("center", {})
        if len(poly) < 3 or not center:
            # Fallback: try to reconstruct from width_mm / height_mm
            cx = center.get("x_mm")
            cy = center.get("y_mm")
            cand_w = cand.get("width_mm") or cand.get("visible_region", {}).get("bbox", {}).get("width_mm")
            cand_h = cand.get("height_mm") or cand.get("visible_region", {}).get("bbox", {}).get("height_mm")
            if cx is not None and cy is not None and cand_w and cand_h:
                hw, hh = cand_w / 2, cand_h / 2
                poly = [
                    {"x_mm": cx - hw, "y_mm": cy - hh},
                    {"x_mm": cx + hw, "y_mm": cy - hh},
                    {"x_mm": cx + hw, "y_mm": cy + hh},
                    {"x_mm": cx - hw, "y_mm": cy + hh},
                ]
            else:
                continue
        xs = [p["x_mm"] for p in poly]
        ys = [p["y_mm"] for p in poly]
        w = max(xs) - min(xs)
        h = max(ys) - min(ys)
        pads.append({
            "cand": cand, "region": region, "poly": poly,
            "cx": center.get("x_mm", 0), "cy": center.get("y_mm", 0),
            "w": w, "h": h,
        })

    if len(pads) < 2:
        return result

    # ── Tolerance for grouping ──
    typical = statistics.median([p["w"] for p in pads] + [p["h"] for p in pads])
    if typical < 0.01:
        return result
    tol = typical * 0.6  # generous tolerance for grouping

    # ── Find aligned groups ──
    used: set[int] = set()
    groups: list[tuple[str, list[int]]] = []

    for i in range(len(pads)):
        if i in used:
            continue
        # Vertical group: similar X center (column)
        v_grp = [j for j in range(len(pads))
                 if j not in used and abs(pads[j]["cx"] - pads[i]["cx"]) < tol]
        if len(v_grp) >= 2:
            groups.append(("v", v_grp))
            used.update(v_grp)
            continue
        # Horizontal group: similar Y center (row)
        h_grp = [j for j in range(len(pads))
                 if j not in used and abs(pads[j]["cy"] - pads[i]["cy"]) < tol]
        if len(h_grp) >= 2:
            groups.append(("h", h_grp))
            used.update(h_grp)

    # ── Process each group ──
    for axis, indices in groups:
        grp = [pads[i] for i in indices]
        n = len(grp)

        # 1) Unified dimensions (median)
        med_w = statistics.median([p["w"] for p in grp])
        med_h = statistics.median([p["h"] for p in grp])

        # 2) Unified alignment coordinate
        #    Use median of the group's actual detected positions on the
        #    alignment axis.  (Previously, 2-pad groups were snapped to
        #    PCB center, but that broke cases where aligned pads are
        #    offset from center — e.g. side-by-side terminal columns or
        #    ID+TH pairs near an edge.)
        if axis == "v":
            med_align = statistics.median([p["cx"] for p in grp])
        else:
            med_align = statistics.median([p["cy"] for p in grp])

        # 3) For 3+ pads: check uniform spacing along the other axis
        target_positions = None  # per-pad target along free axis
        if n >= 3:
            # Sort by free-axis position
            if axis == "v":
                ordered = sorted(grp, key=lambda p: p["cy"])
                coords = [p["cy"] for p in ordered]
            else:
                ordered = sorted(grp, key=lambda p: p["cx"])
                coords = [p["cx"] for p in ordered]

            gaps = [coords[i+1] - coords[i] for i in range(n - 1)]
            if len(gaps) == 0:
                continue
            med_gap = statistics.median(gaps)

            # ═══ Enforce minimum spacing: pads within a group MUST NOT overlap ═══
            # The uniform pad size along the free axis is med_h (vertical group)
            # or med_w (horizontal group).  Spacing between centres must be at
            # least the pad size so that adjacent pads do not overlap.
            pad_size_along_free = med_h if axis == "v" else med_w
            if med_gap < pad_size_along_free:
                # Unrealistically tight spacing — likely caused by an outlier
                # dragging the median down.  Fall back to the maximum gap to
                # preserve the larger, more plausible spacing, but never let
                # it drop below the pad size.
                max_gap = max(gaps)
                if max_gap >= pad_size_along_free:
                    med_gap = max_gap
                else:
                    med_gap = pad_size_along_free
                logger.info(
                    "Align(%s): median gap %.3f < pad size %.3f → bumped to %.3f",
                    axis, statistics.median(gaps), pad_size_along_free, med_gap,
                )

            # For 4+ pads in a column/row, try splitting at the spatial
            # midpoint into two natural halves (e.g. top vs bottom for a
            # vertical group). This uses pure spatial proximity — no labels.
            # Each half gets independent uniform spacing.
            clusters = [ordered]
            if n >= 4:
                mid = (coords[0] + coords[-1]) / 2
                top_half = [p for p in ordered
                            if (p["cy"] if axis == "v" else p["cx"]) < mid]
                bot_half = [p for p in ordered
                            if (p["cy"] if axis == "v" else p["cx"]) >= mid]
                if len(top_half) >= 2 and len(bot_half) >= 2:
                    clusters = [top_half, bot_half]

            # For each cluster, enforce uniform spacing independently
            target_positions = {}
            for cluster in clusters:
                cn = len(cluster)
                if cn >= 2:
                    if axis == "v":
                        cl_ordered = sorted(cluster, key=lambda p: p["cy"])
                        cl_coords = [p["cy"] for p in cl_ordered]
                    else:
                        cl_ordered = sorted(cluster, key=lambda p: p["cx"])
                        cl_coords = [p["cx"] for p in cl_ordered]
                    if cn >= 3:
                        cl_gaps = [cl_coords[i+1] - cl_coords[i] for i in range(cn - 1)]
                        cl_med_gap = statistics.median(cl_gaps)

                        # ═══ Cluster-level minimum spacing ═══
                        # Pads within a cluster must not overlap.  The uniform
                        # pad size along the free axis sets the absolute floor.
                        cl_pad_size = med_h if axis == "v" else med_w
                        cl_min_gap = cl_pad_size + 0.15  # at least 0.15 mm visual gap
                        if cl_med_gap < cl_min_gap:
                            # Try max gap first; if it fits the cluster span, use it.
                            cl_max_gap = max(cl_gaps)
                            span = cl_coords[-1] - cl_coords[0]
                            max_fit = span / (cn - 1)
                            if cl_max_gap >= cl_min_gap and cl_max_gap <= max_fit:
                                cl_med_gap = cl_max_gap
                            elif max_fit >= cl_pad_size:
                                # Can't fit with ideal gap, but at least avoid overlap
                                cl_med_gap = max_fit
                            else:
                                # Physically impossible to fit cn pads without overlap.
                                # Keep original positions — uniform spacing would lie.
                                logger.warning(
                                    "Align cluster(%s): can't fit %d pads in span %.2f "
                                    "(pad=%.2f) — keeping original positions",
                                    axis, cn, span, cl_pad_size,
                                )
                                cl_med_gap = 0  # skip uniform spacing

                        if cl_med_gap > 0:
                            start = cl_coords[0]
                            for idx_p, p in enumerate(cl_ordered):
                                target_positions[id(p)] = start + idx_p * cl_med_gap
                    elif cn == 2:
                        actual_mid = (cl_coords[0] + cl_coords[1]) / 2
                        half_gap = abs(cl_coords[0] - cl_coords[1]) / 2
                        first = cl_ordered[0]
                        second = cl_ordered[1]
                        target_positions[id(first)] = actual_mid - half_gap
                        target_positions[id(second)] = actual_mid + half_gap

        # 3b) For exactly 2 pads on the free axis:
        #     Symmetrize about the ACTUAL detected midpoint (optimal fit).
        #     This is the mathematically optimal position: minimizes total
        #     squared displacement from actual detected positions.
        #     Each pad moves toward/away from the actual midpoint along the
        #     free axis only, preserving the detected gap.
        if n == 2:
            if axis == "h":
                # Horizontal pair: free axis is X
                actual_mid = (grp[0]["cx"] + grp[1]["cx"]) / 2
                half_gap = abs(grp[0]["cx"] - grp[1]["cx"]) / 2
                left = grp[0] if grp[0]["cx"] < grp[1]["cx"] else grp[1]
                right = grp[1] if grp[0]["cx"] < grp[1]["cx"] else grp[0]
                target_positions = {
                    id(left): actual_mid - half_gap,
                    id(right): actual_mid + half_gap,
                }
                logger.info("Align(h): 2-pad symmetric about actual mid X=%.3f, gap=%.3f",
                            actual_mid, half_gap * 2)
            elif axis == "v":
                # Vertical pair: free axis is Y
                actual_mid = (grp[0]["cy"] + grp[1]["cy"]) / 2
                half_gap = abs(grp[0]["cy"] - grp[1]["cy"]) / 2
                top = grp[0] if grp[0]["cy"] < grp[1]["cy"] else grp[1]
                bottom = grp[1] if grp[0]["cy"] < grp[1]["cy"] else grp[0]
                target_positions = {
                    id(top): actual_mid - half_gap,
                    id(bottom): actual_mid + half_gap,
                }
                logger.info("Align(v): 2-pad symmetric about actual mid Y=%.3f, gap=%.3f",
                            actual_mid, half_gap * 2)

        # 4) Estimate corner radius: prefer CV contour-based radius (accurate),
        #    fall back to polygon-based estimation, then default.
        import math
        corner_radii_mm = []
        for p in grp:
            # Try CV contour-based radius first (stored during CV refinement)
            cv_cr_px = p["cand"].get("_cv_corner_radius_px")
            if cv_cr_px is not None and cv_cr_px > 0.5:
                corner_radii_mm.append(cv_cr_px / pixels_per_mm)
                continue
            # Fallback: estimate from original polygon
            poly = p["poly"]
            if len(poly) < 4:
                continue
            xs_p = [pt["x_mm"] for pt in poly]
            ys_p = [pt["y_mm"] for pt in poly]
            bx1, by1 = min(xs_p), min(ys_p)
            bx2, by2 = max(xs_p), max(ys_p)
            short = min(bx2 - bx1, by2 - by1)
            if short < 0.1:
                continue
            search_r = short * 0.45
            corners = [(bx1, by1), (bx2, by1), (bx2, by2), (bx1, by2)]
            radii = []
            for ccx, ccy in corners:
                min_d = float('inf')
                for pt in poly:
                    if abs(pt["x_mm"] - ccx) > search_r or abs(pt["y_mm"] - ccy) > search_r:
                        continue
                    d = math.hypot(pt["x_mm"] - ccx, pt["y_mm"] - ccy)
                    if d < min_d:
                        min_d = d
                if 0 < min_d < float('inf'):
                    radii.append(min_d / (math.sqrt(2) - 1))
            if len(radii) >= 3:
                radii_s = sorted(radii)
                med_r = radii_s[len(radii_s) // 2]
                good = [r for r in radii if 0.6 * med_r <= r <= 1.4 * med_r]
                corner_radii_mm.append(sum(good) / len(good) if good else med_r)
            elif radii:
                corner_radii_mm.append(sum(radii) / len(radii))

        # Group unified corner radius (median)
        grp_radius_mm = round(statistics.median(corner_radii_mm), 3) if corner_radii_mm else round(min(med_w, med_h) * 0.1, 3)
        # Clamp to valid range
        grp_radius_mm = max(0.05, min(grp_radius_mm, min(med_w, med_h) / 2))

        # ── CV-refined position preservation ──
        # If CV refine shifted any pad in this group by ≥ 0.25 mm, its
        # position is more trustworthy than VLM's raw coordinate.  Those
        # pads become group anchors — the remaining pads snap to their
        # median, NOT the overall median (which is still polluted by
        # unrefined VLM positions).
        cv_refined_cxs: list[float] = []
        for p in grp:
            shift_mm = p["cand"].get("_cv_shift_mm", 0)
            if shift_mm >= 0.25:
                # Use the CURRENT center (already shifted by CV refine)
                cv_refined_cxs.append(p["cx"])

        if cv_refined_cxs:
            # Recompute alignment anchor from CV-refined positions only
            cv_med = statistics.median(cv_refined_cxs)
            logger.info(
                "Align(%s): using CV-refined anchor X=%.3f (from %d pads) "
                "instead of group median %.3f",
                axis, cv_med, len(cv_refined_cxs), med_align,
            )
            med_align = cv_med

        # 5) Apply corrections to each pad in the group
        #    Position logic:
        #    - Alignment axis: snap to group median (pads are aligned)
        #    - Free axis: keep each pad's own CV-refined center (true position)
        #      OR uniform spacing if 3+ pads are evenly distributed
        #    Shape: reconstruct a perfect symmetric rectangle (med_w × med_h)
        #    centered at the new position. This ensures all pads in the group
        #    have IDENTICAL polygon shapes, eliminating asymmetry from damage.
        hw = med_w / 2
        hh = med_h / 2
        for p in grp:
            # Determine new center (anchor = CV-refined geometric center)
            if axis == "v":
                new_cx = med_align
                new_cy = target_positions.get(id(p), p["cy"]) if target_positions else p["cy"]
            else:
                new_cx = target_positions.get(id(p), p["cx"]) if target_positions else p["cx"]
                new_cy = med_align

            # Reconstruct polygon as perfect symmetric rectangle
            # centered at (new_cx, new_cy) with unified dimensions
            new_poly = [
                {"x_mm": round(new_cx - hw, 3), "y_mm": round(new_cy - hh, 3)},
                {"x_mm": round(new_cx + hw, 3), "y_mm": round(new_cy - hh, 3)},
                {"x_mm": round(new_cx + hw, 3), "y_mm": round(new_cy + hh, 3)},
                {"x_mm": round(new_cx - hw, 3), "y_mm": round(new_cy + hh, 3)},
            ]
            # Replace polygon in-place
            p["poly"].clear()
            p["poly"].extend(new_poly)

            # Update center
            center = p["region"].get("center", {})
            center["x_mm"] = round(new_cx, 3)
            center["y_mm"] = round(new_cy, 3)

            # Update bbox
            bbox = p["region"].get("bbox", {})
            if bbox:
                bbox["x_mm"] = round(new_cx - hw, 3)
                bbox["y_mm"] = round(new_cy - hh, 3)
                bbox["width_mm"] = round(med_w, 3)
                bbox["height_mm"] = round(med_h, 3)

            # Update visible_position
            vp = p["cand"].get("visible_position")
            if vp:
                vp["x_mm"] = round(new_cx, 3)
                vp["y_mm"] = round(new_cy, 3)

            # Update candidate-level dimensions and unified corner radius
            p["cand"]["width_mm"] = round(med_w, 3)
            p["cand"]["height_mm"] = round(med_h, 3)
            p["cand"]["corner_radius_mm"] = grp_radius_mm

            logger.info("Align(%s): %s → center=(%.3f,%.3f) size=%.3fx%.3f",
                        axis, p["cand"].get("label", "?"),
                        new_cx, new_cy, med_w, med_h)

    return result


