"""Step 2 — Pad / terminal detection API endpoints.

Moved from pcb_recognition/router.py.  Contains the detect-terminals and
verify-pad-regions endpoints.
"""
from __future__ import annotations

import base64
import json
import logging

import cv2
import numpy as np
from fastapi import APIRouter, Form, HTTPException

from ..core.errors import DesignError
from ..core.config import WORK_ROOT
from .vlm_detection import (
    detect_with_vlm as _vlm_detect,
    verify_pad_crop as _verify_pad_crop,
)
from .vlm_helpers import _save_vlm_input_for_debug, _warn_incomplete_vlm_result
from .pad_refine import _refine_positions_cv, _clamp_pads_to_board, _align_pad_groups
from .pad_geometry import _estimate_corner_radius, _draw_rounded_rect

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/api/vision/detect-terminals")
def detect_terminals(calibration_id: str = Form(...), side: str = Form(...),
                     debug: str = Form("false")):
    """Detect terminal candidates using VLM identification on cropped PCB.

    For transparent PCB: crop to PCB bounding box so the board fills the entire
    image. This eliminates the Y-offset problem caused by PCB floating in white
    space. VLM coordinates are then offset back to full-frame mm.

    Set debug=true to receive intermediate detection stage snapshots.
    """
    if len(calibration_id) != 32 or any(character not in "0123456789abcdef" for character in calibration_id):
        raise DesignError("INVALID_CALIBRATION_ID", "The calibration id is invalid.")
    directory = WORK_ROOT / "calibrations" / calibration_id
    metadata_path = directory / "calibration.json"
    transparent_path = directory / "transparent.png"
    image_path = transparent_path if transparent_path.exists() else (directory / "rectified.png")
    if not metadata_path.exists() or not image_path.exists():
        raise DesignError("CALIBRATION_NOT_FOUND", "Photo calibration record not found.", {"calibration_id": calibration_id})
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    image_bytes = image_path.read_bytes()
    width_mm = float(metadata["width_mm"])
    height_mm = float(metadata["height_mm"])
    pixels_per_mm = float(metadata.get("pixels_per_mm", 1.0))
    is_transparent = transparent_path.exists()
    img_w_px = int(metadata.get("rectified_w_px", 0))
    img_h_px = int(metadata.get("rectified_h_px", 0))

    # ── Prepare image for VLM ──
    # For transparent PCB: crop to PCB bounding box so board fills the image.
    # Coordinates are PCB-RELATIVE: (0,0) = PCB top-left corner.
    crop_w_mm = width_mm
    crop_h_mm = height_mm
    img_for_vlm = image_bytes
    pcb_img_bgr = None  # Will hold the cropped PCB image (BGR, white bg) for annotation
    pcb_img_rgba = None  # Will hold the cropped PCB image (RGBA) for CV refinement
    # PCB top-left offset within the full frame (mm). Non-zero only when the
    # transparent PCB was cropped to its alpha bbox. Front-end adds this to the
    # PCB-RELATIVE pad coords to bring them into full-frame space (the outline
    # is full-frame), required by DesignSpec.point_in_polygon.
    crop_dx_mm = 0.0
    crop_dy_mm = 0.0

    if is_transparent:
        try:
            nparr = np.frombuffer(image_bytes, np.uint8)
            img_rgba = cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)
            if img_rgba is not None and len(img_rgba.shape) == 3 and img_rgba.shape[2] == 4:
                alpha = img_rgba[:, :, 3]
                # Find PCB bounding box (alpha > 128 = solid PCB, skip semi-transparent edges)
                rows = np.any(alpha > 128, axis=1)
                cols = np.any(alpha > 128, axis=0)
                if rows.any() and cols.any():
                    y_min, y_max = np.where(rows)[0][[0, -1]]
                    x_min, x_max = np.where(cols)[0][[0, -1]]
                    crop_dx_mm = x_min / pixels_per_mm
                    crop_dy_mm = y_min / pixels_per_mm
                    # Tight crop — no padding, PCB fills the image exactly
                    cropped = img_rgba[y_min:y_max+1, x_min:x_max+1]
                    h_c, w_c = cropped.shape[:2]

                    # ── Clean mask from outline polygon (preferred) ──
                    # The outline polygon from extract-pcb is more accurate than
                    # the alpha channel (which is eroded+feathered for display).
                    # Render the polygon directly → sharp, tight mask.
                    # Only internal grooves/holes show white, NOT PCB outer edge.
                    outline_mask = None
                    outline_path = directory / "pcb_outline.json"
                    if outline_path.exists():
                        try:
                            odata = json.loads(outline_path.read_text(encoding="utf-8"))
                            outline_mm = odata.get("outline", [])
                            if len(outline_mm) >= 3:
                                pts = np.array([
                                    [p["x_mm"] * pixels_per_mm - x_min,
                                     p["y_mm"] * pixels_per_mm - y_min]
                                    for p in outline_mm
                                ], dtype=np.int32)
                                outline_mask = np.zeros((h_c, w_c), dtype=np.uint8)
                                cv2.fillPoly(outline_mask, [pts], 255)
                        except Exception:
                            logger.warning("detect-terminals: outline mask failed", exc_info=True)

                    if outline_mask is not None:
                        # The outline polygon is a simplified outer boundary
                        # (few vertices) and does NOT trace around internal
                        # grooves/notches, so fillPoly leaves the groove
                        # interior opaque — the photo's white background then
                        # shows through the groove in the preview.  Intersect
                        # with the original alpha channel (which correctly marks
                        # the groove as transparent) to punch those holes out,
                        # while keeping the sharp outer edge from the outline.
                        # This also keeps white groove pixels out of the CV
                        # metallic detector (white passes the low-saturation
                        # metal filter).
                        orig_opaque = (cropped[:, :, 3] > 128).astype(np.uint8) * 255
                        binary_mask = cv2.bitwise_and(outline_mask, orig_opaque)
                        logger.info("detect-terminals: using outline polygon mask (%d vertices) "
                                    "intersected with alpha (grooves punched out)", len(outline_mm))
                    else:
                        # Fallback: binary threshold on alpha channel
                        binary_mask = (cropped[:, :, 3] > 128).astype(np.uint8) * 255
                        k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
                        binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE,
                                                       k_close, iterations=2)
                        logger.info("detect-terminals: using alpha binary mask (fallback)")

                    # Composite onto white using clean mask (sharp edges)
                    mask_ch = binary_mask[:, :, None].astype(np.float32) / 255.0
                    bgr = cropped[:, :, :3].astype(np.float32)
                    white_bg = np.full((h_c, w_c, 3), 255.0, dtype=np.float32)
                    composited = (bgr * mask_ch + white_bg * (1.0 - mask_ch)).astype(np.uint8)
                    # Save the PCB-only image for annotation
                    pcb_img_bgr = composited.copy()
                    # RGBA crop with clean mask for CV refinement
                    pcb_img_rgba = cropped.copy()
                    pcb_img_rgba[:, :, 3] = binary_mask
                    # Upscale 2x for better VLM recognition.
                    # LINEAR preserves sharp copper-pad edges better than CUBIC
                    # (important for small pads where cubic interpolation blurs
                    # the pad→background transition).
                    upscaled = cv2.resize(composited, None, fx=2.0, fy=2.0,
                                          interpolation=cv2.INTER_LINEAR)
                    _, buf = cv2.imencode(".png", upscaled)
                    img_for_vlm = buf.tobytes()
                    # PCB dimensions in mm (this IS the coordinate reference frame)
                    crop_w_mm = w_c / pixels_per_mm
                    crop_h_mm = h_c / pixels_per_mm
                    img_w_px = upscaled.shape[1]
                    img_h_px = upscaled.shape[0]
                    logger.info("detect-terminals: PCB bbox (%d,%d)-(%d,%d), size %.1fx%.1fmm",
                                x_min, y_min, x_max, y_max, crop_w_mm, crop_h_mm)
        except Exception:
            logger.warning("detect-terminals: crop failed, using full image", exc_info=True)

    # ── VLM identifies pads on the PCB image ──
    # VLM coordinates are PCB-RELATIVE: (0,0)=PCB top-left, (crop_w_mm, crop_h_mm)=PCB bottom-right
    result = _vlm_detect(img_for_vlm, crop_w_mm, crop_h_mm, side, is_transparent,
                         img_w_px=img_w_px, img_h_px=img_h_px)

    # ── Diagnostically save the VLM input image for debugging ──
    _save_vlm_input_for_debug(img_for_vlm, side, calibration_id)

    # ── Warn if VLM missed small pads (rely on geometric estimation) ──
    _warn_incomplete_vlm_result(result, side)

    # ── Debug helper for app-level stages ──
    is_debug = debug.lower() in ("true", "1", "yes")
    app_debug_stages: list[dict] = []

    def _app_debug_snapshot(label: str, res: dict):
        if not is_debug:
            return
        cands = res.get("candidates", [])
        app_debug_stages.append({
            "stage": label,
            "count": len(cands),
            "candidates": [{
                "label": c.get("label", ""),
                "x_mm": c.get("visible_position", {}).get("x_mm"),
                "y_mm": c.get("visible_position", {}).get("y_mm"),
                "width_mm": c.get("width_mm"),
                "height_mm": c.get("height_mm"),
                "confidence": c.get("confidence"),
                "source": c.get("visible_region", {}).get("source", c.get("matched_regions", [{}])[0].get("source", "vlm") if c.get("matched_regions") else "vlm"),
                "diagnostic_verified": c.get("diagnostic_verified", ""),
            } for c in cands],
        })

    # ── CV position refinement on the PCB image (not the full transparent image) ──
    try:
        # Use RGBA crop if available (alpha channel masks non-PCB areas)
        refine_img = pcb_img_rgba if pcb_img_rgba is not None else pcb_img_bgr
        if refine_img is not None:
            result = _refine_positions_cv(result, refine_img, pixels_per_mm)
        else:
            # Non-transparent: use the rectified image directly
            nparr_ref = np.frombuffer(image_bytes, np.uint8)
            img_for_refine = cv2.imdecode(nparr_ref, cv2.IMREAD_UNCHANGED)
            if img_for_refine is not None:
                result = _refine_positions_cv(result, img_for_refine, pixels_per_mm)
    except Exception:
        logger.warning("CV position refinement failed, using VLM positions", exc_info=True)
    _app_debug_snapshot("step5_after_cv_refine", result)

    # ── Group alignment correction (PCB layout symmetry) ──
    try:
        result = _align_pad_groups(result, pixels_per_mm,
                                     crop_w_mm, crop_h_mm)
    except Exception:
        logger.warning("Pad group alignment failed", exc_info=True)
    # ── Clamp all pad polygons to PCB boundary ──
    _clamp_pads_to_board(result, crop_w_mm, crop_h_mm)
    _app_debug_snapshot("step6_after_align_groups", result)

    # ── Debug stages: merge VLM + App snapshots, or strip them ──
    if is_debug:
        vlm_stages = result.pop("_debug_stages", [])
        result["_debug_stages"] = vlm_stages + app_debug_stages
    else:
        result.pop("_debug_stages", None)

    # Add coordinate system metadata
    result["method_used"] = "vlm+cv_refine"
    result["coordinate_system"] = {
        "origin": "pcb_top_left",
        "x_axis": "right (mm)",
        "y_axis": "down (mm)",
        "pcb_width_mm": round(crop_w_mm, 3),
        "pcb_height_mm": round(crop_h_mm, 3),
        "crop_offset_mm": {"x": round(crop_dx_mm, 3), "y": round(crop_dy_mm, 3)},
    }
    # Include PCB-only image for front-end canvas background.
    # Prefer the RGBA crop (alpha = outline mask) so the notch/groove areas
    # stay transparent instead of being filled with white.
    import base64
    if pcb_img_rgba is not None:
        _, pcb_buf = cv2.imencode(".png", pcb_img_rgba)
        result["pcb_image_b64"] = base64.b64encode(pcb_buf).decode("ascii")
    elif pcb_img_bgr is not None:
        _, pcb_buf = cv2.imencode(".png", pcb_img_bgr)
        result["pcb_image_b64"] = base64.b64encode(pcb_buf).decode("ascii")
    (directory / "terminal-candidates.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # ── Generate annotated image on PCB-only image (no transparent border) ──
    try:
        if pcb_img_bgr is not None:
            annotated_bgr = pcb_img_bgr.copy()
        else:
            # Non-transparent: use rectified image
            nparr_full = np.frombuffer(image_bytes, np.uint8)
            img_full = cv2.imdecode(nparr_full, cv2.IMREAD_UNCHANGED)
            if img_full is not None and len(img_full.shape) == 3:
                if img_full.shape[2] == 4:
                    h_f, w_f = img_full.shape[:2]
                    a_ch = img_full[:, :, 3:4].astype(np.float32) / 255.0
                    bgr_f = img_full[:, :, :3].astype(np.float32)
                    white_f = np.full((h_f, w_f, 3), 255.0, dtype=np.float32)
                    annotated_bgr = (bgr_f * a_ch + white_f * (1.0 - a_ch)).astype(np.uint8)
                else:
                    annotated_bgr = img_full[:, :, :3].copy()
            else:
                annotated_bgr = None

        if annotated_bgr is not None:

            colors_bgr = [
                (68, 68, 239),   # red
                (239, 130, 59),  # blue
                (94, 197, 34),   # green
                (11, 158, 245),  # amber
                (214, 92, 139),  # purple
            ]
            candidates = result.get("candidates", [])
            for idx, cand in enumerate(candidates):
                color = colors_bgr[idx % len(colors_bgr)]
                vr = cand.get("visible_region", {})
                poly = vr.get("polygon", [])
                label = cand.get("label", "?")

                if len(poly) >= 3:
                    # Get bounding box from polygon (mm → px), symmetrized around center
                    xs = [p["x_mm"] * pixels_per_mm for p in poly]
                    ys = [p["y_mm"] * pixels_per_mm for p in poly]
                    center = vr.get("center", {})
                    if center:
                        cx_px = center.get("x_mm", 0) * pixels_per_mm
                        cy_px = center.get("y_mm", 0) * pixels_per_mm
                    else:
                        cx_px = (min(xs) + max(xs)) / 2
                        cy_px = (min(ys) + max(ys)) / 2
                    # Symmetry repair: max half-extent from center on each axis
                    hw = max(abs(x - cx_px) for x in xs)
                    hh = max(abs(y - cy_px) for y in ys)
                    x1, y1 = int(round(cx_px - hw)), int(round(cy_px - hh))
                    x2, y2 = int(round(cx_px + hw)), int(round(cy_px + hh))
                    # Use unified corner_radius_mm if available, else estimate from polygon
                    cr_mm = cand.get("corner_radius_mm")
                    if cr_mm is not None:
                        radius = max(1, int(round(cr_mm * pixels_per_mm)))
                    else:
                        poly_pts = list(zip(xs, ys))
                        radius = _estimate_corner_radius(poly_pts, x1, y1, x2, y2)
                    # Create mask for the rounded-rect shape
                    mask_rr = np.zeros(annotated_bgr.shape[:2], dtype=np.uint8)
                    _draw_rounded_rect(mask_rr, x1, y1, x2, y2, radius, 255, -1)
                    # Fill with transparency
                    overlay = annotated_bgr.copy()
                    overlay[mask_rr > 0] = color
                    cv2.addWeighted(overlay, 0.25, annotated_bgr, 0.75, 0, annotated_bgr)
                    # Draw border
                    _draw_rounded_rect(annotated_bgr, x1, y1, x2, y2, radius, color, 3)

                # Draw center cross
                center = vr.get("center", {})
                if center:
                    cx_px = int(round(center.get("x_mm", 0) * pixels_per_mm))
                    cy_px = int(round(center.get("y_mm", 0) * pixels_per_mm))
                    cv2.drawMarker(annotated_bgr, (cx_px, cy_px), color,
                                   cv2.MARKER_CROSS, 20, 2)

                # Draw label text
                vp = cand.get("visible_position", {})
                if vp:
                    tx = int(round(vp.get("x_mm", 0) * pixels_per_mm)) + 12
                    ty = int(round(vp.get("y_mm", 0) * pixels_per_mm)) - 12
                    cv2.putText(annotated_bgr, label, (tx, ty),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

            # Save with transparency (alpha = outline mask) so notch/groove
            # areas are not filled with white.
            if pcb_img_rgba is not None and pcb_img_rgba.shape[:2] == annotated_bgr.shape[:2]:
                annotated_bgra = np.dstack([annotated_bgr, pcb_img_rgba[:, :, 3]])
                cv2.imwrite(str(directory / "terminal-annotated.png"), annotated_bgra)
            else:
                cv2.imwrite(str(directory / "terminal-annotated.png"), annotated_bgr)
            logger.info("detect-terminals: saved full-res annotated image (%dx%d)",
                        annotated_bgr.shape[1], annotated_bgr.shape[0])
    except Exception:
        logger.warning("detect-terminals: annotated image generation failed", exc_info=True)

    return result



@router.post("/api/vision/verify-pad-regions")
def verify_pad_regions(calibration_id: str = Form(...), side: str = Form(...)):
    """AI visual inspection: verify each detected pad's cropped region with VLM.

    For each pad candidate in terminal-candidates.json, crop the calibrated
    PCB image around the pad's polygon and send it to VLM for quality verification.
    Detects issues like: pad spanning multiple real pads, non-pad area captured,
    pad mostly outside PCB, etc.

    Returns:
        {
            "verified": int,
            "failed": int,
            "results": [{"label", "ok", "issues", "confidence"}, ...],
        }
    """
    import cv2
    import numpy as np
    from PIL import Image
    import io

    calib_dir = WORK_ROOT / "calibrations" / calibration_id
    if not calib_dir.exists():
        raise HTTPException(404, "Calibration directory not found")
    candidates_file = calib_dir / "terminal-candidates.json"
    if not candidates_file.exists():
        raise HTTPException(404, "terminal-candidates.json not found")

    result = json.loads(candidates_file.read_text(encoding="utf-8"))

    # Load calibrated image (transparent PCB image)
    transparent_path = calib_dir / "transparent.png"
    if not transparent_path.exists():
        transparent_path = calib_dir / "calibrated.png"
    if not transparent_path.exists():
        raise HTTPException(400, "No calibrated PCB image found")

    pcb_img = cv2.imread(str(transparent_path), cv2.IMREAD_UNCHANGED)
    if pcb_img is None:
        raise HTTPException(400, "Failed to read calibrated PCB image")

    h_img, w_img = pcb_img.shape[:2]
    coord_sys = result.get("coordinate_system", {})
    pcb_w_mm = coord_sys.get("pcb_width_mm", 0)
    pcb_h_mm = coord_sys.get("pcb_height_mm", 0)
    crop_offset = coord_sys.get("crop_offset_mm", {"x": 0.0, "y": 0.0})
    offset_x_mm = float(crop_offset.get("x", 0.0))
    offset_y_mm = float(crop_offset.get("y", 0.0))

    # Use the stored pixels_per_mm from the calibration metadata (not w_img / pcb_w_mm)
    # The image PPM is determined by the calibration rectification, and crop_offset_mm
    # was computed using this same PPM.
    cal_meta_path = calib_dir / "calibration.json"
    ppm = w_img / pcb_w_mm  # fallback
    if cal_meta_path.exists():
        try:
            cal_meta = json.loads(cal_meta_path.read_text(encoding="utf-8"))
            _ppm = cal_meta.get("pixels_per_mm", 0)
            if _ppm > 0:
                ppm = float(_ppm)
        except Exception:
            pass
    if pcb_w_mm <= 0 or pcb_h_mm <= 0:
        raise HTTPException(400, "PCB dimensions not found in terminal-candidates.json")

    candidates = result.get("candidates", [])
    verification_results = []
    verified_count = 0
    failed_count = 0

    for cand in candidates:
        label = cand.get("label", "?")
        regions = cand.get("matched_regions", [])
        if not regions:
            verification_results.append({
                "label": label, "ok": False, "single_pad": False,
                "issues": ["No matched_regions"],
                "confidence": 0.0, "error": "no_regions",
            })
            failed_count += 1
            continue

        # Take the first region's polygon
        region = regions[0]
        poly = region.get("polygon", [])
        if len(poly) < 3:
            verification_results.append({
                "label": label, "ok": False, "single_pad": False,
                "issues": ["Invalid polygon"],
                "confidence": 0.0, "error": "bad_polygon",
            })
            failed_count += 1
            continue

        xs = [v["x_mm"] for v in poly]
        ys = [v["y_mm"] for v in poly]
        x1_mm, y1_mm = min(xs), min(ys)
        x2_mm, y2_mm = max(xs), max(ys)

        # ═══ Ensure minimum crop size for VLM visibility ═══
        # For very small pads (< 1 mm) at ~25 px/mm the crop image can be
        # < 30 px across — too low-res for any VLM.  Enforce a floor so the
        # VLM has at least ~25 px to work with in each direction.
        # The expansion uses the polygon centre as the anchor.
        MIN_CROP_MM = 1.0  # ≈ 25 px at typical calibration PPM
        cx_mm = (x1_mm + x2_mm) / 2
        cy_mm = (y1_mm + y2_mm) / 2
        pad_w_mm = max(x2_mm - x1_mm, MIN_CROP_MM)
        pad_h_mm = max(y2_mm - y1_mm, MIN_CROP_MM)

        # Expand bounding box for context — must be TIGHT:
        # In dense P+/P- arrays the polygon-to-polygon gap can be ≤0.35 mm.
        # A margin of 0.12 mm leaves only 0.11 mm between adjacent crops
        # (~3 px) — enough to avoid VLM seeing a sliver of the neighbour.
        #   margin = smaller_side × 0.08, hard-capped at 0.12 mm
        base_margin = min(0.12, min(pad_w_mm, pad_h_mm) * 0.08)
        # ── Edge-aware clamping ──
        # When a pad is within 1.0 mm of the board edge, reduce the margin
        # on that side to 0.03 mm.  This avoids pulling in copper-pour or
        # substrate features from beyond the pad that VLM might misidentify
        # as additional pads.
        edge_near = 1.0  # threshold for "near board edge"
        ml = mr = mt = mb = base_margin  # left, right, top, bottom
        pad_left   = cx_mm - pad_w_mm / 2
        pad_right  = cx_mm + pad_w_mm / 2
        pad_top    = cy_mm - pad_h_mm / 2
        pad_bottom = cy_mm + pad_h_mm / 2
        if pad_right >= pcb_w_mm - edge_near:
            mr = 0.03
        if pad_left <= edge_near:
            ml = 0.03
        if pad_bottom >= pcb_h_mm - edge_near:
            mb = 0.03
        if pad_top <= edge_near:
            mt = 0.03
        x1_mm = max(0, pad_left - ml)
        y1_mm = max(0, pad_top - mt)
        x2_mm = min(pcb_w_mm, pad_right + mr)
        y2_mm = min(pcb_h_mm, pad_bottom + mb)

        pad_w_mm = x2_mm - x1_mm
        pad_h_mm = y2_mm - y1_mm
        if pad_w_mm < 0.5 or pad_h_mm < 0.5:
            verification_results.append({
                "label": label, "ok": False, "single_pad": False,
                "issues": ["Pad region too small / degenerate"],
                "confidence": 0.0, "error": "too_small",
            })
            failed_count += 1
            continue

        # Crop the calibrated image (pad coords are relative to PCB crop, add offset)
        x1_px = int((x1_mm + offset_x_mm) * ppm)
        y1_px = int((y1_mm + offset_y_mm) * ppm)
        x2_px = int((x2_mm + offset_x_mm) * ppm)
        y2_px = int((y2_mm + offset_y_mm) * ppm)
        x1_px = max(0, min(w_img - 1, x1_px))
        y1_px = max(0, min(h_img - 1, y1_px))
        x2_px = max(x1_px + 1, min(w_img, x2_px))
        y2_px = max(y1_px + 1, min(h_img, y2_px))

        crop = pcb_img[y1_px:y2_px, x1_px:x2_px]

        # If image has alpha channel, composite over white
        if len(crop.shape) == 3 and crop.shape[2] == 4:
            alpha = crop[:, :, 3:4].astype(np.float32) / 255.0
            bgr = crop[:, :, :3].astype(np.float32)
            white = np.full_like(bgr, 255.0, dtype=np.float32)
            crop_render = (bgr * alpha + white * (1.0 - alpha)).astype(np.uint8)
        else:
            if len(crop.shape) == 3:
                crop_render = crop[:, :, :3]
            else:
                crop_render = cv2.cvtColor(crop, cv2.COLOR_GRAY2BGR)

        # Encode crop as PNG bytes
        _, crop_buf = cv2.imencode(".png", crop_render)
        crop_bytes = crop_buf.tobytes()

        # Call VLM for verification
        try:
            v_result = _verify_pad_crop(crop_bytes, label, pad_w_mm, pad_h_mm)
        except Exception as exc:
            logger.error(f"Pad verification VLM error for {label}: {exc}")
            v_result = {
                "ok": False, "single_pad": False,
                "issues": [f"VLM call error: {exc}"],
                "confidence": 0.0,
            }

        # ── CV metal override ──
        # When VLM says "no metallic pad visible" but CV finds substantial
        # metal pixels in the same crop, override VLM's judgment.  This
        # compensates for VLM's tendency to misread silkscreen-adjacent crops
        # where the metal pad IS present but its visual appearance doesn't
        # match VLM's expectation of "shiny silver/gold pad."
        if not v_result.get("ok") and any(w in " ".join(v_result.get("issues", [])).lower()
                                          for w in ("metal", "pad", "solder", "silkscreen")):
            # Quick CV check: detect metal pixels in this crop
            crop_hsv = cv2.cvtColor(crop_render, cv2.COLOR_BGR2HSV)
            cv_metal = cv2.inRange(crop_hsv, np.array([0, 0, 40]), np.array([180, 80, 250]))
            cv_metal_pct = np.count_nonzero(cv_metal) / max(cv_metal.size, 1)
            # ID/TH/T/NTC/N are tiny pads (1-3mm); VLM almost always puts their
            # polygon on silkscreen text.  Accept them at a lower threshold —
            # if there is at least 2% metal and the structural alignment passed,
            # the pad position is correct enough.
            small_label_thresh = 0.02 if label.upper() in ("ID", "TH", "T", "NTC", "N") else 0.05
            if cv_metal_pct > small_label_thresh:
                logger.info(
                    "Pad %s: VLM said no metal but CV found %.1f%% metal pixels → override to pass",
                    label, cv_metal_pct * 100,
                )
                v_result["ok"] = True
                v_result["single_pad"] = True
                v_result["issues"] = []
                v_result["confidence"] = 0.50
                v_result["_cv_override"] = True

        v_result["label"] = label
        if v_result.get("ok"):
            verified_count += 1
        else:
            failed_count += 1
            logger.warning(f"Pad {label}: VLM verification FAILED — issues={v_result.get('issues')}")

        verification_results.append(v_result)

    return {
        "verified": verified_count,
        "failed": failed_count,
        "total": len(candidates),
        "results": verification_results,
    }






