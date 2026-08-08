from __future__ import annotations

import json
import logging
import os
from pathlib import Path

import cv2
import numpy as np

from ..core.errors import DesignError

logger = logging.getLogger(__name__)

_ROOT = Path(__file__).resolve().parents[2]
WORK_ROOT = Path(os.getenv("BATTERY_DESIGN_WORKDIR", _ROOT / "work"))


def _pcb_crop_offset_mm(directory: Path) -> tuple[float, float]:
    """Return the PCB top-left offset (mm) within the full frame.

    detect-terminals crops transparent.png to the PCB alpha bounding box and
    reports pad coordinates relative to that crop's top-left. This recomputes
    the same bounding box so we can shift pads back into full-frame mm.
    Returns (0, 0) when no transparent.png exists (no crop was performed).
    """
    transparent_path = directory / "transparent.png"
    meta_path = directory / "calibration.json"
    if not transparent_path.exists() or not meta_path.exists():
        return (0.0, 0.0)
    try:
        metadata = json.loads(meta_path.read_text(encoding="utf-8"))
        pixels_per_mm = float(metadata.get("pixels_per_mm", 0.0))
        if pixels_per_mm <= 0:
            return (0.0, 0.0)
        img_rgba = cv2.imdecode(np.frombuffer(transparent_path.read_bytes(), np.uint8), cv2.IMREAD_UNCHANGED)
        if img_rgba is None or len(img_rgba.shape) != 3 or img_rgba.shape[2] != 4:
            return (0.0, 0.0)
        alpha = img_rgba[:, :, 3]
        rows = np.any(alpha > 128, axis=1)
        cols = np.any(alpha > 128, axis=0)
        if not rows.any() or not cols.any():
            return (0.0, 0.0)
        y_min = int(np.where(rows)[0][0])
        x_min = int(np.where(cols)[0][0])
        return (x_min / pixels_per_mm, y_min / pixels_per_mm)
    except Exception:
        logger.warning("recognition: failed to compute PCB crop offset", exc_info=True)
        return (0.0, 0.0)


def _offset_candidate(cand: dict, dx: float, dy: float) -> None:
    """Shift a terminal candidate's coordinates by (dx, dy) in-place."""
    def shift_point(pt: dict) -> None:
        if isinstance(pt, dict) and "x_mm" in pt and "y_mm" in pt:
            pt["x_mm"] = round(pt["x_mm"] + dx, 3)
            pt["y_mm"] = round(pt["y_mm"] + dy, 3)

    def shift_region(region: dict) -> None:
        if not isinstance(region, dict):
            return
        shift_point(region.get("center"))
        for pt in region.get("polygon", []) or []:
            shift_point(pt)
        bbox = region.get("bbox")
        if isinstance(bbox, dict):
            shift_point(bbox)

    shift_point(cand.get("visible_position"))
    shift_region(cand.get("visible_region"))
    shift_region(cand.get("text_region"))
    for region in cand.get("matched_regions", []) or []:
        shift_region(region)


def _load_calibration(calibration_id: str) -> tuple[bytes, float, float, float]:
    """Load rectified.png + metadata for a calibration_id.
    Returns (png_bytes, width_mm, height_mm, pixels_per_mm).
    Raises DesignError on failure.
    """
    if len(calibration_id) != 32 or any(c not in "0123456789abcdef" for c in calibration_id):
        raise DesignError("INVALID_CALIBRATION_ID", "The calibration id is invalid.")
    directory = WORK_ROOT / "calibrations" / calibration_id
    meta = directory / "calibration.json"
    img = directory / "rectified.png"
    if not meta.exists() or not img.exists():
        raise DesignError("CALIBRATION_NOT_FOUND",
                          "Photo calibration record not found.", {"calibration_id": calibration_id})
    metadata = json.loads(meta.read_text(encoding="utf-8"))
    return (img.read_bytes(),
            float(metadata["width_mm"]),
            float(metadata["height_mm"]),
            float(metadata.get("pixels_per_mm", max(
                int(metadata.get("pixel_width", 1000)) / metadata["width_mm"],
                int(metadata.get("pixel_height", 1000)) / metadata["height_mm"],
            ))))
