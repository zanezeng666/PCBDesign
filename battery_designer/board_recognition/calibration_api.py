"""Calibration query API endpoints.

Moved from ``routers/calibration.py``.  Uses the shared
``_pcb_crop_offset_mm`` / ``_offset_candidate`` helpers from
``calibration_utils`` (no more local duplicates).
"""

from __future__ import annotations

import datetime
import json
import logging

import cv2
import numpy as np
from fastapi import APIRouter
from fastapi.responses import FileResponse

from ..core.errors import DesignError
from ..core.config import WORK_ROOT
from .calibration_utils import _pcb_crop_offset_mm, _offset_candidate

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/api/calibrations/with-recognition")
def list_calibrations_with_recognition():
    """List calibration records that have PCB recognition results."""
    cal_dir = WORK_ROOT / "calibrations"
    items = []
    if cal_dir.exists():
        for directory in sorted(cal_dir.iterdir(), reverse=True):
            if not directory.is_dir():
                continue
            outline_path = directory / "pcb_outline.json"
            candidates_path = directory / "terminal-candidates.json"
            if not (outline_path.exists() and candidates_path.exists()):
                continue
            entry = {"calibration_id": directory.name}
            try:
                cand_data = json.loads(candidates_path.read_text(encoding="utf-8"))
                cands = cand_data.get("candidates", [])
                entry["side"] = cand_data.get("side", "front")
                entry["candidate_count"] = len(cands)
                entry["labels"] = [c.get("label", "") for c in cands]
            except Exception:
                entry["candidate_count"] = 0
                entry["labels"] = []
            try:
                outline_data = json.loads(outline_path.read_text(encoding="utf-8"))
                entry["outline_points"] = len(outline_data.get("outline", []))
            except Exception:
                entry["outline_points"] = 0
            try:
                mtime = outline_path.stat().st_mtime
                entry["created"] = datetime.datetime.fromtimestamp(mtime).strftime("%Y-%m-%d %H:%M")
            except Exception:
                entry["created"] = ""
            items.append(entry)
    return {"calibrations": items}


@router.get("/api/calibrations/{calibration_id}/recognition")
def get_calibration_recognition(calibration_id: str):
    """Load saved PCB recognition results (outline + terminal candidates)."""
    if len(calibration_id) != 32 or any(c not in "0123456789abcdef" for c in calibration_id):
        raise DesignError("INVALID_CALIBRATION_ID", "The calibration id is invalid.")
    directory = WORK_ROOT / "calibrations" / calibration_id
    outline_path = directory / "pcb_outline.json"
    candidates_path = directory / "terminal-candidates.json"
    if not outline_path.exists():
        raise DesignError("RECOGNITION_NOT_FOUND", "No PCB outline found for this calibration.", {"calibration_id": calibration_id})
    outline_data = json.loads(outline_path.read_text(encoding="utf-8"))
    result = {
        "calibration_id": calibration_id,
        "outline": outline_data.get("outline", []),
        "grooves": outline_data.get("grooves", []),
        "pixels_per_mm": outline_data.get("pixels_per_mm", 0.0),
        "candidates": [],
        "side": "front",
        "width_mm": 0.0,
        "height_mm": 0.0,
    }
    if candidates_path.exists():
        cand_data = json.loads(candidates_path.read_text(encoding="utf-8"))
        candidates = cand_data.get("candidates", [])
        origin = (cand_data.get("coordinate_system") or {}).get("origin")
        dx, dy = _pcb_crop_offset_mm(directory) if origin == "pcb_top_left" else (0.0, 0.0)
        if dx or dy:
            for cand in candidates:
                _offset_candidate(cand, dx, dy)
        result["candidates"] = candidates
        result["side"] = cand_data.get("side", "front")
        result["width_mm"] = cand_data.get("width_mm", 0.0)
        result["height_mm"] = cand_data.get("height_mm", 0.0)
    return result


@router.get("/api/calibrations/{calibration_id}/rectified.png")
def get_rectified_image(calibration_id: str):
    """Serve the rectified PNG image for preview in the upload zone."""
    if len(calibration_id) != 32 or any(c not in "0123456789abcdef" for c in calibration_id):
        raise DesignError("INVALID_CALIBRATION_ID", "The calibration id is invalid.")
    rectified_path = WORK_ROOT / "calibrations" / calibration_id / "rectified.png"
    if not rectified_path.exists():
        raise DesignError("CALIBRATION_NOT_FOUND", "Rectified image not found.", {"calibration_id": calibration_id})
    return FileResponse(rectified_path, media_type="image/png")
