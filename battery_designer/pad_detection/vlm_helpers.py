from __future__ import annotations

import logging
import os
from pathlib import Path

import cv2
import numpy as np

logger = logging.getLogger(__name__)

_ROOT = Path(__file__).resolve().parents[2]
WORK_ROOT = Path(os.getenv("BATTERY_DESIGN_WORKDIR", _ROOT / "work"))

VLM_DIAG_DIR = WORK_ROOT / "diag_vlm"


def _save_vlm_input_for_debug(img_bgr, side: str, calibration_id: str) -> None:
    """Save the image sent to VLM for later debugging.

    The saved image helps diagnose WHY VLM missed certain pads — you can
    visually inspect exactly what VLM received and whether small pads
    are clearly visible.

    Saving is best-effort — failures are silently swallowed.
    """
    try:
        VLM_DIAG_DIR.mkdir(parents=True, exist_ok=True)
        cal_id = calibration_id[:12] if calibration_id else "unknown"
        out_path = VLM_DIAG_DIR / f"vlm_input_{side}_{cal_id}.png"
        cv2.imwrite(str(out_path), np.ascontiguousarray(img_bgr))
        logger.info("Saved VLM input image to %s (%dx%d)", out_path,
                     img_bgr.shape[1], img_bgr.shape[0])
    except Exception:
        pass  # best-effort


def _warn_incomplete_vlm_result(result: dict, side: str) -> None:
    """Warn if VLM returned too few pads — small pads likely missed.

    The detect-terminals pipeline can detect up to ~8 total pads per side:
      - Large: B+, B- (2 pads)
      - Medium: P+, P- (4-6 pads)
      - Small: TH/T, ID, NTC (2-3 pads)

    If VLM returns < 4 candidates, it almost certainly missed small pads
    (T, ID) and possibly some P+/P- pads.  The remaining positions will
    be filled by geometric estimation, which is much less accurate.
    """
    candidates = result.get("candidates", [])
    n = len(candidates)
    vlm_detected = sum(1 for c in candidates
                       if c.get("matched_regions", [{}])[0].get("source") == "vlm"
                       or c.get("visible_region", {}).get("source") == "vlm")
    vlm_labels = sorted(set(
        c.get("label", "") for c in candidates
        if c.get("matched_regions", [{}])[0].get("source") == "vlm"
        or c.get("visible_region", {}).get("source") == "vlm"
    ))

    small_labels = {"TH", "T", "ID", "NTC", "N"}
    detected_small = small_labels & set(vlm_labels)
    missing_small = small_labels - set(vlm_labels)

    threshold = 4
    if n < threshold:
        logger.warning(
            "VLM returned only %d candidate(s) on %s side (expected >= %d). "
            "Detected labels: %s. Missing small pads will be estimated geometrically "
            "— positions may be inaccurate.",
            n, side, threshold, vlm_labels or ["(none)"]
        )
    elif missing_small:
        logger.info(
            "VLM on %s side: detected %d candidates (%s), but missed small pads: %s. "
            "These will be geometrically estimated.",
            side, n, vlm_labels, sorted(missing_small)
        )
