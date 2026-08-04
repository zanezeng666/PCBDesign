"""VLM-based pad and component detection using Aliyun DashScope qwen3.7-plus.

Pure VLM detection pipeline — no OCR or heuristic fallback.

Architecture:
    rectified_png (bytes) + width_mm/height_mm/side
        │
        ▼
    _vlm_detect_raw() ──► raw JSON from qwen3.7-plus
        │
        ▼
    _parse_vlm_response() ──► standardized candidate list

    Component detection:
    rectified_png (原图) + width_mm/height_mm/side
        │
        ▼
    _vlm_detect_components_raw() ──► raw JSON from qwen3.7-plus
        │
        ▼
    _parse_components_response() ──► component list with silkscreen/package
"""

from __future__ import annotations

import base64
import json
import logging
import math
import os
import re
import time
import threading

from .logger import get_logger

_log = get_logger(__name__)

try:
    import dashscope
    from dashscope import MultiModalConversation
except ImportError:
    MultiModalConversation = None
    dashscope = None


# ── constants ────────────────────────────────────────────────────────

MODEL_NAME = os.getenv("VLM_MODEL_NAME", "qwen3.7-plus")
TEMPERATURE = 0.05
MAX_TOKENS = 2048
ENABLE_THINKING = False  # Disable thinking mode for deterministic pad detection

# Rate limiting: enforce minimum interval between VLM API calls to avoid 429
_VLM_CALL_LOCK = threading.Lock()
_VLM_LAST_CALL_TS = 0.0
_VLM_MIN_INTERVAL = float(os.getenv("VLM_MIN_INTERVAL", "0.8"))  # seconds
_VLM_MAX_RETRIES = int(os.getenv("VLM_MAX_RETRIES", "4"))
_VLM_RETRY_BASE_DELAY = float(os.getenv("VLM_RETRY_BASE_DELAY", "2.0"))  # seconds

# Tolerance (in mm) for P+/P- pads sharing the same x-column during symmetry retry.
# Pads within this horizontal distance are considered in the same column.
_SYMMETRY_COLUMN_TOLERANCE_MM = 2.0

# Labels we care about
TARGET_LABELS = {
    "B+", "B-", "P+", "P-", "C+", "C-",
    "NTC", "TH", "ID", "N",
}


def _vlm_call_with_retry(
    model: str,
    messages: list,
    temperature: float,
    max_tokens: int,
    enable_thinking: bool = False,
    max_retries: int = _VLM_MAX_RETRIES,
    base_delay: float = _VLM_RETRY_BASE_DELAY,
) -> object:
    """Call MultiModalConversation.call with rate limiting and 429 retry.

    - Enforces a minimum interval between calls (_VLM_MIN_INTERVAL) to avoid 429.
    - On 429 (rate limit), retries with exponential backoff up to max_retries times.
    - On other errors, retries once with a short delay.
    """
    last_exc = None

    for attempt in range(max_retries + 1):
        # ── Rate limit: wait if needed ──
        with _VLM_CALL_LOCK:
            global _VLM_LAST_CALL_TS
            now = time.monotonic()
            wait = _VLM_LAST_CALL_TS + _VLM_MIN_INTERVAL - now
            if wait > 0:
                pass  # will sleep outside the lock
            _VLM_LAST_CALL_TS = now + max(wait, 0)

        if wait > 0:
            _log.debug("VLM rate limit: waiting %.1fs before next call", wait)
            time.sleep(wait)

        try:
            response = MultiModalConversation.call(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                enable_thinking=enable_thinking,
            )

            if response.status_code == 429:
                delay = base_delay * (2 ** attempt)
                _log.warning(
                    "VLM 429 rate limit (attempt %d/%d), retrying in %.1fs...",
                    attempt + 1, max_retries + 1, delay,
                )
                time.sleep(delay)
                last_exc = RuntimeError(f"HTTP 429 rate limit (attempt {attempt + 1})")
                continue

            if response.status_code != 200:
                _log.error(
                    "VLM API error: status=%s, code=%s, message=%s",
                    response.status_code, getattr(response, "code", "?"),
                    getattr(response, "message", "?"),
                )
                if attempt < max_retries:
                    time.sleep(base_delay)
                    last_exc = RuntimeError(f"HTTP {response.status_code}")
                    continue
                return None

            return response

        except Exception as exc:
            last_exc = exc
            if attempt < max_retries:
                delay = base_delay * (2 ** attempt)
                _log.warning(
                    "VLM call exception (attempt %d/%d): %s, retrying in %.1fs...",
                    attempt + 1, max_retries + 1, exc, delay,
                )
                time.sleep(delay)
            else:
                _log.error("VLM call failed after %d retries: %s", max_retries + 1, exc)
                return None

    _log.error("VLM call failed after all retries: %s", last_exc)
    return None

# Label contract: roles + polarity for each terminal type
LABEL_CONTRACT: dict[str, tuple[set[str], str | None]] = {
    "B+":  ({"battery"}, "positive"),
    "B-":  ({"battery"}, "negative"),
    "P+":  ({"charge", "discharge"}, "positive"),
    "P-":  ({"charge", "discharge"}, "negative"),
    "C+":  ({"charge"}, "positive"),
    "C-":  ({"charge"}, "negative"),
    "NTC": ({"temperature"}, None),
    "TH":  ({"temperature"}, None),
    "N":   ({"temperature"}, None),
    "ID":  ({"identification"}, None),
}


# ── public API ───────────────────────────────────────────────────────

def detect_with_vlm(
    rectified_png: bytes,
    width_mm: float,
    height_mm: float,
    side: str,
    is_transparent: bool = False,
    img_w_px: int = 0,
    img_h_px: int = 0,
) -> dict:
    """Detect terminal candidates using qwen3.7-plus VLM.

    Pipeline:
      1. VLM identifies pads and returns approximate polygon coordinates

    Args:
        img_w_px / img_h_px: actual pixel dimensions of the image being sent.
            If 0, estimated from width_mm * 50.

    Raises DesignError if VLM is unavailable or fails.
    """
    if side not in {"front", "back"}:
        from .errors import DesignError
        raise DesignError("INVALID_BOARD_SIDE", "side must be front or back")

    _check_vlm_available()

    raw = _vlm_detect_raw(rectified_png, width_mm, height_mm, is_transparent,
                          img_w_px=img_w_px, img_h_px=img_h_px)
    if raw is None:
        from .errors import DesignError
        raise DesignError("VLM_DETECTION_FAILED", "qwen3.7-plus returned no usable response")

    candidates = _parse_vlm_response(raw, width_mm, height_mm, side)
    _log.info("VLM returned %d candidates for side %r", len(candidates), side)

    # ── Debug snapshot helper ──
    _debug_stages: list[dict] = []

    def _snapshot(label: str, cands: list[dict]):
        _debug_stages.append({
            "stage": label,
            "count": len(cands),
            "candidates": [_pad_for_debug(c) for c in cands],
        })

    def _pad_for_debug(pad: dict) -> dict:
        """Extract key fields from a pad for debug output."""
        vp = pad.get("visible_position", {})
        vr = pad.get("visible_region", {})
        mr = (pad.get("matched_regions") or [{}])[0]
        return {
            "label": pad.get("label", ""),
            "x_mm": vp.get("x_mm"),
            "y_mm": vp.get("y_mm"),
            "width_mm": pad.get("width_mm"),
            "height_mm": pad.get("height_mm"),
            "confidence": pad.get("confidence"),
            "source": vr.get("source", mr.get("source", "vlm")),
            "diagnostic_verified": pad.get("diagnostic_verified", ""),
        }

    _snapshot("step1_vlm_raw", candidates)

    # ── Symmetry check: if P+/P- counts are asymmetric in the same column, retry ──
    pre_sym_n = len(candidates)
    candidates = _symmetry_fill_missing(
        rectified_png, width_mm, height_mm, candidates,
        img_w_px=img_w_px, img_h_px=img_h_px,
    )
    _log.info("After symmetry_fill: %d → %d candidates", pre_sym_n, len(candidates))
    _snapshot("step2_after_symmetry", candidates)

    # ── Safety net: remove P+/P- outliers not in the main x-column ──
    pre_filter_n = len(candidates)
    candidates = _filter_pp_pn_column_outliers(candidates, width_mm, height_mm)
    _log.info("After column_outlier_filter: %d → %d candidates", pre_filter_n, len(candidates))
    _snapshot("step3_after_outlier_filter", candidates)

    # ── Final geometric inference: fill any remaining P+/P- asymmetry ──
    _log.info("Calling geometric_fill_missing with %d candidates", len(candidates))
    candidates = _geometric_fill_missing(candidates, width_mm, height_mm)
    _log.info("After geometric_fill_missing: %d candidates", len(candidates))
    _snapshot("step4_after_geometric", candidates)

    return {
        "side": side,
        "width_mm": width_mm,
        "height_mm": height_mm,
        "candidate_count": len(candidates),
        "candidates": candidates,
        "annotated_png_base64": "",  # frontend draws its own overlay
        "method": "vlm+qwen3.7-plus",
        "notice": "VLM 视觉识别结果；请逐项人工确认。",
        "_debug_stages": _debug_stages,
    }


# ── Post-detection symmetry check ───────────────────────────────────

_SYMMETRY_MIN_RATIO = 0.5  # min ratio for retry (at least 50% of major count found)


def _compute_column_tolerance(width_mm: float, height_mm: float) -> float:
    """Column tolerance proportional to PCB dimensions.
    
    Pads in the same column (same P+/P- side) typically share a similar
    x-coordinate. The tolerance is proportional to the smaller PCB dimension
    (usually ~6%), with a 1.0mm minimum for very small PCBs.
    """
    return max(min(width_mm, height_mm) * 0.06, 1.0)

def _symmetry_fill_missing(
    png_bytes: bytes,
    width_mm: float,
    height_mm: float,
    candidates: list[dict],
    img_w_px: int = 0,
    img_h_px: int = 0,
) -> list[dict]:
    """Post-detection symmetry check for P+/P- pad counts.

    If the VLM detected more P+ than P- (or vice versa) and they appear
    to share the same vertical column, run a *targeted* retry focused
    on the region where the missing pads most likely reside.
    """
    p_plus_all = [c for c in candidates if c["label"] == "P+"]
    p_minus_all = [c for c in candidates if c["label"] == "P-"]

    # ── Column clustering: filter out isolated false positives ──
    # P+/P- pads should cluster at specific x-coordinates on the board edges.
    # Isolated pads at completely different x positions (VLM noise) are removed
    # before symmetry analysis.
    col_tol = _compute_column_tolerance(width_mm, height_mm)
    def _cluster_by_x(pads: list[dict], eps: float = None) -> list[list[dict]]:
        if eps is None:
            eps = col_tol
        """DBSCAN-like 1D clustering by x-coordinate."""
        if not pads:
            return []
        sorted_pads = sorted(pads, key=lambda c: c["visible_position"]["x_mm"])
        clusters = []
        current = [sorted_pads[0]]
        for p in sorted_pads[1:]:
            if p["visible_position"]["x_mm"] - current[-1]["visible_position"]["x_mm"] <= eps:
                current.append(p)
            else:
                clusters.append(current)
                current = [p]
        clusters.append(current)
        return clusters

    p_plus_clusters = _cluster_by_x(p_plus_all)
    p_minus_clusters = _cluster_by_x(p_minus_all)

    # Keep only the largest cluster for each label
    p_plus = max(p_plus_clusters, key=len) if p_plus_clusters else []
    p_minus = max(p_minus_clusters, key=len) if p_minus_clusters else []

    # Log if we filtered any outliers
    n_plus_removed = len(p_plus_all) - len(p_plus)
    n_minus_removed = len(p_minus_all) - len(p_minus)
    if n_plus_removed or n_minus_removed:
        _log.info(
            "Column filter: removed %d P+ and %d P- outliers (not in main cluster)",
            n_plus_removed, n_minus_removed,
        )
        # Remove outliers from candidates list by position+label match
        out_keys = set()
        for c in (p_plus_all + p_minus_all):
            if c not in p_plus and c not in p_minus:
                out_keys.add((c["label"], round(c["visible_position"]["x_mm"], 2), round(c["visible_position"]["y_mm"], 2)))
        candidates = [
            c for c in candidates
            if (c["label"], round(c["visible_position"]["x_mm"], 2), round(c["visible_position"]["y_mm"], 2)) not in out_keys
        ]

    n_plus, n_minus = len(p_plus), len(p_minus)

    # Quick exit: both present and symmetric, or both absent
    if n_plus == n_minus:
        return candidates
    if n_plus == 0 or n_minus == 0:
        return candidates

    missing_label = "P-" if n_plus > n_minus else "P+"
    n_expected = max(n_plus, n_minus)
    n_found = min(n_plus, n_minus)

    _log.info(
        "Symmetry check: P+=%d, P-=%d (expected symmetric, missing %d %s)",
        n_plus, n_minus, n_expected - n_found, missing_label,
    )

    # ── Verify they share the same x column ──
    all_pads = p_plus + p_minus
    xs = [c["visible_position"]["x_mm"] for c in all_pads]
    x_spread = max(xs) - min(xs)
    if x_spread > col_tol:
        _log.info("Symmetry skip: x-spread=%.1fmm > tolerance, not same column", x_spread)
        return candidates

    # ── Build region description for retry ──
    avg_x = sum(xs) / len(xs)
    found_ys = sorted(c["visible_position"]["y_mm"] for c in (p_plus + p_minus))
    y_min, y_max = found_ys[0], found_ys[-1]

    # Infer expected y-range: extend beyond last found pad
    # If we have N found + M missing, the total span = (N+M-1) * avg_gap
    if len(found_ys) >= 2:
        avg_gap = (found_ys[-1] - found_ys[0]) / (len(found_ys) - 1)
    else:
        avg_gap = 2.5  # typical pad pitch

    # Extend search region: look beyond the last found pad
    total_count = n_expected + n_found  # total pads in column
    expected_span = (total_count - 1) * avg_gap

    if n_plus > n_minus:
        # P- pads are below P+ → extend downward
        search_y_min = y_max + avg_gap * 0.5
        search_y_max = max(height_mm, y_min + expected_span * 1.2)
    else:
        # P+ pads are above P- → extend upward
        search_y_min = max(0, y_max - expected_span * 1.2)
        search_y_max = y_min - avg_gap * 0.5

    search_y_min = max(0, min(search_y_min, height_mm))
    search_y_max = max(0, min(search_y_max, height_mm))

    _log.info(
        "Symmetry retry: searching for %s pads near x=%.1fmm, y ∈ [%.1f, %.1f]mm",
        missing_label, avg_x, search_y_min, search_y_max,
    )

    # ── Build targeted retry prompt ──
    # CRITICAL: All coordinates must be in the SAME system — fractional 0.0–1.0.
    # Mixing mm and fractional causes VLM coordinate confusion and hallucination.
    px_w = img_w_px if img_w_px > 0 else int(width_mm * 50)
    px_h = img_h_px if img_h_px > 0 else int(height_mm * 50)
    avg_x_frac = avg_x / width_mm if width_mm > 0 else 0.5
    search_y_min_frac = search_y_min / height_mm if height_mm > 0 else 0.0
    search_y_max_frac = search_y_max / height_mm if height_mm > 0 else 1.0
    col_tol_frac = _SYMMETRY_COLUMN_TOLERANCE_MM / width_mm if width_mm > 0 else 0.05

    retry_prompt = f"""You previously identified {n_plus} P+ and {n_minus} P- pads on this PCB.
The P+ and P- pads are arranged in a single vertical column along the same edge.

There should be {n_expected} P- and {n_expected} P+ pads (equal count).
You may have missed {n_expected - n_found} {missing_label} pad(s).

Look ONLY along the vertical strip near x_frac ≈ {avg_x_frac:.4f} (±{col_tol_frac:.4f}).
Scan the y_frac range [{search_y_min_frac:.4f}, {search_y_max_frac:.4f}].

The pads are uniformly spaced. Look for a metallic rectangular pad
with "{missing_label}" silkscreen label nearby. The pad appearance
is identical to the {missing_label} pads you already found — same size,
same shape, same text.

Return ONLY a JSON array of pad(s) found in this region.
Each pad: label, polygon (x_frac/y_frac), confidence.
If you genuinely cannot find any {missing_label} pad in this region, return [].
DO NOT re-list pads you have already identified."""

    image_b64 = base64.b64encode(png_bytes).decode("ascii")
    image_url = f"data:image/png;base64,{image_b64}"

    retry_messages = [{
        "role": "user",
        "content": [
            {"image": image_url},
            {"text": retry_prompt},
        ],
    }]

    # ── Call VLM for retry ──
    try:
        dashscope.api_key = _get_api_key()
        _log.info("Symmetry retry: calling VLM with focused prompt")
        response = _vlm_call_with_retry(
            model=MODEL_NAME,
            messages=retry_messages,
            temperature=0.05,  # lower temp for more focused search
            max_tokens=2048,
            enable_thinking=ENABLE_THINKING,
        )

        if response is None or response.status_code != 200:
            _log.warning("Symmetry retry API error: code=%s, skipping",
                         getattr(response, 'code', 'none') if response else 'none')
            return candidates

        contents = response.output.choices[0].message.content
        raw_text = "".join(p.get("text", "") for p in contents if isinstance(p, dict))
        if not raw_text.strip():
            _log.info("Symmetry retry: VLM returned empty, no additional pads found")
            return candidates

        _log.debug("Symmetry retry raw (first 300 chars): %s", raw_text[:300])

        parsed = _extract_json(raw_text)
        if not parsed or not parsed.get("items"):
            _log.info("Symmetry retry: no valid JSON items")
            return candidates

        # Parse retry results using the same pipeline
        retry_candidates = _parse_vlm_response(parsed, width_mm, height_mm, "front")
        retry_candidates = [c for c in retry_candidates if c["label"] == missing_label]

        _log.info("Symmetry retry: found %d additional %s pad(s)", len(retry_candidates), missing_label)

        if not retry_candidates:
            return candidates

        # ── Merge: append retry candidates, then re-dedup ──
        merged = list(candidates) + retry_candidates
        merged = _spatial_dedup_candidates(merged)
        merged = _assign_unique_ids(merged)

        merged = sorted(merged, key=lambda c: (
            c["visible_position"]["x_mm"], c["visible_position"]["y_mm"], c["label"],
        ))

        # ── Level-2 check: still asymmetric → re-verify all labels ──
        p_plus2 = [c for c in merged if c["label"] == "P+"]
        p_minus2 = [c for c in merged if c["label"] == "P-"]
        n2_plus, n2_minus = len(p_plus2), len(p_minus2)

        if n2_plus == n2_minus or n2_plus == 0 or n2_minus == 0:
            return merged

        # Still asymmetric — one of the P+ or P- might be mislabeled.
        # Ask VLM to re-verify ALL pads in the column.
        _log.info(
            "Symmetry L2: still asymmetric (P+=%d, P-=%d), running label re-verification",
            n2_plus, n2_minus,
        )

        all_col_ys = sorted(c["visible_position"]["y_mm"] for c in p_plus2 + p_minus2)
        pad_list = "\n".join(
            f"  Pad at y={y:.2f} mm — originally labeled \"{c['label']}\""
            for c, y in sorted(
                [(c, c["visible_position"]["y_mm"]) for c in p_plus2 + p_minus2],
                key=lambda t: t[1],
            )
        )

        recheck_prompt = f"""CRITICAL LABEL VERIFICATION: I need you to re-examine the P+/P- pads
in the vertical column at x ≈ {avg_x:.1f} mm. Currently detected:

{pad_list}

IMPORTANT: There should be EQUAL numbers of P+ and P- pads in this column.
But we have {n2_plus} P+ and {n2_minus} P-. This means at least one pad
was assigned the wrong label.

Look VERY carefully at the silkscreen labels next to each pad:
  - "P+" means a positive charge terminal
  - "P-" means a negative charge terminal
  - The label text may be small or partially visible

Return a JSON array with the CORRECTED labels for ALL {n2_plus + n2_minus} pads
in this column. For each pad, provide label, x_frac, y_frac, and confidence.
If a label is already correct, keep it as-is.
If a label is wrong, output the CORRECTED label.
Only include pads in the column at x ≈ {avg_x:.1f} mm."""

        recheck_messages = [{
            "role": "user",
            "content": [
                {"image": image_url},
                {"text": recheck_prompt},
            ],
        }]

        recheck_raw = _vlm_call_with_retry(
            model=MODEL_NAME,
            messages=recheck_messages,
            temperature=0.03,
            max_tokens=2048,
            enable_thinking=ENABLE_THINKING,
        )
        if recheck_raw and recheck_raw.status_code == 200:
            recheck_text = "".join(
                p.get("text", "") for p in recheck_raw.output.choices[0].message.content
                if isinstance(p, dict)
            )
            recheck_parsed = _extract_json(recheck_text)
            if recheck_parsed and recheck_parsed.get("items"):
                recheck_cands = _parse_vlm_response(recheck_parsed, width_mm, height_mm, "front")
                # Only keep P+/P- from the recheck, replace originals
                kept = [c for c in merged if c["label"] not in ("P+", "P-")]
                replaced = [c for c in recheck_cands if c["label"] in ("P+", "P-")]
                merged = kept + replaced
                merged = _spatial_dedup_candidates(merged)
                merged = _assign_unique_ids(merged)
                merged = sorted(merged, key=lambda c: (
                    c["visible_position"]["x_mm"], c["visible_position"]["y_mm"], c["label"],
                ))
                p_final = [c for c in merged if c["label"] in ("P+", "P-")]
                _log.info(
                    "Symmetry L2 result: %d total P+/P- pads (%d P+, %d P-)",
                    len(p_final),
                    len([c for c in merged if c["label"] == "P+"]),
                    len([c for c in merged if c["label"] == "P-"]),
                )
            else:
                _log.info("Symmetry L2: VLM returned no usable recheck data")
        else:
            _log.warning("Symmetry L2: API error code=%s",
                         getattr(recheck_raw, 'code', 'none') if recheck_raw else 'none')

        # ── L3: Geometric inference — extrapolate missing pads from pattern ──
        p_plus3 = [c for c in merged if c["label"] == "P+"]
        p_minus3 = [c for c in merged if c["label"] == "P-"]
        n3_plus, n3_minus = len(p_plus3), len(p_minus3)

        if n3_plus != n3_minus and n3_plus > 0 and n3_minus > 0:
            _log.info("Symmetry L3: geometric inference (P+=%d, P-=%d)", n3_plus, n3_minus)
            merged = _geometric_fill_missing(merged, width_mm, height_mm)

        # ── Final cleanup: column filter again + remove overlaps ──
        merged = _filter_pp_pn_column_outliers(merged, width_mm, height_mm)
        merged = _resolve_pp_pn_overlaps(merged)

        final_p = [c for c in merged if c["label"] in ("P+", "P-")]
        _log.info(
            "Symmetry final: %d P+/P- pads (%d P+, %d P-)",
            len(final_p), len([c for c in final_p if c["label"] == "P+"]), len([c for c in final_p if c["label"] == "P-"]),
        )

        return merged

    except Exception as exc:
        _log.warning("Symmetry retry failed with exception: %s, falling back", exc)
        return candidates


def _geometric_fill_missing(
    candidates: list[dict],
    width_mm: float,
    height_mm: float,
) -> list[dict]:
    """Geometric inference: extrapolate missing P+ or P- pads from column pattern.

    When the VLM cannot find all pads (e.g., silkscreen too faint at edges),
    we use the uniform-spacing pattern of the known pads to predict the
    missing ones. The inferred pad gets a lower confidence and a flag
    marking it as geometrically inferred.
    """
    p_plus = sorted(
        [c for c in candidates if c["label"] == "P+"],
        key=lambda c: c["visible_position"]["y_mm"],
    )
    p_minus = sorted(
        [c for c in candidates if c["label"] == "P-"],
        key=lambda c: c["visible_position"]["y_mm"],
    )

    col_tol = _compute_column_tolerance(width_mm, height_mm)
    n_plus, n_minus = len(p_plus), len(p_minus)
    _log.info(
        "Geometric fill: entry n_plus=%d n_minus=%d total_candidates=%d width=%.2f height=%.2f col_tol=%.2f",
        n_plus, n_minus, len(candidates), width_mm, height_mm, col_tol,
    )
    if n_plus == n_minus:
        _log.info("Geometric fill: n_plus==n_minus, nothing to do")
        return candidates

    major_label, major_pads = ("P+", p_plus) if n_plus > n_minus else ("P-", p_minus)
    minor_label = "P-" if major_label == "P+" else "P+"
    minor_pads = p_minus if major_label == "P+" else p_plus

    # Check same column: all pads share approximately same x
    all_col = major_pads + minor_pads
    xs = [c["visible_position"]["x_mm"] for c in all_col]
    x_min, x_max = min(xs), max(xs)
    x_spread = x_max - x_min
    _log.info(
        "Geometric fill: x-coords=%s, spread=%.2f, tolerance=%.2f",
        [round(x, 2) for x in xs], x_spread, col_tol,
    )
    if x_spread > col_tol:
        _log.warning(
            "Geometric fill: x-spread %.2f > tolerance %.2f, skipping (P+ and P- not in same column)",
            x_spread, col_tol,
        )
        return candidates

    avg_x = sum(xs) / len(xs)
    all_ys = sorted(c["visible_position"]["y_mm"] for c in all_col)
    _log.info("Geometric fill: all_ys=%s", [round(y, 2) for y in all_ys])

    # Compute avg gap from all known pads
    if len(all_ys) < 2:
        _log.info("Geometric fill: only 1 pad total, cannot compute gap")
        return candidates
    gaps = [all_ys[i + 1] - all_ys[i] for i in range(len(all_ys) - 1)]
    avg_gap = sum(gaps) / len(gaps)
    _log.info("Geometric fill: gaps=%s avg_gap=%.2f", [round(g, 2) for g in gaps], avg_gap)

    n_missing = len(major_pads) - len(minor_pads)
    _log.info("Geometric fill: n_missing=%d for label=%s", n_missing, minor_label)

    # Determine where missing pads should be
    major_ys = sorted(c["visible_position"]["y_mm"] for c in major_pads)
    minor_ys = sorted(c["visible_position"]["y_mm"] for c in minor_pads)
    _log.info("Geometric fill: major_ys=%s minor_ys=%s",
              [round(y, 2) for y in major_ys], [round(y, 2) for y in minor_ys])

    # Compute spacing from the minor-label pads (or major if minor only has 1)
    if len(minor_ys) >= 2:
        minor_gaps = [minor_ys[i + 1] - minor_ys[i] for i in range(len(minor_ys) - 1)]
        step = sum(minor_gaps) / len(minor_gaps)
        ref_ys = minor_ys
        _log.info("Geometric fill: using minor gaps=%s step=%.2f", [round(g, 2) for g in minor_gaps], step)
    else:
        if len(major_ys) >= 2:
            major_gaps = [major_ys[i + 1] - major_ys[i] for i in range(len(major_ys) - 1)]
            step = sum(major_gaps) / len(major_gaps)
            _log.info("Geometric fill: minor has <2 pads, using major gaps=%s step=%.2f",
                      [round(g, 2) for g in major_gaps], step)
        else:
            step = avg_gap
            _log.info("Geometric fill: both major and minor have <2 pads, using avg_gap=%.2f", step)
        ref_ys = minor_ys

    # Direction: if minor pads are above major, extend upward; else downward
    if minor_ys and major_ys:
        minor_center = sum(minor_ys) / len(minor_ys)
        major_center = sum(major_ys) / len(major_ys)
        extend_up = minor_center < major_center  # minor is above → extend further up
        _log.info("Geometric fill: minor_center=%.2f major_center=%.2f extend_up=%s",
                  minor_center, major_center, extend_up)
    else:
        extend_up = False

    last_minor_y = max(minor_ys) if minor_ys else max(major_ys)
    first_minor_y = min(minor_ys) if minor_ys else min(major_ys)
    _log.info("Geometric fill: last_minor_y=%.2f first_minor_y=%.2f", last_minor_y, first_minor_y)

    # Compute average pad dimensions from existing pads for inferred ones
    # Prefer same-label pads, fall back to major-label pads
    minor_all = minor_pads
    avg_w, avg_h = width_mm * 0.04, height_mm * 0.03  # proportional to PCB size
    for source_pads in [minor_pads, major_pads]:
        widths = [p.get("width_mm") for p in source_pads if p.get("width_mm")]
        heights = [p.get("height_mm") for p in source_pads if p.get("height_mm")]
        if widths and heights:
            avg_w = sum(widths) / len(widths)
            avg_h = sum(heights) / len(heights)
            break

    _log.info(
        "Geometric inference: filling %d missing %s pad(s), step=%.2fmm, x=%.2f, extend_up=%s, size=%.2fx%.2f",
        n_missing, minor_label, step, avg_x, extend_up, avg_w, avg_h,
    )

    inferred = []
    for i in range(n_missing):
        if extend_up:
            predicted_y = first_minor_y - step * (i + 1)
        else:
            predicted_y = last_minor_y + step * (i + 1)
        _log.info("Geometric fill: candidate %d predicted_y=%.2f board=[0, %.2f]",
                  i, predicted_y, height_mm)

        # Tight margin: geometric inference should not place pads far outside.
        # A margin of 3mm or 5% of PCB height, whichever is smaller, prevents
        # pads from being generated that will later become degenerate after
        # _clamp_pads_to_board clips them to the board edge.
        MARGIN_MM = min(height_mm * 0.05, 3.0)
        if predicted_y < -MARGIN_MM or predicted_y > height_mm + MARGIN_MM:
            _log.info("Geometric inference: predicted y=%.2f far outside PCB [0, %.2f] margin=%.1f, skipping",
                      predicted_y, height_mm, MARGIN_MM)
            continue

        # Create synthetic candidate with geometric inference flag.
        # Build a synthetic rectangle polygon so downstream alignment
        # (_align_pad_groups) can read polygon dimensions correctly.
        hw = avg_w / 2
        hh = avg_h / 2
        synthetic_poly = [
            {"x_mm": round(avg_x - hw, 3), "y_mm": round(predicted_y - hh, 3)},
            {"x_mm": round(avg_x + hw, 3), "y_mm": round(predicted_y - hh, 3)},
            {"x_mm": round(avg_x + hw, 3), "y_mm": round(predicted_y + hh, 3)},
            {"x_mm": round(avg_x - hw, 3), "y_mm": round(predicted_y + hh, 3)},
        ]
        synthetic_bbox = {
            "x_mm": round(avg_x - hw, 3),
            "y_mm": round(predicted_y - hh, 3),
            "width_mm": round(avg_w, 3),
            "height_mm": round(avg_h, 3),
        }
        inferred_pad = {
            "label": minor_label,
            "visible_position": {
                "x_mm": round(avg_x, 3),
                "y_mm": round(predicted_y, 3),
            },
            "matched_regions": [{
                "center": {"x_mm": round(avg_x, 3), "y_mm": round(predicted_y, 3)},
                "polygon": synthetic_poly,
                "bbox": synthetic_bbox,
                "source": "geometric",
            }],
            "visible_region": {
                "center": {"x_mm": round(avg_x, 3), "y_mm": round(predicted_y, 3)},
                "polygon": synthetic_poly,
                "bbox": synthetic_bbox,
                "source": "geometric",
                "confidence": 0.4,
            },
            "diagnostic_verified": "geometric_inference",
            "confidence": 0.4,
            "width_mm": round(avg_w, 3),
            "height_mm": round(avg_h, 3),
        }
        inferred.append(inferred_pad)
        _log.info("  Inferred %s at (%.2f, %.2f)", minor_label, avg_x, predicted_y)

    result = list(candidates) + inferred
    _log.info("Geometric fill: before dedup=%d after inferred=%d, result=%d",
              len(candidates), len(inferred), len(result))
    result = _spatial_dedup_candidates(result)
    result = _assign_unique_ids(result)
    _log.info("Geometric fill: after dedup+assign=%d candidates", len(result))
    return sorted(result, key=lambda c: (
        c["visible_position"]["x_mm"], c["visible_position"]["y_mm"], c["label"],
    ))


def _filter_pp_pn_column_outliers(candidates: list[dict],
                                    width_mm: float = 50.0,
                                    height_mm: float = 30.0) -> list[dict]:
    """Remove P+/P- pads not belonging to the main column cluster by x-coordinate."""
    pp_all = [c for c in candidates if c["label"] == "P+"]
    pm_all = [c for c in candidates if c["label"] == "P-"]
    all_polarity = pp_all + pm_all
    if len(all_polarity) <= 1:
        return candidates

    col_tol = _compute_column_tolerance(width_mm, height_mm)

    # Simple 1D clustering by x
    sorted_pads = sorted(all_polarity, key=lambda c: c["visible_position"]["x_mm"])
    clusters = [[sorted_pads[0]]]
    for p in sorted_pads[1:]:
        if p["visible_position"]["x_mm"] - clusters[-1][-1]["visible_position"]["x_mm"] <= col_tol:
            clusters[-1].append(p)
        else:
            clusters.append([p])

    # Keep only pads in the largest cluster (match by position+label, not id)
    main_cluster = max(clusters, key=len)
    out_keys = set()
    for c in all_polarity:
        if c not in main_cluster:
            key = (c["label"], round(c["visible_position"]["x_mm"], 2), round(c["visible_position"]["y_mm"], 2))
            out_keys.add(key)

    if out_keys:
        candidates = [
            c for c in candidates
            if (c["label"], round(c["visible_position"]["x_mm"], 2), round(c["visible_position"]["y_mm"], 2)) not in out_keys
        ]
        _log.info("Column cleanup: removed %d outlier(s) not in main cluster", len(out_keys))

    return candidates


def _resolve_pp_pn_overlaps(candidates: list[dict]) -> list[dict]:
    """Remove P+/P- pairs that overlap at the same physical position.

    When the VLM assigns P+ and P- labels to the same pad location,
    we resolve the conflict by keeping the label that is more consistent
    with the majority count in its column.
    """
    p_plus = [c for c in candidates if c["label"] == "P+"]
    p_minus = [c for c in candidates if c["label"] == "P-"]
    if len(p_plus) <= 1 or len(p_minus) <= 1:
        return candidates

    overlaps = []  # pairs of (p_plus_idx, p_minus_idx)
    for i, pp in enumerate(p_plus):
        px, py = pp["visible_position"]["x_mm"], pp["visible_position"]["y_mm"]
        for j, pm in enumerate(p_minus):
            mx, my = pm["visible_position"]["x_mm"], pm["visible_position"]["y_mm"]
            dist = ((px - mx) ** 2 + (py - my) ** 2) ** 0.5
            if dist < 1.0:  # same position within 1mm
                overlaps.append((i, j, dist))

    if not overlaps:
        return candidates

    _log.info("Resolving %d P+/P- overlap(s)", len(overlaps))

    # Remove the overlapping pad with fewer total pads in its group
    # If a P- overlaps with a P+ and there are more P+ than P-, keep P+
    non_pp = [c for c in candidates if c["label"] not in ("P+", "P-")]
    kept_pp = list(p_plus)
    kept_pm = list(p_minus)

    for i, j, dist in overlaps:
        if len(kept_pp) >= len(kept_pm):
            # Keep P+, remove P- at index j
            _log.info("  Removing overlapping P- at (%.2f, %.2f) — keeping P+", kept_pm[j]["visible_position"]["x_mm"], kept_pm[j]["visible_position"]["y_mm"])
            kept_pm[j] = None
        else:
            _log.info("  Removing overlapping P+ at (%.2f, %.2f) — keeping P-", kept_pp[i]["visible_position"]["x_mm"], kept_pp[i]["visible_position"]["y_mm"])
            kept_pp[i] = None

    kept_pp = [c for c in kept_pp if c is not None]
    kept_pm = [c for c in kept_pm if c is not None]

    result = non_pp + kept_pp + kept_pm
    result = _spatial_dedup_candidates(result)
    result = _assign_unique_ids(result)
    return sorted(result, key=lambda c: (
        c["visible_position"]["x_mm"], c["visible_position"]["y_mm"], c["label"],
    ))


# ── VLM core ────────────────────────────────────────────────────────

def _get_api_key() -> str:
    """Resolve DASHSCOPE_API_KEY from env (process → user → machine)."""
    key = os.getenv("DASHSCOPE_API_KEY", "").strip()
    if key:
        return key
    # Fallback to Windows registry (system/user env vars)
    try:
        import winreg
        for hive, subkey in (
            (winreg.HKEY_LOCAL_MACHINE, r"SYSTEM\CurrentControlSet\Control\Session Manager\Environment"),
            (winreg.HKEY_CURRENT_USER, r"Environment"),
        ):
            try:
                with winreg.OpenKey(hive, subkey) as reg:
                    key, _ = winreg.QueryValueEx(reg, "DASHSCOPE_API_KEY")
                    if key:
                        return key.strip()
            except OSError:
                continue
    except Exception:
        pass
    return ""


def _check_vlm_available() -> None:
    """Verify DashScope API key and SDK are available. Raises DesignError if not."""
    from .errors import DesignError

    if not _get_api_key():
        raise DesignError("VLM_UNAVAILABLE", "DASHSCOPE_API_KEY not set")
    if MultiModalConversation is None:
        raise DesignError("VLM_UNAVAILABLE", "dashscope SDK not installed")


def _vlm_detect_raw(rectified_png: bytes, width_mm: float, height_mm: float, is_transparent: bool = False,
                    img_w_px: int = 0, img_h_px: int = 0) -> dict | None:
    """Call qwen3.7-plus and return the parsed JSON response, or None on failure."""

    dashscope.api_key = _get_api_key()

    image_b64 = base64.b64encode(rectified_png).decode("ascii")
    image_url = f"data:image/png;base64,{image_b64}"

    prompt = _build_prompt(width_mm, height_mm, is_transparent, img_w_px, img_h_px)

    messages = [{
        "role": "user",
        "content": [
            {"image": image_url},
            {"text": prompt},
        ],
    }]

    _log.info("Calling VLM (model=%s, image_size=%d bytes)", MODEL_NAME, len(rectified_png))

    response = _vlm_call_with_retry(
        model=MODEL_NAME,
        messages=messages,
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS,
        enable_thinking=ENABLE_THINKING,
    )

    if response is None:
        return None

    if response.status_code != 200:
        _log.error("VLM API error: code=%s message=%s",
                   getattr(response, 'code', '?'),
                   getattr(response, 'message', '?'))

    # response.output.choices[0].message.content is a list of dicts
    # Each dict has either "text" or "image" key
    try:
        contents = response.output.choices[0].message.content
    except (AttributeError, IndexError, KeyError) as exc:
        _log.error("Unexpected DashScope response structure: %s", exc)
        return None

    raw_text = ""
    for part in contents:
        if isinstance(part, dict) and "text" in part:
            raw_text += part["text"]

    if not raw_text:
        _log.error("DashScope returned empty text response")
        return None

    _log.debug("VLM raw response (first 500 chars): %s", raw_text[:500])

    # Parse JSON from the response (model may wrap it in markdown code fences)
    parsed = _extract_json(raw_text)
    return parsed


def _build_prompt(width_mm: float, height_mm: float, is_transparent: bool = False,
                  img_w_px: int = 0, img_h_px: int = 0) -> str:
    """Construct the system-style instruction for qwen3.7-plus."""

    # Use actual pixel dimensions if provided, otherwise estimate
    px_w = img_w_px if img_w_px > 0 else int(width_mm * 50)
    px_h = img_h_px if img_h_px > 0 else int(height_mm * 50)

    image_desc = (
        "Clean PCB image on white background. The PCB board FILLS the entire image "
        "edge-to-edge — the image has been cropped to exactly the PCB boundary. "
        "This is a processed image with all background noise removed. "
        "Use the ENTIRE image as your coordinate reference: "
        "(0,0)=top-left corner of image=top-left of PCB, "
        "(1,1)=bottom-right corner of image=bottom-right of PCB."
        if is_transparent else
        "Rectified photo with possible black frame border and white paper background."
    )

    return f"""You are an expert PCB (printed circuit board) inspector. Analyze the uploaded image of a battery protection board.

Image type: {image_desc}
Board physical dimensions: {width_mm:.1f} mm × {height_mm:.1f} mm.
Image dimensions: exactly {px_w} × {px_h} pixels.

CRITICAL DOMAIN KNOWLEDGE — read carefully:
- ALL solder pads on this PCB are ROUNDED RECTANGLES in reality.
- Due to the camera angle (perspective distortion), each pad appears as a SKEWED QUADRILATERAL
  with slightly rounded corners in the image. The 4 edges may not be parallel to image axes.
- Your polygon vertices MUST capture this shape: 4 main corner vertices defining the outer
  boundary of the metallic pad, PLUS (optionally) transition points at rounded corners.

Your task:
1. Locate ALL terminal solder pads on this PCB. Each pad has a silkscreen text label printed NEAR or ON it.
2. For each terminal pad, use the text ONLY to identify which pad it is (its label).
3. Then trace the EXACT outline of the METALLIC CONDUCTIVE AREA (the silver/gold/copper exposed metal).

⚠️ CRITICAL DISTINCTION:
- The silkscreen TEXT (white characters like "P-", "TH") is NOT the pad. Text is just a label.
- The PAD is the exposed METALLIC area (shiny silver/gold/copper) that a probe or wire would contact.
- The text may be printed ON TOP of the metallic area, or ADJACENT to it — either way,
  your polygon must trace the METAL boundary, not the text character boundary.
- If text is printed on the pad, ignore the text and trace the full metallic area underneath.

Terminal labels to look for: B+, B-, P+, P-, C+, C-, NTC, TH, ID, N
- "B+" and "B-" are battery connection terminals (usually larger pads spanning most of the board height).
- "P+" and "P-" are charge/discharge terminals (medium pads typically on one edge).
- "C+" and "C-" are charge-only terminals.
- "NTC" or "TH" or "N" are thermistor terminals (small pads).
- "ID" is an identification/communication terminal.

For EACH terminal you find, provide:
- "label": the terminal label
- "polygon": array of vertex objects tracing the METALLIC PAD OUTLINE. Each vertex:
  {{"x_frac": ..., "y_frac": ...}} — fractional positions relative to image dimensions
  (0.0 = left/top edge, 1.0 = right/bottom edge). Use 4-5 DECIMAL PLACES.
  Points MUST be in CLOCKWISE order forming a closed polygon.

  POLYGON VERTEX GUIDELINES (rounded rectangle under perspective):
  - 4 vertices: the 4 CORNERS of the quadrilateral that bounds the metallic pad.
    This is the RECOMMENDED format. Choose the 4 outer corner points where the
    metallic pad meets the PCB substrate. If the pad has visible rounded corners,
    place the corner vertex at the intersection of the extrapolated straight edges.
  - 6-8 vertices: the 4 corners PLUS 1 transition point per rounded corner.
    Use this ONLY if you can clearly see the rounded corner curves.
  - Do NOT output more than 8 vertices per pad.

  KEY: Even under perspective distortion, trace the ACTUAL metallic boundary you see.
  If the top edge tilts 5 degrees due to camera angle, your vertices must reflect that tilt.

- "confidence": 0.0 to 1.0 (how certain you are, 1.0 = very certain)

CRITICAL PRECISION RULES:
- MEASURE each pad's position INDEPENDENTLY by looking at its actual location in the image.
- DO NOT reuse the same x_frac (or y_frac) across different pads — tiny differences matter.
- Even pads that appear vertically aligned may have slightly different x positions — measure each one.
- Use the FULL image as your coordinate reference (0 to 1 = entire image width/height).
- If you can see the pad's metallic boundary clearly, estimate x_frac to 0.0001 precision.

Return ONLY a JSON array. No markdown, no explanation.

Example — 4-corner quadrilateral format (PREFERRED for rounded rectangles):
[
  {{"label":"B+","polygon":[
    {{"x_frac":0.3123,"y_frac":0.0534}},
    {{"x_frac":0.3845,"y_frac":0.0512}},
    {{"x_frac":0.3867,"y_frac":0.1023}},
    {{"x_frac":0.3101,"y_frac":0.1045}}
  ],"confidence":0.95}},
  {{"label":"B-","polygon":[
    {{"x_frac":0.1124,"y_frac":0.1456}},
    {{"x_frac":0.1803,"y_frac":0.1501}},
    {{"x_frac":0.2045,"y_frac":0.1987}},
    {{"x_frac":0.1856,"y_frac":0.2602}},
    {{"x_frac":0.1778,"y_frac":0.2820}},
    {{"x_frac":0.1098,"y_frac":0.2775}},
    {{"x_frac":0.0987,"y_frac":0.2556}},
    {{"x_frac":0.1002,"y_frac":0.1678}}
  ],"confidence":0.93}}
]

FINAL REMINDERS:
- Pad = METALLIC conductive area only (silver/gold/copper). NOT the text label, NOT surrounding PCB.
- The text label (e.g. "P-", "TH") tells you WHICH pad it is, but your polygon traces the METAL area.
- Pads on opposite edges have very different x_frac (e.g. 0.05 vs 0.95).
- All pads are rounded rectangles in 3D → appear as skewed quadrilaterals in the 2D image.
- 4-vertex output is preferred and sufficient for rounded rectangle pads.
- If uncertain about a pad, set confidence < 0.8 and still provide your best polygon estimate.
- Order results alphabetically by label."""
def _extract_json(text: str) -> dict | None:
    """Robustly extract JSON array from model output (handles markdown fences)."""
    text = text.strip()

    # Try direct parse first
    try:
        value = json.loads(text)
        if isinstance(value, list):
            return {"items": value}
        return value
    except json.JSONDecodeError:
        pass

    # Try stripping markdown fences
    for fence in ("```json", "```"):
        if fence in text:
            text = text.split(fence, 1)[-1]
            text = text.rsplit("```", 1)[0]
            text = text.strip()
            break

    try:
        value = json.loads(text)
        if isinstance(value, list):
            return {"items": value}
        return value
    except json.JSONDecodeError:
        pass

    # Last resort: regex extract JSON array
    match = re.search(r"\[[\s\S]*\]", text)
    if match:
        try:
            return {"items": json.loads(match.group(0))}
        except json.JSONDecodeError:
            pass

    _log.error("Failed to parse JSON from VLM response: %s", text[:300])
    return None


# ── response parsing ────────────────────────────────────────────────

def _parse_vlm_response(
    raw: dict,
    width_mm: float,
    height_mm: float,
    side: str,
) -> list[dict]:
    """Convert raw VLM JSON items into the standard candidate format.

    Supports two VLM response formats:
      - Polygon format (preferred): each item has a "polygon" array of {{x_frac, y_frac}}
        vertices tracing the actual metallic pad outline.
      - BBox format (legacy fallback): x_frac / y_frac / w_frac / h_frac rectangle.

    Fractional coordinates (0.0 to 1.0) relative to image dimensions -> mm.
    """

    items = raw.get("items", [])
    if not isinstance(items, list):
        items = []

    raw_candidates: list[dict] = []

    for item in items:
        if not isinstance(item, dict):
            continue
        label = str(item.get("label", "")).strip().upper()

        # Normalize common VLM output variations
        label = _clean_label(label)
        if label not in TARGET_LABELS:
            continue

        confidence = float(item.get("confidence", 0.7))

        # ── Parse polygon vertices (preferred) or fall back to bbox ──
        polygon_raw = item.get("polygon")
        polygon_mm: list[dict] | None = None
        use_polygon = isinstance(polygon_raw, list) and len(polygon_raw) >= 3

        if use_polygon:
            polygon_mm, poly_valid = _parse_polygon_vertices(polygon_raw, width_mm, height_mm)
            if not poly_valid:
                use_polygon = False
                polygon_mm = None

        if use_polygon and polygon_mm:
            # ── Polygon path: compute center & bbox from vertices ──
            xs = [p["x_mm"] for p in polygon_mm]
            ys = [p["y_mm"] for p in polygon_mm]
            cx = sum(xs) / len(xs)
            cy = sum(ys) / len(ys)
            min_x, max_x = min(xs), max(xs)
            min_y, max_y = min(ys), max(ys)
            pad_w_mm = max_x - min_x
            pad_h_mm = max_y - min_y
            n_vertices = len(polygon_mm)

            # Classify shape from polygon characteristics
            if n_vertices == 4:
                shape = "rect"
            elif n_vertices <= 8:
                ratio = pad_w_mm / max(pad_h_mm, 0.01)
                if 0.8 < ratio < 1.25:
                    shape = "circle"
                elif 1.2 < ratio < 3.5:
                    shape = "rounded_rect"
                else:
                    shape = "oval"
            else:
                shape = "custom"
        else:
            # ── BBox fallback (legacy format) ──
            x_frac = float(item.get("x_frac", -1))
            y_frac = float(item.get("y_frac", -1))
            w_frac = float(item.get("w_frac", 0))
            h_frac = float(item.get("h_frac", 0))

            if x_frac < 0:
                x_frac = float(item.get("x_px", -1)) / max(width_mm * 20, 1)
            if y_frac < 0:
                y_frac = float(item.get("y_px", -1)) / max(height_mm * 20, 1)
            if w_frac <= 0:
                w_frac = float(item.get("w_px", 0)) / max(width_mm * 20, 1)
            if h_frac <= 0:
                h_frac = float(item.get("h_px", 0)) / max(height_mm * 20, 1)

            # Reject items with zero-area bbox (VLM hallucination / empty item)
            # A valid pad MUST have non-zero width AND height in fraction space
            if w_frac <= 0.0 and h_frac <= 0.0:
                _log.info("Parse: rejecting %s at (%.2f,%.2f) — zero-area bbox (w=%.3f h=%.3f)",
                          label, x_frac, y_frac, w_frac, h_frac)
                continue

            # Bounds check
            if x_frac < 0 or x_frac > 1 or y_frac < 0 or y_frac > 1:
                if x_frac < -0.5 or x_frac > 1.5 or y_frac < -0.5 or y_frac > 1.5:
                    continue
                x_frac = max(0, min(1, x_frac))
                y_frac = max(0, min(1, y_frac))

            cx = x_frac * width_mm
            cy = y_frac * height_mm
            pad_w_mm = w_frac * width_mm if w_frac > 0 else width_mm * 0.08
            pad_h_mm = h_frac * height_mm if h_frac > 0 else height_mm * 0.05

            aspect = pad_w_mm / max(pad_h_mm, 0.01)
            shape = "rounded_rect" if 1.4 < aspect < 3.0 else ("rect" if aspect >= 3.0 else "circle")

            bbox_x = round(cx - pad_w_mm / 2, 3)
            bbox_y = round(cy - pad_h_mm / 2, 3)
            bbox_w = round(pad_w_mm, 3)
            bbox_h = round(pad_h_mm, 3)

            polygon_mm = [
                {"x_mm": bbox_x, "y_mm": bbox_y},
                {"x_mm": round(bbox_x + bbox_w, 3), "y_mm": bbox_y},
                {"x_mm": round(bbox_x + bbox_w, 3), "y_mm": round(bbox_y + bbox_h, 3)},
                {"x_mm": bbox_x, "y_mm": round(bbox_y + bbox_h, 3)},
            ]

            min_x, min_y, max_x, max_y = bbox_x, bbox_y, bbox_x + bbox_w, bbox_y + bbox_h

        # ── Build visible_region ──
        visible_region = {
            "type": "solder_pad",
            "visual_class": "metallic",
            "shape": shape,
            "center": {"x_mm": round(cx, 3), "y_mm": round(cy, 3)},
            "bbox": {
                "x_mm": round(min_x, 3),
                "y_mm": round(min_y, 3),
                "width_mm": round(pad_w_mm, 3),
                "height_mm": round(pad_h_mm, 3),
            },
            "polygon": polygon_mm,
            "source": "vlm",
        }

        # ── PCB boundary validation ──
        # Pads must lie within (or very near) the PCB image area.
        # A pad whose bounding box is entirely outside the board is a
        # hallucination; reject it now rather than forward garbage downstream.
        # Uses min_x/min_y/pad_w_mm/pad_h_mm which are defined in both
        # the polygon path and the bbox fallback path.
        BOUNDARY_MARGIN_MM = max(min(width_mm, height_mm) * 0.03, 0.5)
        bbox_right = min_x + pad_w_mm
        bbox_bottom = min_y + pad_h_mm
        if (bbox_right < -BOUNDARY_MARGIN_MM
                or min_x > width_mm + BOUNDARY_MARGIN_MM
                or bbox_bottom < -BOUNDARY_MARGIN_MM
                or min_y > height_mm + BOUNDARY_MARGIN_MM):
            _log.warning(
                "Parse: rejecting %s — bbox (%.1f,%.1f %.1f×%.1f) outside PCB %.1f×%.1f",
                label, min_x, min_y, pad_w_mm, pad_h_mm, width_mm, height_mm,
            )
            continue
        elif (min_x < -BOUNDARY_MARGIN_MM or min_y < -BOUNDARY_MARGIN_MM
              or bbox_right > width_mm + BOUNDARY_MARGIN_MM
              or bbox_bottom > height_mm + BOUNDARY_MARGIN_MM):
            _log.info(
                "Parse: clipping %s bbox (%.1f,%.1f %.1f×%.1f) to PCB %.1f×%.1f",
                label, min_x, min_y, pad_w_mm, pad_h_mm, width_mm, height_mm,
            )
            # Clip polygon vertices to PCB area
            for pt in polygon_mm:
                pt["x_mm"] = max(0.0, min(width_mm, pt["x_mm"]))
                pt["y_mm"] = max(0.0, min(height_mm, pt["y_mm"]))

        roles, polarity = LABEL_CONTRACT.get(label, (set(), None))

        candidate = {
            "id": label.replace("+", "_POS").replace("-", "_NEG"),  # temporary; will be uniquified below
            "label": label,
            "recognized_text": label,
            "visible_position": {"x_mm": round(cx, 3), "y_mm": round(cy, 3)},
            "visible_region": visible_region,
            "matched_regions": [visible_region],
            "text_region": None,
            "roles": sorted(roles),
            "polarity": polarity,
            "side": side,
            "shape": shape,
            "width_mm": round(pad_w_mm, 3),
            "height_mm": round(pad_h_mm, 3),
            "confidence": round(min(0.99, confidence), 3),
            "method": "vlm→qwen3.7-plus",
            "region_resolved": True,
            "match_quality": "auto",
            "match_distance_mm": None,
            "requires_confirmation": True,
        }
        raw_candidates.append(candidate)

    # ── Flag via-like false positives (small circular spots) ──
    raw_candidates = _filter_via_like_candidates(raw_candidates, width_mm, height_mm)

    # ── Spatial deduplication: merge near-identical pads sharing the same label ──
    candidates = _spatial_dedup_candidates(raw_candidates)

    # ── Assign unique sequential IDs per label group (internal use only) ──
    candidates = _assign_unique_ids(candidates)

    return sorted(candidates, key=lambda c: (c["visible_position"]["x_mm"], c["visible_position"]["y_mm"], c["label"]))


def _compute_dedup_threshold(candidates: list[dict]) -> float:
    """Spatial dedup threshold proportional to the candidates' spatial extent."""
    if len(candidates) >= 2:
        xs = [c["visible_position"]["x_mm"] for c in candidates]
        ys = [c["visible_position"]["y_mm"] for c in candidates]
        spread = max(max(xs) - min(xs), max(ys) - min(ys), 0.1)
        return max(spread * 0.03, 0.5)
    return 1.5  # fallback for single candidate


def _spatial_dedup_candidates(raw_candidates: list[dict]) -> list[dict]:
    """Remove near-duplicate candidates that share the same label and have
    overlapping / very-close center positions.

    Within each label group, a pair of candidates whose center-to-center
    distance is below a threshold (proportional to the spatial extent of
    all candidates) is considered the same physical pad; the one with
    higher confidence is kept.
    """
    if len(raw_candidates) <= 1:
        return list(raw_candidates)

    threshold = _compute_dedup_threshold(raw_candidates)

    # ── Group by label ──
    groups: dict[str, list[dict]] = {}
    for c in raw_candidates:
        groups.setdefault(c["label"], []).append(c)

    deduped: list[dict] = []
    for _label, group in groups.items():
        n = len(group)
        if n == 1:
            deduped.append(group[0])
            continue

        # Sort by (area descending, confidence descending).
        # Large-area candidates represent real pads; small-area ones are
        # likely false-positives (e.g. via solder dots).  Preferring area
        # first prevents a tiny high-confidence blob from replacing a
        # genuinely large pad.
        def _area_key(c: dict) -> float:
            bb = (c.get("visible_region") or {}).get("bbox") or {}
            return float((bb.get("width_mm") or 0) * (bb.get("height_mm") or 0))

        group.sort(key=lambda c: (_area_key(c), c["confidence"]), reverse=True)
        kept: list[dict] = []
        # Keep track of already-kept centers for this label
        kept_centers: list[tuple[float, float]] = []

        for c in group:
            cx = c["visible_position"]["x_mm"]
            cy = c["visible_position"]["y_mm"]
            is_dup = False
            for kcx, kcy in kept_centers:
                dist = math.hypot(cx - kcx, cy - kcy)
                if dist <= threshold:
                    is_dup = True
                    _log.info(
                        "Spatial dedup: merging %s at (%.1f,%.1f) into "
                        "existing (%.1f,%.1f), dist=%.2f mm",
                        c["label"], cx, cy, kcx, kcy, dist,
                    )
                    break
            if not is_dup:
                kept.append(c)
                kept_centers.append((cx, cy))

        deduped.extend(kept)

    return deduped


def _assign_unique_ids(candidates: list[dict]) -> list[dict]:
    """Assign unique internal IDs per label group.

    If a label appears only once, the ID stays as-is (e.g. ``P_POS``).
    If a label appears multiple times, each candidate gets a numbered suffix
    (e.g. ``P_POS_1``, ``P_POS_2``). The ``label`` field is **never** modified
    — these suffixes are purely for internal disambiguation.
    """
    # ── Group by label ──
    groups: dict[str, list[dict]] = {}
    for c in candidates:
        groups.setdefault(c["label"], []).append(c)

    result: list[dict] = []
    for _label, group in groups.items():
        if len(group) == 1:
            # Single instance — keep original ID
            result.extend(group)
        else:
            # Multiple instances — append numeric suffix
            for idx, c in enumerate(group, start=1):
                base_id = _label.replace("+", "_POS").replace("-", "_NEG")
                c["id"] = f"{base_id}_{idx}"
                result.append(c)

    return result


def _compute_via_max_area(width_mm: float, height_mm: float) -> float:
    """Via detection area threshold proportional to PCB area.

    A typical via/solder dot is < 0.2% of the total PCB area.
    """
    return max(width_mm * height_mm * 0.002, 0.5)

_VIA_MIN_ASPECT_RATIO = 0.85    # width/height >= this → nearly square/circular


def _filter_via_like_candidates(raw_candidates: list[dict],
                                 width_mm: float = 50.0,
                                 height_mm: float = 30.0) -> list[dict]:
    """Flag (do not remove) candidates whose shape resembles a via/solder dot.

    Through-hole solder joints appear as tiny, nearly circular metallic spots.
    Real terminal pads are larger rounded rectangles.  This filter LOWERS the
    confidence of via-like candidates so they appear with a warning flag in
    the review UI rather than being silently dropped.
    """
    via_max_area = _compute_via_max_area(width_mm, height_mm)
    for c in raw_candidates:
        w = c.get("width_mm", 0)
        h = c.get("height_mm", 0)
        if w <= 0 or h <= 0:
            continue
        area = w * h
        aspect = min(w, h) / max(w, h) if max(w, h) > 0 else 0
        if area <= via_max_area and aspect >= _VIA_MIN_ASPECT_RATIO:
            _log.info(
                "Via filter: %s area=%.2f mm² aspect=%.2f → marking as via-like",
                c["label"], area, aspect,
            )
            c["confidence"] = round(min(c["confidence"], 0.50), 3)
            c["requires_confirmation"] = True
            c["_via_like"] = True
    return raw_candidates


def _parse_polygon_vertices(
    polygon_raw: list,
    width_mm: float,
    height_mm: float,
) -> tuple[list[dict], bool]:
    """Parse and validate polygon vertices from VLM response.

    Returns (polygon_mm, is_valid).
    Fraction coords are converted to mm. Out-of-range vertices are rejected.
    """
    result: list[dict] = []
    for pt in polygon_raw:
        if not isinstance(pt, dict):
            continue
        try:
            fx = float(pt.get("x_frac", pt.get("x", float("nan"))))
            fy = float(pt.get("y_frac", pt.get("y", float("nan"))))
        except (ValueError, TypeError):
            continue

        # Reject clearly out-of-range coords
        if fx < -0.1 or fx > 1.1 or fy < -0.1 or fy > 1.1:
            continue
        fx = max(0.0, min(1.0, fx))
        fy = max(0.0, min(1.0, fy))

        result.append({
            "x_mm": round(fx * width_mm, 3),
            "y_mm": round(fy * height_mm, 3),
        })

    # Need at least 3 non-collinear points for a valid polygon
    if len(result) < 3:
        return [], False

    # Check for degenerate (all points same or near-collinear)
    xs = [p["x_mm"] for p in result]
    ys = [p["y_mm"] for p in result]
    range_x = max(xs) - min(xs)
    range_y = max(ys) - min(ys)
    if range_x < 0.1 and range_y < 0.1:
        return [], False  # degenerate

    return result, True


def _clean_label(raw: str) -> str:
    """Normalize common OCR/VLM label variations."""
    raw = raw.replace(" ", "").replace("＋", "+").replace("－", "-")
    raw = re.sub(r"[^A-Z0-9+\-]", "", raw.upper())
    mapping = {
        "B-": "B-", "B-L": "B-", "B--": "B-",
        "B+": "B+", "+B": "B+",
        "P-": "P-", "PS": "P-", "P5": "P-",
        "P+": "P+", "P4": "P+",
        "C-": "C-",
        "C+": "C+",
        "NTC": "NTC", "N": "N", "TH": "TH", "T": "TH",
        "1D": "ID", "LD": "ID", "ID": "ID",
    }
    return mapping.get(raw, raw)


# ══════════════════════════════════════════════════════════════════════
#  Unified 3-element detection (outline + holes + pads)
# ══════════════════════════════════════════════════════════════════════

def detect_all_vlm(
    rectified_png: bytes,
    width_mm: float,
    height_mm: float,
    side: str,
) -> dict:
    """Unified detection: board outline + holes/slots + solder pads.

    One VLM call returns all three element types. The returned dict has keys
    "outline", "holes", "pads" — each a list of standardised candidate dicts.

    This is the recommended entry point for the black-frame calibration
    pipeline, replacing the pad-only detect_with_vlm().
    """
    if side not in {"front", "back"}:
        from .errors import DesignError
        raise DesignError("INVALID_BOARD_SIDE", "side must be front or back")

    _check_vlm_available()

    raw = _vlm_detect_unified(rectified_png, width_mm, height_mm)
    if raw is None:
        from .errors import DesignError
        raise DesignError("VLM_DETECTION_FAILED",
                          "qwen3.7-plus returned no usable response for unified detection")

    result = _parse_unified_response(raw, width_mm, height_mm, side)
    _log.info("Unified VLM: outline=%d holes=%d pads=%d for side %r",
              len(result["outline"]), len(result["holes"]), len(result["pads"]), side)

    return {
        "side": side,
        "width_mm": width_mm,
        "height_mm": height_mm,
        "outline": result["outline"],
        "holes": result["holes"],
        "pads": result["pads"],
        "candidate_count": len(result["pads"]),
        "hole_count": len(result["holes"]),
        "method": "vlm+unified",
    }


# ── Unified prompt ───────────────────────────────────────────────────

_UNIFIED_PROMPT_TEMPLATE = """You are an expert PCB (printed circuit board) inspector.
Analyze this rectified top-down photo of a battery protection PCB sitting on white
paper, inside a printed black rectangular frame.

Board physical dimensions: {width:.1f} mm × {height:.1f} mm (interior of the black frame).

IMPORTANT — The black rectangular frame is a PRINTED REFERENCE, NOT part of the PCB.
The PCB itself is the dark/colored object INSIDE the frame on white paper.

Your task — identify THREE categories of regions:

═══════════════════════════════════════
CATEGORY 1 – BOARD OUTLINE
═══════════════════════════════════════
Identify the outer boundary of the entire PCB board (the dark green/blue/black
substrate material). Return ONE polygon tracing the board perimeter.

- The board is INSIDE the printed black frame, surrounded by white paper.
- Exclude the black printed frame — that's the reference, not the PCB.
- Exclude protruding wires or connectors that extend beyond the board edge.
- The board is typically rectangular with possible corner cutouts, edge grooves,
  or edge notches — concave indentations that cut INWARD from the board perimeter.
  Trace these indentation contours faithfully; do NOT approximate the board as a
  simple rectangle that ignores notches.

Output:
  "outline": {{
    "polygon": [{{"x_frac":..., "y_frac":...}}, ...]
    — Clockwise order, tracing the board boundary faithfully.
    — Start from the top-leftmost corner.
    — Include vertices at every visible corner, notch, and protrusion.
    — The exact vertex count depends on the board's shape.
  }}

═══════════════════════════════════════
CATEGORY 2 – HOLES, SLOTS & EDGE GROOVES
═══════════════════════════════════════
Identify all physical voids ON the PCB board — through-holes, milled slots,
and edge grooves (notches cut into the board perimeter).

TYPES:
  "round"       — circular hole (one center + 4-8 polygon points)
  "slot"        — elongated milled slot (rounded rectangle ends)
  "edge_groove" — a concave notch or groove at the board EDGE (U-shaped or
                  rectangular cutout that intrudes inward from the perimeter)
  "irregular"   — any other shape

RULES:
- Holes appear as DARK regions within the board area. They are often black or
  very dark because you see through the board to the background.
- Edge grooves are indentations along the board perimeter — the cutout area
  is usually LIGHTER (white paper showing through) or DARKER (shadow).
  Look for NOTCHES in the board outline contour.
- Large plated mounting holes usually have a silvery/gold ring around them.
- Small via holes may be filled or have a tiny dark center.
- Ignore components and text — focus on physical voids in the board material.
- Ignore the area OUTSIDE the board — only detect holes within the board outline.

For EACH hole, provide:
  "type": "round" | "slot" | "irregular"
  "polygon": [{{"x_frac":...,"y_frac":...}}, ...]
      — 4-8 vertices, clockwise order.
      — For round holes: approximate the circle with 4-8 evenly-spaced points.
      — For slots: trace the slot contour.
  "confidence": 0.0 to 1.0

═══════════════════════════════════════
CATEGORY 3 – SOLDER PADS (same as before)
═══════════════════════════════════════
Locate ALL terminal solder pads labeled with silkscreen text.

Labels: B+, B-, P+, P-, C+, C-, NTC, TH, ID, N

For each pad:
  "label": terminal label string
  "polygon": [{{"x_frac":...,"y_frac":...}}, ...]
      — 4-8 vertices tracing the metallic pad boundary, clockwise.
  "confidence": 0.0 to 1.0

PAD GUIDELINES:
- All pads are ROUNDED RECTANGLES with perspective → appear as skewed quadrilaterals.
- 4-vertex output (corner points) is preferred and sufficient.
- Measure each pad independently — don't reuse coordinates across pads.
- Metallic area only (silver/gold/copper), NOT surrounding PCB or text.

═══════════════════════════════════════
OUTPUT FORMAT
═══════════════════════════════════════
Return ONLY a single JSON object — no markdown, no explanation.
Fractional coordinates (0.0 to 1.0, 4-5 decimal places) relative to full image.

{{
  "outline": {{
    "polygon": [
      {{"x_frac":0.0500,"y_frac":0.0300}},
      {{"x_frac":0.9500,"y_frac":0.0280}},
      {{"x_frac":0.9520,"y_frac":0.9700}},
      {{"x_frac":0.0480,"y_frac":0.9720}}
    ]
  }},
  "holes": [
    {{
      "type": "round",
      "polygon": [
        {{"x_frac":0.3000,"y_frac":0.1000}},
        {{"x_frac":0.3120,"y_frac":0.0980}},
        {{"x_frac":0.3150,"y_frac":0.1100}},
        {{"x_frac":0.3030,"y_frac":0.1120}}
      ],
      "confidence": 0.95
    }},
    {{
      "type": "edge_groove",
      "polygon": [
        {{"x_frac":0.4500,"y_frac":0.1200}},
        {{"x_frac":0.4700,"y_frac":0.1150}},
        {{"x_frac":0.4680,"y_frac":0.1450}},
        {{"x_frac":0.4520,"y_frac":0.1480}}
      ],
      "confidence": 0.90
    }}
  ],
  "pads": [
    {{
      "label": "B+",
      "polygon": [
        {{"x_frac":0.3123,"y_frac":0.0534}},
        {{"x_frac":0.3845,"y_frac":0.0512}},
        {{"x_frac":0.3867,"y_frac":0.1023}},
        {{"x_frac":0.3101,"y_frac":0.1045}}
      ],
      "confidence": 0.95
    }}
  ]
}}

FINAL REMINDERS:
- The black printed frame is NOT part of your output — it's just a calibration reference.
- Board outline = the PCB substrate boundary, not the frame.
- Holes must be ON the board, not in the white paper area.
- Pads are metallic contact areas on the board surface.
- ALL coordinates as x_frac / y_frac (0.0–1.0).
- Output exactly ONE JSON object. No markdown fences."""


# ── Unified VLM call ─────────────────────────────────────────────────

def _vlm_detect_unified(
    rectified_png: bytes,
    width_mm: float,
    height_mm: float,
) -> dict | None:
    """Call qwen3.7-plus with the unified 3-element prompt."""
    dashscope.api_key = _get_api_key()

    image_b64 = base64.b64encode(rectified_png).decode("ascii")
    image_url = f"data:image/png;base64,{image_b64}"

    prompt = _UNIFIED_PROMPT_TEMPLATE.format(width=width_mm, height=height_mm)

    messages = [{
        "role": "user",
        "content": [
            {"image": image_url},
            {"text": prompt},
        ],
    }]

    _log.info("Calling VLM (unified, model=%s)", MODEL_NAME)

    response = _vlm_call_with_retry(
        model=MODEL_NAME,
        messages=messages,
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS,
        enable_thinking=ENABLE_THINKING,
    )

    if response is None:
        return None

    if response.status_code != 200:
        _log.error("VLM API error: code=%s message=%s",
                   getattr(response, 'code', '?'),
                   getattr(response, 'message', '?'))
        return None

    try:
        contents = response.output.choices[0].message.content
    except (AttributeError, IndexError, KeyError) as exc:
        _log.error("Unexpected DashScope response: %s", exc)
        return None

    raw_text = ""
    for part in contents:
        if isinstance(part, dict) and "text" in part:
            raw_text += part["text"]

    if not raw_text:
        _log.error("DashScope returned empty text (unified)")
        return None

    _log.debug("VLM unified response (first 500 chars): %s", raw_text[:500])

    return _extract_json(raw_text)


# ── Unified response parser ──────────────────────────────────────────

def _parse_unified_response(
    raw: dict,
    width_mm: float,
    height_mm: float,
    side: str,
) -> dict:
    """Parse unified VLM response into outline + holes + pads structures.

    Returns: {"outline": [...], "holes": [...], "pads": [...]}
    Each list contains standardised candidate dicts compatible with
    the existing pipeline (visible_region, label, confidence, etc.)
    """
    result: dict = {"outline": [], "holes": [], "pads": []}

    # ── Parse board outline ──
    outline_data = raw.get("outline")
    if isinstance(outline_data, dict):
        poly = _parse_polygon_vertices(
            outline_data.get("polygon", []), width_mm, height_mm)
        if poly[1]:
            result["outline"] = [_make_outline_candidate(poly[0], side)]

    # ── Parse holes ──
    holes_data = raw.get("holes", [])
    if isinstance(holes_data, list):
        for i, hole in enumerate(holes_data):
            if not isinstance(hole, dict):
                continue
            hole_type = str(hole.get("type", "round")).lower()
            if hole_type not in ("round", "slot", "edge_groove", "irregular"):
                hole_type = "irregular"
            poly = _parse_polygon_vertices(
                hole.get("polygon", []), width_mm, height_mm)
            if not poly[1]:
                continue
            confidence = float(hole.get("confidence", 0.7))

            xs = [p["x_mm"] for p in poly[0]]
            ys = [p["y_mm"] for p in poly[0]]
            cx = sum(xs) / len(xs)
            cy = sum(ys) / len(ys)
            min_x, max_x = min(xs), max(xs)
            min_y, max_y = min(ys), max(ys)

            result["holes"].append({
                "id": f"hole_{i + 1}",
                "label": f"h{i + 1}",
                "hole_type": hole_type,
                "visible_position": {"x_mm": round(cx, 3), "y_mm": round(cy, 3)},
                "visible_region": {
                    "type": "edge_groove" if hole_type == "edge_groove" else "hole",
                    "visual_class": "void",
                    "shape": hole_type if hole_type != "irregular" else "custom",
                    "center": {"x_mm": round(cx, 3), "y_mm": round(cy, 3)},
                    "bbox": {
                        "x_mm": round(min_x, 3),
                        "y_mm": round(min_y, 3),
                        "width_mm": round(max_x - min_x, 3),
                        "height_mm": round(max_y - min_y, 3),
                    },
                    "polygon": poly[0],
                    "source": "vlm",
                },
                "matched_regions": [],
                "text_region": None,
                "roles": [],
                "polarity": None,
                "side": side,
                "shape": hole_type,
                "width_mm": round(max_x - min_x, 3),
                "height_mm": round(max_y - min_y, 3),
                "confidence": round(min(0.99, confidence), 3),
                "method": "vlm→qwen3.7-plus→unified",
                "region_resolved": True,
                "match_quality": "auto",
                "match_distance_mm": None,
                "requires_confirmation": True,
            })

    # ── Parse pads (reuse existing logic) ──
    pads_data = raw.get("pads", [])
    if isinstance(pads_data, list):
        # Wrap in expected format for _parse_vlm_response
        pad_wrapper = {"items": pads_data}
        result["pads"] = _parse_vlm_response(pad_wrapper, width_mm, height_mm, side)

    return result


def _make_outline_candidate(
    polygon_mm: list[dict],
    side: str,
) -> dict:
    """Build a board-outline candidate from polygon vertices."""
    xs = [p["x_mm"] for p in polygon_mm]
    ys = [p["y_mm"] for p in polygon_mm]
    cx = sum(xs) / len(xs)
    cy = sum(ys) / len(ys)
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)

    return {
        "id": "board_outline",
        "label": "BOARD_OUTLINE",
        "hole_type": "board_outline",
        "visible_position": {"x_mm": round(cx, 3), "y_mm": round(cy, 3)},
        "visible_region": {
            "type": "board_outline",
            "visual_class": "boundary",
            "shape": "polygon",
            "center": {"x_mm": round(cx, 3), "y_mm": round(cy, 3)},
            "bbox": {
                "x_mm": round(min_x, 3),
                "y_mm": round(min_y, 3),
                "width_mm": round(max_x - min_x, 3),
                "height_mm": round(max_y - min_y, 3),
            },
            "polygon": polygon_mm,
            "source": "vlm",
        },
        "matched_regions": [{
            "type": "board_outline",
            "visual_class": "boundary",
            "shape": "polygon",
            "center": {"x_mm": round(cx, 3), "y_mm": round(cy, 3)},
            "bbox": {
                "x_mm": round(min_x, 3),
                "y_mm": round(min_y, 3),
                "width_mm": round(max_x - min_x, 3),
                "height_mm": round(max_y - min_y, 3),
            },
            "polygon": polygon_mm,
            "source": "vlm",
        }],
        "text_region": None,
        "roles": [],
        "polarity": None,
        "side": side,
        "shape": "polygon",
        "width_mm": round(max_x - min_x, 3),
        "height_mm": round(max_y - min_y, 3),
        "confidence": 0.85,
        "method": "vlm→qwen3.7-plus→unified",
        "region_resolved": True,
        "match_quality": "auto",
        "match_distance_mm": None,
        "requires_confirmation": True,
    }


# ══════════════════════════════════════════════════════════════════════
#  Component Detection (元器件识别)
# ══════════════════════════════════════════════════════════════════════

def detect_components(
    rectified_png: bytes,
    width_mm: float,
    height_mm: float,
    side: str,
) -> dict:
    """Detect electronic components on the PCB using the original rectified image.

    Unlike pad detection which uses cropped/upscaled images, this function
    uses the original rectified PNG to preserve full visual context needed
    to read tiny silkscreen text on ICs, MOSFETs, resistors, etc.

    Returns:
        {
            "side": str,
            "components": [
                {
                    "id": str,
                    "type": "ic"|"mosfet"|"resistor"|"capacitor"|"diode"|"ntc"|"led"|"other",
                    "silkscreen": str,        # 丝印文字
                    "package": str,           # 封装类型
                    "position_mm": {"x": float, "y": float},
                    "size_mm": {"w": float, "h": float},
                    "confidence": float,
                    "side": str,
                },
                ...
            ],
            "inferred_ic": str | None,   # 自动推断的 IC 型号
            "inferred_mos": str | None,  # 自动推断的 MOS 型号
            "method": str,
        }
    """
    if side not in {"front", "back"}:
        from .errors import DesignError
        raise DesignError("INVALID_BOARD_SIDE", "side must be front or back")

    _check_vlm_available()

    raw = _vlm_detect_components_raw(rectified_png, width_mm, height_mm, side)
    if raw is None:
        from .errors import DesignError
        raise DesignError("VLM_DETECTION_FAILED",
                          "qwen3.7-plus returned no usable response for component detection")

    components = _parse_components_response(raw, width_mm, height_mm, side)

    # Auto-infer IC and MOS models from detected components
    inferred_ic = None
    inferred_mos = None
    for comp in components:
        if comp["type"] == "ic" and comp["silkscreen"] and inferred_ic is None:
            inferred_ic = comp["silkscreen"]
        elif comp["type"] == "mosfet" and comp["silkscreen"] and inferred_mos is None:
            inferred_mos = comp["silkscreen"]

    _log.info("Component detection: %d components for side %r (ic=%s, mos=%s)",
              len(components), side, inferred_ic, inferred_mos)

    return {
        "side": side,
        "components": components,
        "inferred_ic": inferred_ic,
        "inferred_mos": inferred_mos,
        "method": "vlm→qwen3.7-plus→components",
    }


# ── Component prompt ─────────────────────────────────────────────────

_COMPONENT_PROMPT_TEMPLATE = """You are an expert PCB (printed circuit board) inspector.
Analyze this rectified top-down photo of a battery protection PCB.

Board physical dimensions: {width:.1f} mm × {height:.1f} mm.
Image side: {side} (front = component side with most ICs; back = usually has battery pads and some components).

Your task: Identify ALL electronic components visible on this PCB board.

COMPONENT TYPES to identify:
1. **IC** — Integrated circuits (protection IC, controller IC). Usually black rectangular packages with many pins.
   - Read the silkscreen text ON the chip body (e.g., "DW01", "FS8205", "HY2113", "S-8261").
   - Identify package type (SOT-23-6, SOT-23-5, TSSOP-8, SOP-8, QFN, etc.)

2. **MOSFET** — Power MOS transistors. Usually small black packages with 3-8 pins.
   - Read the silkscreen code on the chip body (e.g., "A09", "FS8205A", "Si2302").
   - Identify package type (SOT-23, TDFN, SOT-23-6, etc.)

3. **Resistor** — Small rectangular surface-mount components.
   - Read the resistance marking if visible (e.g., "102" = 1kΩ, "472" = 4.7kΩ, "100" = 10Ω).
   - Estimate package size (0402, 0603, 0805, 1206).

4. **Capacitor** — Small rectangular SMD components, often brown/tan or dark.
   - Note any polarity marking (+ or stripe for tantalum).
   - Estimate package size (0402, 0603, 0805).

5. **Diode** — Two-terminal SMD components, often with a stripe/cathode mark.
   - Read marking if visible (e.g., "SS14", "1N4148").

6. **NTC** — Thermistor, usually a small round or rectangular component.
   - May have "NTC" or resistance marking.

7. **LED** — Light-emitting diode, usually has colored or translucent body.

8. **Other** — Any other component not fitting above categories.

IMPORTANT GUIDELINES:
- The PCB sits INSIDE a black rectangular calibration frame on white paper.
- Focus on components ON the PCB board, ignore the black frame and white paper.
- Read silkscreen text CAREFULLY — component markings are tiny and may be partially obscured.
- For ICs and MOSFETs: the silkscreen on the chip body is the MOST IMPORTANT identifier.
- For resistors: the 3-digit or 4-digit code is the resistance value.
- Position coordinates are FRACTIONAL (0.0 to 1.0) relative to the FULL IMAGE dimensions.
- Size (bounding box) is also FRACTIONAL.

OUTPUT FORMAT — Return ONLY a JSON object, no markdown fences, no explanation:
{{
  "components": [
    {{
      "type": "ic",
      "silkscreen": "DW01",
      "package": "SOT-23-6",
      "position": {{"x_frac": 0.4500, "y_frac": 0.3500}},
      "size": {{"w_frac": 0.0600, "h_frac": 0.0300}},
      "confidence": 0.92
    }},
    {{
      "type": "mosfet",
      "silkscreen": "FS8205A",
      "package": "SOT-23-6",
      "position": {{"x_frac": 0.5500, "y_frac": 0.4200}},
      "size": {{"w_frac": 0.0550, "h_frac": 0.0280}},
      "confidence": 0.88
    }},
    {{
      "type": "resistor",
      "silkscreen": "102",
      "package": "0603",
      "position": {{"x_frac": 0.3200, "y_frac": 0.5000}},
      "size": {{"w_frac": 0.0200, "h_frac": 0.0120}},
      "confidence": 0.75
    }}
  ]
}}

If a component's silkscreen is unreadable, set "silkscreen" to "" (empty string).
If package type is uncertain, use "unknown".
Only include components you can actually SEE on the PCB board surface."""


def _vlm_detect_components_raw(
    rectified_png: bytes,
    width_mm: float,
    height_mm: float,
    side: str,
) -> dict | None:
    """Call qwen3.7-plus with the component detection prompt."""
    dashscope.api_key = _get_api_key()

    image_b64 = base64.b64encode(rectified_png).decode("ascii")
    image_url = f"data:image/png;base64,{image_b64}"

    prompt = _COMPONENT_PROMPT_TEMPLATE.format(
        width=width_mm, height=height_mm, side=side)

    messages = [{
        "role": "user",
        "content": [
            {"image": image_url},
            {"text": prompt},
        ],
    }]

    _log.info("Calling VLM (components, model=%s, side=%s, image_size=%d bytes)",
              MODEL_NAME, side, len(rectified_png))

    response = _vlm_call_with_retry(
        model=MODEL_NAME,
        messages=messages,
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS,
        enable_thinking=ENABLE_THINKING,
    )

    if response is None:
        return None

    if response.status_code != 200:
        _log.error("VLM API error (components): code=%s message=%s",
                   getattr(response, 'code', '?'),
                   getattr(response, 'message', '?'))
        return None

    try:
        contents = response.output.choices[0].message.content
    except (AttributeError, IndexError, KeyError) as exc:
        _log.error("Unexpected DashScope response (components): %s", exc)
        return None

    raw_text = ""
    for part in contents:
        if isinstance(part, dict) and "text" in part:
            raw_text += part["text"]

    if not raw_text:
        _log.error("DashScope returned empty text (components)")
        return None

    _log.debug("VLM components response (first 500 chars): %s", raw_text[:500])

    return _extract_json(raw_text)


def _parse_components_response(
    raw: dict,
    width_mm: float,
    height_mm: float,
    side: str,
) -> list[dict]:
    """Parse VLM component detection response into standardized component list.

    Returns list of component dicts with mm-based positions and sizes.
    """
    VALID_TYPES = {"ic", "mosfet", "resistor", "capacitor", "diode", "ntc", "led", "other"}
    components: list[dict] = []

    items = raw.get("components", [])
    if not isinstance(items, list):
        _log.warning("Component response 'components' is not a list")
        return components

    for i, item in enumerate(items):
        if not isinstance(item, dict):
            continue

        comp_type = str(item.get("type", "other")).lower().strip()
        if comp_type not in VALID_TYPES:
            comp_type = "other"

        silkscreen = str(item.get("silkscreen", "")).strip()
        package = str(item.get("package", "unknown")).strip()
        confidence = float(item.get("confidence", 0.5))
        confidence = max(0.0, min(1.0, confidence))

        # Parse position
        pos = item.get("position", {})
        x_frac = float(pos.get("x_frac", -1))
        y_frac = float(pos.get("y_frac", -1))
        if x_frac < -0.1 or x_frac > 1.1 or y_frac < -0.1 or y_frac > 1.1:
            _log.debug("Component %d: position out of range (%.3f, %.3f), skipping", i, x_frac, y_frac)
            continue
        x_frac = max(0.0, min(1.0, x_frac))
        y_frac = max(0.0, min(1.0, y_frac))

        # Parse size
        size = item.get("size", {})
        w_frac = float(size.get("w_frac", 0.03))
        h_frac = float(size.get("h_frac", 0.02))
        w_frac = max(0.001, min(0.5, w_frac))
        h_frac = max(0.001, min(0.5, h_frac))

        # Convert to mm
        cx_mm = round(x_frac * width_mm, 3)
        cy_mm = round(y_frac * height_mm, 3)
        w_mm = round(w_frac * width_mm, 3)
        h_mm = round(h_frac * height_mm, 3)

        components.append({
            "id": f"comp_{i + 1}",
            "type": comp_type,
            "silkscreen": silkscreen,
            "package": package,
            "position_mm": {"x": cx_mm, "y": cy_mm},
            "size_mm": {"w": w_mm, "h": h_mm},
            "confidence": round(confidence, 3),
            "side": side,
        })

    return components


# ══════════════════════════════════════════════════════════════════════
#  Pad visual quality verification (single-pad VLM check)
# ══════════════════════════════════════════════════════════════════════

def verify_pad_crop(
    crop_png: bytes,
    label: str,
    pad_w_mm: float,
    pad_h_mm: float,
) -> dict:
    """Send a single pad's cropped image to VLM for quality verification.

    Returns:
        {
            "ok": bool,
            "single_pad": bool,
            "issues": [str],
            "confidence": float,
        }
    """
    _check_vlm_available()

    dashscope.api_key = _get_api_key()
    image_b64 = base64.b64encode(crop_png).decode("ascii")
    image_url = f"data:image/png;base64,{image_b64}"

    prompt = f"""You are inspecting a single cropped region from a PCB board, supposedly containing ONE solder pad labeled "{label}".

The crop region is approximately {pad_w_mm:.1f}mm x {pad_h_mm:.1f}mm on the actual PCB.
Pixel resolution: roughly {pad_w_mm * 25:.0f} x {pad_h_mm * 25:.0f} pixels (assume 25 px/mm).

Your task: verify whether this crop genuinely contains exactly ONE well-formed solder pad.

Answer these questions:
1. Is there clearly ONE metallic pad in this crop? (YES / NO)
2. Is the crop showing parts of TWO or more distinct pads? (YES if the crop spans across multiple pads, cutting through them)
3. Is the metal pad actually inside the visible area (not cut off at the edge, not mostly outside)?
4. Are there any obvious quality issues?

IMPORTANT about low-resolution / small crops:
- If the crop is very small (pad_w_mm < 1.5 mm), do NOT fail just because the image is low-res or slightly blurry.
- Look for the characteristic COLOR and TEXTURE of a solder pad: shiny silver/gold metallic surface against a green solder-mask background.
- A small low-res metallic rectangle with expected pad color IS a valid pad. Only mark FAIL if you are CERTAIN no metal pad exists (e.g., pure green background, silkscreen text without any metal, or a dark component body).
- "Blurry" or "low resolution" alone is NOT a failure reason — the crop is necessarily small.

Return a JSON object:
{{"single_pad": true/false, "issues": ["list of problems found, empty if clean"], "confidence": 0.0-1.0}}

Only return the JSON object. No markdown, no explanation."""

    messages = [{
        "role": "user",
        "content": [
            {"image": image_url},
            {"text": prompt},
        ],
    }]

    _log.info("Pad verification VLM call: label=%s, image_size=%d bytes", label, len(crop_png))

    response = _vlm_call_with_retry(
        model=MODEL_NAME,
        messages=messages,
        temperature=0.1,
        max_tokens=1024,
        enable_thinking=False,
    )

    if response is None:
        return {"ok": False, "single_pad": False, "issues": ["VLM call failed after retries"], "confidence": 0.0}

    if response.status_code != 200:
        _log.error("Pad verification VLM error: code=%s msg=%s",
                   getattr(response, 'code', '?'),
                   getattr(response, 'message', '?'))
        return {"ok": False, "single_pad": False, "issues": [f"API error {getattr(response, 'code', '?')}"], "confidence": 0.0}

    try:
        contents = response.output.choices[0].message.content
    except (AttributeError, IndexError, KeyError) as exc:
        _log.error("Pad verification response parse error: %s", exc)
        return {"ok": False, "single_pad": False, "issues": ["Response parse error"], "confidence": 0.0}

    raw_text = ""
    for part in contents:
        if isinstance(part, dict) and "text" in part:
            raw_text += part["text"]

    if not raw_text:
        return {"ok": False, "single_pad": False, "issues": ["Empty VLM response"], "confidence": 0.0}

    parsed = _extract_json(raw_text)
    if not isinstance(parsed, dict):
        return {"ok": False, "single_pad": False, "issues": ["Unparseable VLM response"], "confidence": 0.0}

    single = bool(parsed.get("single_pad", False))
    issues = parsed.get("issues", []) if isinstance(parsed.get("issues"), list) else []
    conf = max(0.0, min(1.0, float(parsed.get("confidence", 0.0))))
    ok = single and not issues

    return {
        "ok": ok,
        "single_pad": single,
        "issues": issues,
        "confidence": round(conf, 3),
    }

