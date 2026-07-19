"""VLM-based pad detection using Aliyun DashScope qwen3.7-plus.

Pure VLM detection pipeline — no OCR or heuristic fallback.

Architecture:
    rectified_png (bytes) + width_mm/height_mm/side
        │
        ▼
    _vlm_detect_raw() ──► raw JSON from qwen3.7-plus
        │
        ▼
    _parse_vlm_response() ──► standardized candidate list
"""

from __future__ import annotations

import base64
import json
import logging
import os
import re

_log = logging.getLogger(__name__)

try:
    import dashscope
    from dashscope import MultiModalConversation
except ImportError:
    MultiModalConversation = None
    dashscope = None


# ── constants ────────────────────────────────────────────────────────

MODEL_NAME = "qwen3.7-plus"
TEMPERATURE = 0.05
MAX_TOKENS = 2048
ENABLE_THINKING = False  # Disable thinking mode for deterministic pad detection

# Labels we care about
TARGET_LABELS = {
    "B+", "B-", "P+", "P-", "C+", "C-",
    "NTC", "TH", "ID", "N",
}

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
    cv_refine: bool = True,
) -> dict:
    """Detect terminal candidates using qwen3.7-plus + optional CV refinement.

    Pipeline:
      1. VLM identifies pads and returns approximate polygon coordinates
      2. (Optional) CV edge detection refines each pad's polygon for precision

    Raises DesignError if VLM is unavailable or fails.
    """
    if side not in {"front", "back"}:
        from .errors import DesignError
        raise DesignError("INVALID_BOARD_SIDE", "side must be front or back")

    _check_vlm_available()

    raw = _vlm_detect_raw(rectified_png, width_mm, height_mm)
    if raw is None:
        from .errors import DesignError
        raise DesignError("VLM_DETECTION_FAILED", "qwen3.7-plus returned no usable response")

    candidates = _parse_vlm_response(raw, width_mm, height_mm, side)
    _log.info("VLM returned %d candidates for side %r", len(candidates), side)

    # ── CV refinement (optional) ──
    refined_count = 0
    if cv_refine and candidates:
        try:
            from .cv_refine import refine_pad_positions
            candidates = refine_pad_positions(rectified_png, candidates, width_mm, height_mm)
            refined_count = sum(
                1 for c in candidates if "cv_refine" in c.get("method", "")
            )
            _log.info("CV refinement applied to %d/%d candidates for side %r",
                      refined_count, len(candidates), side)
        except ImportError:
            _log.warning("cv_refine module not available, skipping CV refinement")
        except Exception as exc:
            _log.warning("CV refinement failed: %s, keeping VLM-only results", exc)

    return {
        "side": side,
        "width_mm": width_mm,
        "height_mm": height_mm,
        "candidate_count": len(candidates),
        "candidates": candidates,
        "annotated_png_base64": "",  # frontend draws its own overlay
        "method": "vlm+cv_refine" if refined_count > 0 else "vlm+qwen3.7-plus",
        "notice": "VLM 视觉识别结果；请逐项人工确认。" if refined_count == 0
                  else f"VLM + CV精修（{refined_count}个焊盘已精修）；请逐项人工确认。",
    }


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


def _vlm_detect_raw(rectified_png: bytes, width_mm: float, height_mm: float) -> dict | None:
    """Call qwen3.7-plus and return the parsed JSON response, or None on failure."""

    dashscope.api_key = _get_api_key()

    image_b64 = base64.b64encode(rectified_png).decode("ascii")
    image_url = f"data:image/png;base64,{image_b64}"

    prompt = _build_prompt(width_mm, height_mm)

    messages = [{
        "role": "user",
        "content": [
            {"image": image_url},
            {"text": prompt},
        ],
    }]

    _log.info("Calling DashScope VLM (model=%s, image_size=%d bytes)", MODEL_NAME, len(rectified_png))

    try:
        response = MultiModalConversation.call(
            model=MODEL_NAME,
            messages=messages,
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            enable_thinking=ENABLE_THINKING,
        )
    except Exception as exc:
        _log.error("DashScope API call failed: %s", exc)
        return None

    if response.status_code != 200:
        _log.error("DashScope API error: code=%s message=%s", response.code, response.message)
        return None

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


def _build_prompt(width_mm: float, height_mm: float) -> str:
    """Construct the system-style instruction for qwen3.7-plus."""

    return f"""You are an expert PCB (printed circuit board) inspector. Analyze the uploaded image of a battery protection board.

Board physical dimensions: {width_mm:.1f} mm × {height_mm:.1f} mm.
Image dimensions: exactly {int(width_mm * 50)} × {int(height_mm * 50)} pixels.

Your task:
1. Locate ALL terminal solder pads on this PCB that are labeled with silkscreen text (white text near metallic pads).
2. For each terminal pad, identify its label and trace the EXACT metallic pad outline with pixel-level precision.

Terminal labels to look for: B+, B-, P+, P-, C+, C-, NTC, TH, ID, N
- "B+" and "B-" are battery connection terminals (usually larger pads spanning most of the board height).
- "P+" and "P-" are charge/discharge terminals (medium pads typically on one edge).
- "C+" and "C-" are charge-only terminals.
- "NTC" or "TH" or "N" are thermistor terminals (small pads).
- "ID" is an identification/communication terminal.

For EACH terminal you find, provide:
- "label": the terminal label (B+, B-, P+, P-, C+, C-, NTC, TH, ID, N)
- "polygon": an array of vertex objects tracing the METALLIC PAD OUTLINE. Each vertex is
  {{"x_frac": ..., "y_frac": ...}} where x_frac and y_frac are fractional positions
  relative to image dimensions (0.0 = left/top edge, 1.0 = right/bottom edge).
  Use 4-5 DECIMAL PLACES for precision (e.g., 0.9563, 0.2104).
  Points MUST be in clockwise order forming a closed polygon.
  Trace ONLY the metallic (silver/gold/copper) pad area — NOT the silkscreen text.

  Guidelines for polygon vertices:
  - Rectangular pads (sharp corners): 4 corner vertices
  - Rounded-corner pads: use 6-8 vertices — corners PLUS extra transition points
  - Oval/elliptical pads: 6-8 vertices spaced around the curved boundary
  - Circular pads: 6-8 evenly spaced vertices around the circumference
  - Irregular shapes: trace with up to 12 vertices to capture the shape

- "confidence": 0.0 to 1.0 (how certain you are, 1.0 = very certain)

CRITICAL PRECISION RULES:
- MEASURE each pad's position INDEPENDENTLY by looking at its actual location in the image.
- DO NOT reuse the same x_frac (or y_frac) across different pads — tiny differences matter.
- Even pads that appear vertically aligned may have slightly different x positions — measure each one.
- Use the FULL image as your coordinate reference (0 to 1 = entire image width/height).
- If you can see the pad's metallic boundary clearly, estimate x_frac to 0.0001 precision.

Return ONLY a JSON array. No markdown, no explanation. Example format:
[
  {{"label":"B+","polygon":[{{"x_frac":0.3200,"y_frac":0.0600}},{{"x_frac":0.3800,"y_frac":0.0600}},{{"x_frac":0.3800,"y_frac":0.1000}},{{"x_frac":0.3200,"y_frac":0.1000}}],"confidence":0.95}},
  {{"label":"B-","polygon":[{{"x_frac":0.1200,"y_frac":0.1500}},{{"x_frac":0.1800,"y_frac":0.1500}},{{"x_frac":0.2000,"y_frac":0.1700}},{{"x_frac":0.2000,"y_frac":0.2600}},{{"x_frac":0.1800,"y_frac":0.2800}},{{"x_frac":0.1200,"y_frac":0.2800}},{{"x_frac":0.1000,"y_frac":0.2600}},{{"x_frac":0.1000,"y_frac":0.1700}}],"confidence":0.93}}
]

FINAL REMINDERS:
- Pad = metallic area only (silver/gold/copper), NOT surrounding PCB or text.
- Pads on opposite edges have very different x_frac (e.g. 0.05 vs 0.95).
- Measure each pad's polygon vertices individually by visually tracing its metallic boundary.
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

    Fractional coordinates (0.0–1.0) relative to image dimensions → mm.
    """

    items = raw.get("items", [])
    if not isinstance(items, list):
        items = []

    candidates: list[dict] = []
    seen_labels: set[str] = set()

    for item in items:
        if not isinstance(item, dict):
            continue
        label = str(item.get("label", "")).strip().upper()

        # Normalize common VLM output variations
        label = _clean_label(label)
        if label not in TARGET_LABELS:
            continue

        # Deduplicate: keep highest confidence per label
        confidence = float(item.get("confidence", 0.7))
        if label in seen_labels:
            existing = next(c for c in candidates if c["label"] == label)
            if confidence <= existing["confidence"]:
                continue
            candidates.remove(existing)
            seen_labels.discard(label)

        seen_labels.add(label)

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

            # Bounds check
            if x_frac < 0 or x_frac > 1 or y_frac < 0 or y_frac > 1:
                if x_frac < -0.5 or x_frac > 1.5 or y_frac < -0.5 or y_frac > 1.5:
                    continue
                x_frac = max(0, min(1, x_frac))
                y_frac = max(0, min(1, y_frac))

            cx = x_frac * width_mm
            cy = y_frac * height_mm
            pad_w_mm = w_frac * width_mm if w_frac > 0 else 4.0
            pad_h_mm = h_frac * height_mm if h_frac > 0 else 2.5

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

        roles, polarity = LABEL_CONTRACT.get(label, (set(), None))

        candidate = {
            "id": label.replace("+", "_POS").replace("-", "_NEG"),
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
        candidates.append(candidate)

    return sorted(candidates, key=lambda c: (c["visible_position"]["x_mm"], c["visible_position"]["y_mm"], c["label"]))


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
        "NTC": "NTC", "N": "N", "TH": "TH",
        "1D": "ID", "LD": "ID", "ID": "ID",
    }
    return mapping.get(raw, raw)



