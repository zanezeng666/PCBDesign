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
) -> dict:
    """Detect terminal candidates using qwen3.7-plus VLM.

    Pipeline:
      1. VLM identifies pads and returns approximate polygon coordinates

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

    return {
        "side": side,
        "width_mm": width_mm,
        "height_mm": height_mm,
        "candidate_count": len(candidates),
        "candidates": candidates,
        "annotated_png_base64": "",  # frontend draws its own overlay
        "method": "vlm+qwen3.7-plus",
        "notice": "VLM 视觉识别结果；请逐项人工确认。",
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

CRITICAL DOMAIN KNOWLEDGE — read carefully:
- ALL solder pads on this PCB are ROUNDED RECTANGLES in reality.
- Due to the camera angle (perspective distortion), each pad appears as a SKEWED QUADRILATERAL
  with slightly rounded corners in the image. The 4 edges may not be parallel to image axes.
- Your polygon vertices MUST capture this shape: 4 main corner vertices defining the outer
  boundary of the metallic pad, PLUS (optionally) transition points at rounded corners.

Your task:
1. Locate ALL terminal solder pads on this PCB that are labeled with silkscreen text (white text near metallic pads).
2. For each terminal pad, identify its label and trace the EXACT metallic pad outline.

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
- Pad = metallic area only (silver/gold/copper), NOT surrounding PCB or text.
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

    _log.info("Calling DashScope VLM (unified, model=%s)", MODEL_NAME)

    try:
        response = MultiModalConversation.call(
            model=MODEL_NAME,
            messages=messages,
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            enable_thinking=ENABLE_THINKING,
        )
    except Exception as exc:
        _log.error("DashScope API call failed (unified): %s", exc)
        return None

    if response.status_code != 200:
        _log.error("DashScope API error: code=%s message=%s",
                   response.code, response.message)
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



