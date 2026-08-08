"""Step 3 — Component detection via VLM.

Extracted from ``vlm_detection.py``.  Contains:
- ``detect_components`` — main entry point
- ``_vlm_detect_components_raw`` — raw VLM call
- ``_parse_components_response`` — response parser
- ``_COMPONENT_PROMPT_TEMPLATE`` — prompt template
"""

from __future__ import annotations

import base64

from ..core.logger import get_logger
from ..core.errors import DesignError
from ..core.vlm_client import (
    get_api_key as _get_api_key,
    check_vlm_available as _check_vlm_available,
    extract_json as _extract_json,
    vlm_call as _vlm_call_with_retry,
    MODEL_NAME,
    TEMPERATURE,
    MAX_TOKENS,
    ENABLE_THINKING,
)

try:
    import dashscope
except ImportError:
    dashscope = None  # type: ignore[assignment]

_log = get_logger(__name__)


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
                    "silkscreen": str,
                    "package": str,
                    "position_mm": {"x": float, "y": float},
                    "size_mm": {"w": float, "h": float},
                    "confidence": float,
                    "side": str,
                },
                ...
            ],
            "inferred_ic": str | None,
            "inferred_mos": str | None,
            "method": str,
        }
    """
    if side not in {"front", "back"}:
        raise DesignError("INVALID_BOARD_SIDE", "side must be front or back")

    _check_vlm_available()

    raw = _vlm_detect_components_raw(rectified_png, width_mm, height_mm, side)
    if raw is None:
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
