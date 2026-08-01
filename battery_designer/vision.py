"""PCB contour recognition using adaptive multi-colour HSV pipeline + VLM refinement.

Pipeline (verified via pcb_image_pipeline.md):
  1. EXIF orientation fix — OpenCV ignores EXIF, piexif normalises it
  2. Black frame detection — simple threshold + convexHull + approxPolyDP(0.02*peri)
  3. Perspective correction — warp to 60×30mm reference frame
  4. Adaptive HSV PCB extraction — auto-detect green/blue/black/yellow + Otsu fallback
  5. Edge decontamination — 1px erosion + distanceTransform for clean transparent PNG
  6. VLM groove detection — validates CV convexity defects for edge grooves
"""

from __future__ import annotations

import base64, json, logging, math, os, re, uuid
from pathlib import Path

import cv2, numpy as np

_log = logging.getLogger(__name__)

try:
    import dashscope
    from dashscope import MultiModalConversation
except ImportError:
    MultiModalConversation = None
    dashscope = None

MODEL_NAME = "qwen3.7-plus"
TEMPERATURE = 0.05
MAX_TOKENS = 4096
ENABLE_THINKING = False
MAX_GROOVES = 6


# ═══════════════════════════════════════════════════════════════════════
#  EXIF orientation fix (verified pipeline, step 1)
# ═══════════════════════════════════════════════════════════════════════

def _fix_exif_orientation(img_buf: bytes) -> bytes:
    """Normalise EXIF orientation to 1 so OpenCV decodes images correctly.

    Many mobile cameras embed an orientation tag (e.g. 6 = 90° CW) that
    OpenCV ignores.  We use piexif to encode the correct orientation directly
    into the JPEG data before decoding.
    Returns unchanged bytes if the input is not JPEG or has no EXIF.
    """
    try:
        import piexif
    except ImportError:
        _log.info("piexif not installed — skipping EXIF orientation fix")
        return img_buf
    # Only JPEG files carry EXIF
    if img_buf[:2] != b'\xff\xd8':
        return img_buf
    try:
        exif_dict = piexif.load(img_buf)
        current = exif_dict.get("0th", {}).get(piexif.ImageIFD.Orientation, 1)
        if current == 1:
            return img_buf  # already normal
        exif_dict["0th"][piexif.ImageIFD.Orientation] = 1
        exif_bytes = piexif.dump(exif_dict)
        fixed = piexif.insert(exif_bytes, img_buf)
        _log.info("EXIF orientation fixed: %d → 1", current)
        return fixed
    except Exception as e:
        _log.warning("EXIF orientation fix failed (non-fatal): %s", e)
        return img_buf


# ═══════════════════════════════════════════════════════════════════════
#  Public API
# ═══════════════════════════════════════════════════════════════════════

def extract_pcb(rectified_png: bytes, width_mm: float, height_mm: float,
                pixels_per_mm: float) -> dict:
    """Extract PCB from rectified image using verified HSV pipeline.

    Pipeline:
      1. HSV PCB extraction (green/blue/yellow + Otsu fallback)
      2. Paper background model (colour + texture fingerprint)
      3. Outline refinement (~12 vertices via epsilon scan)
      4. Paper-validated edge notch detection
      5. Paper-matched internal hole/slot detection
      6. Edge-decontaminated transparent PNG

    Returns: outline, grooves, groove_count, groove_warning, holes, hole_count,
             pcb_mask_b64, transparent_pcb_b64, paper_model, method,
             debug_steps, vertex_count.
    """
    nparr = np.frombuffer(rectified_png, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        from .errors import DesignError
        raise DesignError("INVALID_IMAGE", "Failed to decode rectified image.")
    h, w = img.shape[:2]
    debug_steps = [{"step": "00_raw", "label": "原始图像", "image_base64": _to_b64(img)}]

    # ── Step 1: HSV PCB extraction ──
    pcb_mask, _ = _extract_pcb_hsv(img)
    if pcb_mask is None:
        from .errors import DesignError
        raise DesignError("NO_PCB_FOUND",
                          "Could not detect PCB board in rectified image.")
    debug_steps.append({"step": "01_hsv_extract", "label": "HSV PCB提取",
                        "pcb_mask_b64": _to_b64(pcb_mask)})

    # ── Step 2: Paper background model ──
    paper_model = _build_paper_model(img, pcb_mask)
    if paper_model:
        debug_steps.append({"step": "02_paper_model", "label": f"纸张底色模型(H{paper_model['h_lo']}-{paper_model['h_hi']})",
                            "paper_model": {k: v for k, v in paper_model.items() if isinstance(v, (int, float, str))}})

    # ── Step 2b: Paper-model shadow subtraction ──
    # Remove pixels from pcb_mask whose HSV values match the paper colour model.
    # Shadow regions near the PCB edge often have paper-like hue/saturation but
    # get captured by the broad HSV green range.  Subtracting them here cleans
    # the mask BEFORE outline refinement, eliminating shadow burrs.
    if paper_model:
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        paper_colour = cv2.inRange(hsv,
            np.array([paper_model["h_lo"], paper_model["s_lo"], paper_model["v_lo"]]),
            np.array([paper_model["h_hi"], paper_model["s_hi"], paper_model["v_hi"]]))
        before_nz = cv2.countNonZero(pcb_mask)
        pcb_mask = cv2.bitwise_and(pcb_mask, cv2.bitwise_not(paper_colour))
        after_nz = cv2.countNonZero(pcb_mask)
        removed_pct = (before_nz - after_nz) / max(before_nz, 1) * 100
        _log.info("Paper shadow subtraction: removed %d px (%.1f%%)", before_nz - after_nz, removed_pct)
        # Re-fill small holes created by subtraction
        if after_nz > 100:
            inv = cv2.bitwise_not(pcb_mask)
            nl, labs, stats, _ = cv2.connectedComponentsWithStats(inv, 8)
            for i in range(1, nl):
                if stats[i, cv2.CC_STAT_AREA] < 80:
                    pcb_mask[labs == i] = 255
            # Trim thin edge protrusions with a light morphological OPEN (3×3, 1 iter)
            k_edge = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            pcb_mask = cv2.morphologyEx(pcb_mask, cv2.MORPH_OPEN, k_edge, iterations=1)

    # ── Step 3: Outline refinement ──
    outline = _refine_outline(pcb_mask, width_mm, height_mm, pixels_per_mm)
    vertex_count = len(outline)
    debug_steps.append({"step": "03_outline", "label": f"PCB轮廓({vertex_count}顶点)",
                        "outline_mm": outline})

    # ── Step 3b: Rebuild clean mask from refined outline (eliminates HSV shadow burrs) ──
    if len(outline) >= 3:
        clean_mask = _outline_to_mask(outline, width_mm, height_mm, w, h)
        if cv2.countNonZero(clean_mask) > 100:
            pcb_mask = clean_mask

    # ── Step 4: Paper-validated edge notches ──
    notches, notch_warning = _validate_notches_by_paper(
        img, pcb_mask, outline, paper_model, width_mm, height_mm, pixels_per_mm)
    debug_steps.append({"step": "04_notches", "label": f"纸色验证凹槽({len(notches)}个)",
                        "notches": notches, "warning": notch_warning})

    # ── Step 5: Paper-matched internal holes ──
    holes = _detect_paper_holes(img, pcb_mask, paper_model,
                                 width_mm, height_mm, pixels_per_mm)
    debug_steps.append({"step": "05_holes", "label": f"纸色匹配孔槽({len(holes)}个)",
                        "holes": holes})

    # ── Step 6: Edge-decontaminated transparent PNG ──
    transparent_png = _make_transparent(img, pcb_mask)
    return {
        "outline": outline, "grooves": notches,
        "groove_count": len(notches), "groove_warning": notch_warning,
        "holes": holes, "hole_count": len(holes),
        "pcb_mask_b64": _to_b64(pcb_mask),
        "transparent_pcb_b64": base64.b64encode(transparent_png).decode("ascii"),
        "paper_model": {k: v for k, v in (paper_model or {}).items() if isinstance(v, (int, float, str))},
        "method": "hsv-pipeline+paper-model",
        "debug_steps": debug_steps,
        "vertex_count": vertex_count,
    }


def detect_holes(rectified_png: bytes, width_mm: float, height_mm: float,
                 pixels_per_mm: float, outline_mm: list[dict]) -> list[dict]:
    """Detect holes/slots inside PCB using paper colour+texture matching.

    Principle: PCB is on white paper.  Regions within the PCB outline whose
    colour+texture match the paper are holes (paper visible through them).
    """
    nparr = np.frombuffer(rectified_png, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        return []

    h, w_img = img.shape[:2]

    # Build PCB mask from outline
    outline_px = _mm_to_px(outline_mm, width_mm, height_mm, w_img, h)
    mask = np.zeros((h, w_img), dtype=np.uint8)
    if len(outline_px) >= 3:
        cv2.fillPoly(mask, [np.array(outline_px, dtype=np.int32)], 255)

    # Build paper model from rectified image
    paper_model = _build_paper_model(img, mask)

    # Detect holes via paper matching
    holes = _detect_paper_holes(img, mask, paper_model,
                                 width_mm, height_mm, pixels_per_mm)
    _log.info("detect_holes(paper): %d holes found", len(holes))
    return holes


# ═══════════════════════════════════════════════════════════════════════
#  VLM prompt & detection
# ═══════════════════════════════════════════════════════════════════════

_CONTOUR_PROMPT = """You are a PCB visual inspector. This is a rectified top-down photo of a
BATTERY PROTECTION BOARD (锂电池保护板) sitting on white A4 paper.

Board physical size: {width:.1f}mm × {height:.1f}mm. Image: ~{px_w}×{px_h}px.
The black calibration frame has been cropped away — what remains is the PCB board
(dark green, blue, or black solder mask) on clean white paper background.

CRITICAL — BOUNDING BOX vs PERIMETER TRACE:
This is NOT a bounding-box task. A bounding box (4 corners) is WRONG. Battery
protection PCBs have IRREGULAR perimeters with MANY notches, cutouts, and
protrusions along the edges. You MUST trace the actual perimeter, including
EVERY corner formed by edge deviations — even small ones (≥1mm).

HOW TO TRACE:
1. Look at the board edge carefully. Follow it pixel by pixel in your mind.
2. Place a vertex at EVERY point where the edge changes direction.
3. If the edge goes in (notch/groove) or out (protrusion/tab), you need
   EXTRA vertices to follow that shape — do NOT bridge across with a straight line.
4. Battery protection boards often have 6-16 perimeter vertices — 4 corners
   is always too few.

TASKS:
1. PCB PERIMETER TRACE — polygon following the actual board edge, CLOCKWISE:
   - Start from the top-leftmost vertex on the board perimeter.
   - Place vertices at EVERY visible corner, notch, protrusion, and edge
     direction change — even small ones. The polygon must follow the real
     perimeter shape, not just the four extreme corners.
   - Shadows on paper are NOT part of the board.
   - Bright PCB surface elements ARE part of the board.
2. SHADOWS — Darker paper regions OUTSIDE the PCB. One polygon per region.
3. EDGE GROOVES (凹槽) — MAX {max_grooves} most prominent concave indentations
   into the board edge where white paper is visible inside the notch.

Return ONLY a JSON object (no markdown, no explanation):
{{
  "outline": [{{"x_frac":0.1234,"y_frac":0.0567}}, ...],
  "shadows": [{{"polygon":[{{"x_frac":...,"y_frac":...}},...]}}],
  "grooves": [
    {{"type":"groove","polygon":[{{"x_frac":...,"y_frac":...}},...],
      "depth_mm":2.5,"confidence":0.85}}
  ]
}}

Coordinates: x_frac/y_frac are 0.0-1.0 fractions, 4-5 decimal places.
The outline polygon should faithfully follow the board's perimeter."""





def _get_api_key() -> str:
    key = os.getenv("DASHSCOPE_API_KEY", "").strip()
    if key:
        return key
    try:
        import winreg
        for hive, sk in ((winreg.HKEY_LOCAL_MACHINE,
                          r"SYSTEM\CurrentControlSet\Control\Session Manager\Environment"),
                         (winreg.HKEY_CURRENT_USER, r"Environment")):
            try:
                with winreg.OpenKey(hive, sk) as reg:
                    key, _ = winreg.QueryValueEx(reg, "DASHSCOPE_API_KEY")
                    if key: return key.strip()
            except OSError: continue
    except Exception: pass
    return ""


def _vlm_detect_contour(img: np.ndarray, width_mm: float, height_mm: float) -> dict:
    """Call Qwen VLM → {outline, grooves, shadows, holes}."""
    if not _get_api_key() or MultiModalConversation is None:
        _log.warning("VLM unavailable")
        return _empty()

    dashscope.api_key = _get_api_key()
    h, w = img.shape[:2]
    _, png = cv2.imencode(".png", img)
    url = f"data:image/png;base64,{base64.b64encode(png).decode('ascii')}"
    prompt = _CONTOUR_PROMPT.format(width=width_mm, height=height_mm,
                                     px_w=w, px_h=h, max_grooves=MAX_GROOVES)

    _log.info("VLM contour: %d×%d image", w, h)
    try:
        resp = MultiModalConversation.call(model=MODEL_NAME,
            messages=[{"role":"user","content":[{"image":url},{"text":prompt}]}],
            temperature=TEMPERATURE, max_tokens=MAX_TOKENS, enable_thinking=ENABLE_THINKING)
    except Exception as exc:
        _log.error("VLM API error: %s", exc)
        return _empty()

    if resp.status_code != 200:
        _log.error("VLM API status: %s", getattr(resp, 'code', '?'))
        return _empty()

    try:
        raw = "".join(p.get("text","") for p in resp.output.choices[0].message.content
                      if isinstance(p, dict))
    except Exception as exc:
        _log.error("VLM response parse: %s", exc)
        return _empty()

    if not raw: return _empty()
    _log.debug("VLM raw(500): %s", raw[:500])
    parsed = _extract_json(raw)
    if parsed is None: return _empty()

    return {
        "outline": _frac_list(parsed.get("outline",[]), width_mm, height_mm),
        "grooves": _parse_grooves(parsed.get("grooves",[]), width_mm, height_mm),
        "shadows": [{"polygon": _frac_list(s.get("polygon",[]), width_mm, height_mm)}
                    for s in parsed.get("shadows",[])],
        "holes": _parse_holes(parsed.get("holes",[]), width_mm, height_mm),
    }


def _vlm_detect_holes(img: np.ndarray, width_mm: float, height_mm: float) -> list[dict]:
    r = _vlm_detect_contour(img, width_mm, height_mm)
    return r.get("holes", [])


def _empty() -> dict:
    return {"outline":[],"grooves":[],"shadows":[],"holes":[]}











def _extract_json(text: str) -> dict | None:
    text = text.strip()
    for transform in (
        json.loads,
        lambda t: json.loads(t.split("```json",1)[-1].rsplit("```",1)[0].strip()),
        lambda t: json.loads(re.search(r"\{[\s\S]*\}",t).group()) if re.search(r"\{[\s\S]*\}",t) else None,
        lambda t: json.loads(re.search(r"\[[\s\S]*\]",t).group()) if re.search(r"\[[\s\S]*\]",t) else None,
    ):
        try:
            val = transform(text)
            return val if isinstance(val, dict) else {"items":val}
        except Exception:
            continue
    return None


def _frac_list(pts, w_mm, h_mm):
    return [{"x_mm":round(float(p.get("x_frac",0))*w_mm,3),
             "y_mm":round(float(p.get("y_frac",0))*h_mm,3)}
            for p in pts if isinstance(p,dict) and 0<=float(p.get("x_frac",-1))<=1]

def _parse_grooves(vg, w, h):
    out = []
    for i,g in enumerate(vg[:MAX_GROOVES]):
        if not isinstance(g,dict): continue
        poly = _frac_list(g.get("polygon",[]),w,h)
        if len(poly)<3: continue
        t = g.get("type","groove"); t = t if t in ("groove","protrusion") else "groove"
        out.append({"id":f"groove_{i+1:02d}","groove_type":t,"polygon":poly,
                     "depth_mm":float(g.get("depth_mm",0)),
                     "confidence":float(g.get("confidence",0.7)),"source":"vlm"})
    return out

def _parse_holes(vh, w, h):
    out = []
    for i,hh in enumerate(vh):
        if not isinstance(hh,dict): continue
        poly = _frac_list(hh.get("polygon",[]),w,h)
        if len(poly)<3: continue
        xs = [p["x_mm"] for p in poly]; ys = [p["y_mm"] for p in poly]
        out.append({"id":f"hole_{i+1:02d}","hole_type":hh.get("type","round"),
                     "center":{"x_mm":round(sum(xs)/len(xs),3),"y_mm":round(sum(ys)/len(ys),3)},
                     "polygon":poly,"confidence":float(hh.get("confidence",0.7)),"source":"vlm"})
    return out


# ═══════════════════════════════════════════════════════════════════════
#  HSV PCB extraction (adaptive multi-colour detection)
# ═══════════════════════════════════════════════════════════════════════

# Candidate PCB solder-mask colours: (label, HSV lower, HSV upper)
_PCB_COLOR_RANGES = [
    ("green",  np.array([55, 40, 18]),  np.array([100, 255, 255])),
    ("blue",   np.array([100, 30, 18]), np.array([130, 255, 255])),
    ("black",  np.array([0, 0, 10]),    np.array([180, 80, 100])),
    ("yellow", np.array([20, 30, 18]),  np.array([40, 255, 255])),
]


def _pcb_area_score(area_ratio: float) -> float:
    """Score how plausible a detected area ratio is for a PCB on paper.

    Ideal range is roughly 5%-70% of the image.  Returns higher score for
    more plausible ratios, penalising extremes.
    """
    if area_ratio < 0.02 or area_ratio > 0.92:
        return -1.0
    if 0.05 <= area_ratio <= 0.70:
        return 1.0
    if area_ratio < 0.05:
        return area_ratio / 0.05
    return (0.92 - area_ratio) / (0.92 - 0.70)


def _extract_pcb_hsv(img):
    """Extract PCB binary mask using adaptive multi-colour HSV detection.

    Automatically detects PCB board colour by evaluating ALL candidate
    colour ranges (green, blue, black, yellow) in parallel and selecting
    the one with the most plausible area ratio.

    Supported board colours:
      - Green (most common): H=55-100, S=40-255
      - Blue: H=100-130, S=30-255
      - Black: S≤80, V≤100 (low saturation + low brightness)
      - Yellow/tan: H=20-40, S=30-255

    Falls back to Otsu on LAB L-channel if no colour range yields a
    reasonable area.

    Returns (binary_mask, contour) or (None, None).
    """
    h, w = img.shape[:2]
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    k_close = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    k_open = np.ones((3, 3), np.uint8)

    best_mask = None
    best_contour = None
    best_area = 0.0
    best_label = None
    best_score = -1.0

    for label, lower, upper in _PCB_COLOR_RANGES:
        mask = cv2.inRange(hsv, lower, upper)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k_close, iterations=4)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k_open, iterations=1)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue

        pcb_contour = max(contours, key=cv2.contourArea)
        area_ratio = cv2.contourArea(pcb_contour) / (h * w)

        if area_ratio < 0.015 or area_ratio > 0.95:
            continue

        score = _pcb_area_score(area_ratio)
        _log.debug("HSV %s: area=%.1f%% score=%.3f", label, area_ratio * 100, score)

        if score > best_score:
            best_score = score
            best_mask = mask
            best_contour = pcb_contour
            best_area = area_ratio
            best_label = label

    if best_mask is not None:
        _log.info("PCB colour auto-detected: %s (area=%.1f%%, score=%.3f)",
                  best_label, best_area * 100, best_score)
        mask, pcb_contour, area_ratio = best_mask, best_contour, best_area
    else:
        # ── Last resort: Otsu on LAB L-channel ──
        _log.info("No HSV colour range yielded valid PCB — trying Otsu fallback")
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        L = lab[:, :, 0]
        _, mask = cv2.threshold(L, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k_close, iterations=4)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k_open, iterations=1)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            pcb_contour = max(contours, key=cv2.contourArea)
            area_ratio = cv2.contourArea(pcb_contour) / (h * w)
            _log.info("Otsu fallback: area=%.1f%%", area_ratio * 100)
        else:
            _log.error("All PCB extraction methods failed")
            return None, None

    # Fine approx (0.001 * perimeter — preserves micro-arcs and grooves)
    peri = cv2.arcLength(pcb_contour, True)
    smooth = cv2.approxPolyDP(pcb_contour, 0.001 * peri, True)

    # Fill binary mask with fine polygon
    binary = np.zeros((h, w), dtype=np.uint8)
    cv2.drawContours(binary, [smooth], -1, 255, -1)

    # Fill small internal holes
    inv = cv2.bitwise_not(binary)
    nl, labs, stats, _ = cv2.connectedComponentsWithStats(inv, 8)
    for i in range(1, nl):
        if stats[i, cv2.CC_STAT_AREA] < 50:
            binary[labs == i] = 255

    _log.info("PCB extracted [%s]: area=%.1f%%  contour_len=%.0fpx  fine_verts=%d",
              best_label or "otsu", area_ratio * 100, peri, len(smooth))
    return binary, pcb_contour


# ═══════════════════════════════════════════════════════════════════════
#  Paper background model (white paper colour + texture fingerprint)
# ═══════════════════════════════════════════════════════════════════════

def _build_paper_model(img, pcb_mask):
    """Build a colour+texture model of the white paper background.

    Samples pixels OUTSIDE the PCB to characterize the paper despite:
      - paper texture (not pure white)
      - uneven lighting / vignetting
      - shadows near the board edge

    Returns None if not enough paper pixels are available.
    H and S channels describe paper COLOUR (lighting-invariant).  V channel is
    loosened to allow darker paper that is visible through PCB holes.
    """
    h, w = img.shape[:2]
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Paper area = outside PCB (inverted mask), eroded to avoid edge bleed
    paper_mask = cv2.bitwise_not(pcb_mask)
    k_erode = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    paper_mask = cv2.erode(paper_mask, k_erode, iterations=2)

    paper_px = hsv[paper_mask > 0]
    if len(paper_px) < 200:
        # Fallback: sample from image corners
        corners = [(0, 0, 60, 60), (w-60, 0, w, 60),
                   (0, h-60, 60, h), (w-60, h-60, w, h)]
        samples = []
        for x1, y1, x2, y2 in corners:
            samples.append(hsv[y1:y2, x1:x2].reshape(-1, 3))
        paper_px = np.vstack(samples)

    if len(paper_px) < 100:
        _log.warning("Paper model: insufficient samples (%d)", len(paper_px))
        return None

    h_mean, h_std = float(np.mean(paper_px[:, 0])), float(np.std(paper_px[:, 0]))
    s_mean, s_std = float(np.mean(paper_px[:, 1])), float(np.std(paper_px[:, 1]))
    v_mean, v_std = float(np.mean(paper_px[:, 2])), float(np.std(paper_px[:, 2]))

    # ── Texture fingerprint: local intensity variation ──
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    lap = np.abs(cv2.Laplacian(gray, cv2.CV_64F))
    tex_blur = cv2.GaussianBlur(lap, (15, 15), 0)
    tex_vals = tex_blur[paper_mask > 0]
    tex_mean = float(np.mean(tex_vals)) if len(tex_vals) > 0 else 0.0

    model = {
        # H, S tight (±2.5σ → catches paper colour ignoring brightness)
        "h_lo": max(0, int(h_mean - 2.5 * h_std)),
        "h_hi": min(180, int(h_mean + 2.5 * h_std)),
        "s_lo": max(0, int(s_mean - 2.5 * s_std)),
        "s_hi": min(255, int(s_mean + 2.5 * s_std)),
        # V loose (±4σ low side → catches shadowed paper in holes)
        "v_lo": max(0, int(v_mean - 4.0 * v_std)),
        "v_hi": min(255, int(v_mean + 2.0 * v_std)),
        "h_mean": round(h_mean, 1), "s_mean": round(s_mean, 1), "v_mean": round(v_mean, 1),
        "tex_mean": round(tex_mean, 2),
        "sample_count": len(paper_px),
    }
    _log.info("Paper model: H[%d,%d] S[%d,%d] V[%d,%d] tex=%.1f samples=%d",
              model["h_lo"], model["h_hi"], model["s_lo"], model["s_hi"],
              model["v_lo"], model["v_hi"], model["tex_mean"], model["sample_count"])
    return model


# ═══════════════════════════════════════════════════════════════════════
#  Paper-matched hole & notch detection
# ═══════════════════════════════════════════════════════════════════════

def _detect_paper_holes(img, pcb_mask, paper_model, w_mm, h_mm, ppm):
    """Detect holes/slots within PCB by matching paper colour + texture.

    Principle: PCB sits on white paper.  If there's a hole through the board,
    the paper background is visible through it.  We find regions inside the
    PCB outline whose colour matches the paper model.

    Returns: list of hole dicts (id, hole_type, center, polygon, ...).
    """
    if paper_model is None:
        return []

    h, w = img.shape[:2]
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # ── Paper-colour matching mask (full image) ──
    paper_color = cv2.inRange(
        hsv,
        np.array([paper_model["h_lo"], paper_model["s_lo"], paper_model["v_lo"]]),
        np.array([paper_model["h_hi"], paper_model["s_hi"], paper_model["v_hi"]]),
    )

    # Get the external boundary of the PCB
    ext_cnts, _ = cv2.findContours(pcb_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not ext_cnts:
        return []
    ext_cnt = max(ext_cnts, key=cv2.contourArea)

    # Mask of everything INSIDE the PCB outline (including potential holes)
    inside_pcb = np.zeros((h, w), dtype=np.uint8)
    cv2.drawContours(inside_pcb, [ext_cnt], -1, 255, -1)

    # Paper-like areas that are NOT PCB material, but INSIDE the outline → holes
    non_pcb = cv2.bitwise_not(pcb_mask)
    paper_in_holes = cv2.bitwise_and(cv2.bitwise_and(non_pcb, inside_pcb), paper_color)

    # Clean up: remove tiny specks, close small gaps
    k_morph = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    paper_in_holes = cv2.morphologyEx(paper_in_holes, cv2.MORPH_OPEN, k_morph, iterations=1)
    paper_in_holes = cv2.morphologyEx(paper_in_holes, cv2.MORPH_CLOSE, k_morph, iterations=2)

    # ── Connected-component analysis ──
    nl, labels, stats, centroids = cv2.connectedComponentsWithStats(paper_in_holes, 8)
    min_area_mm2 = 0.5   # 0.5 mm² minimum hole
    max_area_mm2 = 400.0  # 400 mm² maximum (not the whole PCB)
    min_area_px = min_area_mm2 * ppm * ppm
    max_area_px = max_area_mm2 * ppm * ppm

    holes = []
    for i in range(1, nl):
        area_px = stats[i, cv2.CC_STAT_AREA]
        if area_px < min_area_px or area_px > max_area_px:
            continue

        # Get contour of this hole component
        comp_mask = (labels == i).astype(np.uint8) * 255
        hole_cnts, _ = cv2.findContours(comp_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not hole_cnts:
            continue

        hcnt = max(hole_cnts, key=cv2.contourArea)
        area_mm2 = area_px / (ppm * ppm)
        peri = cv2.arcLength(hcnt, True)
        circularity = 4 * math.pi * area_px / (peri * peri) if peri > 0 else 0

        # Simplify polygon
        eps = max(ppm * 0.2, 4.0)
        approx = cv2.approxPolyDP(hcnt, eps, True).reshape(-1, 2)

        cx, cy = centroids[i]

        hole_type = "round" if circularity > 0.7 else "slot" if circularity > 0.3 else "irregular"
        hid = f"paper_hole_{i:02d}"
        holes.append({
            "id": hid,
            "hole_type": hole_type,
            "center": {"x_mm": round(float(cx) / w * w_mm, 3),
                       "y_mm": round(float(cy) / h * h_mm, 3)},
            "polygon": [{"x_mm": round(float(px) / w * w_mm, 3),
                         "y_mm": round(float(py) / h * h_mm, 3)}
                        for px, py in approx.tolist()],
            "confidence": round(min(0.95, 0.5 + 0.5 * circularity + 0.1 * min(area_mm2 / 10.0, 1.0)), 3),
            "area_mm2": round(area_mm2, 2),
            "circularity": round(circularity, 3),
            "source": "paper_match",
        })

    _log.info("Paper holes: %d found (paper model H=[%d,%d] S=[%d,%d])",
              len(holes), paper_model["h_lo"], paper_model["h_hi"],
              paper_model["s_lo"], paper_model["s_hi"])
    return holes


def _validate_notches_by_paper(img, pcb_mask, outline_mm, paper_model,
                                w_mm, h_mm, ppm):
    """Validate edge notches using paper colour matching.

    An edge notch is a concave indentation where paper is visible INSIDE
    the notch.  We check each convexity defect: if the defect interior
    matches paper colour → real notch.  Otherwise → shadow / noise.
    """
    h, w_img = img.shape[:2]
    if paper_model is None:
        return [], None

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    paper_color = cv2.inRange(
        hsv,
        np.array([paper_model["h_lo"], paper_model["s_lo"], paper_model["v_lo"]]),
        np.array([paper_model["h_hi"], paper_model["s_hi"], paper_model["v_hi"]]),
    )

    contours, _ = cv2.findContours(pcb_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return [], "No PCB contour found"
    cnt = max(contours, key=cv2.contourArea)
    n_contour = len(cnt)
    perim = cv2.arcLength(cnt, True)

    hull = cv2.convexHull(cnt, returnPoints=False)
    if len(hull) < 4:
        return [], None

    defects = cv2.convexityDefects(cnt, hull)
    if defects is None:
        return [], None

    min_depth_px = max(ppm * 0.12, 6.0)
    max_seg_ratio = 0.35
    max_seg_px = perim * max_seg_ratio
    min_paper_ratio = 0.25  # At least 25% of notch interior must match paper

    notches = []
    for i in range(defects.shape[0]):
        row = defects[i].flatten()
        s, e, f, d = int(row[0]), int(row[1]), int(row[2]), int(row[3])
        depth = d / 256.0
        if depth < min_depth_px:
            continue

        fwd = (e - s) % n_contour
        bwd = (s - e) % n_contour
        seg_arc = min(fwd, bwd)
        if seg_arc > max_seg_px:
            continue

        depth_mm = depth / ppm
        arc_mm = seg_arc / ppm
        depth_ratio = depth_mm / max(arc_mm, 0.1)
        min_ratio = 0.04 if depth_mm > 0.5 else 0.06
        if depth_ratio < min_ratio:
            continue

        # ── Paper-match validation: is paper visible inside the notch? ──
        # Build a triangle: s, f, e → check paper fill ratio
        sf = cnt[s][0], cnt[f][0]
        with_subscript = False  # check surrounding area for paper
        tri_mask = np.zeros((h, w_img), dtype=np.uint8)
        tri = np.array([[cnt[s][0], cnt[f][0], cnt[e][0]]], dtype=np.int32)
        cv2.fillPoly(tri_mask, tri, 255)

        paper_in_notch = cv2.bitwise_and(tri_mask, paper_color)
        paper_fill = np.sum(paper_in_notch) / max(np.sum(tri_mask), 1)

        if paper_fill < min_paper_ratio:
            _log.debug("Notch rejected: paper_fill=%.2f < %.2f", paper_fill, min_paper_ratio)
            continue

        # Collect contour points
        if s < e:
            groove_pts = cnt[s:e+1, 0, :].tolist()
        else:
            groove_pts = cnt[s:, 0, :].tolist() + cnt[:e+1, 0, :].tolist()
        if len(groove_pts) < 4:
            continue

        segments = np.array(groove_pts, dtype=np.float32).reshape(-1, 1, 2)
        poly_eps = max(ppm * 0.15, 5.0)
        simplified = cv2.approxPolyDP(segments, poly_eps, True).reshape(-1, 2)
        if len(simplified) < 3:
            simplified = np.array(groove_pts[:min(6, len(groove_pts))])

        cx = int(np.mean([p[0] for p in groove_pts]))
        cy = int(np.mean([p[1] for p in groove_pts]))

        idx = len(notches) + 1
        notches.append({
            "id": f"notch_{idx:02d}",
            "groove_type": "groove",
            "polygon": [{"x_mm": round(float(px) / w_img * w_mm, 3),
                         "y_mm": round(float(py) / h * h_mm, 3)}
                        for px, py in simplified.tolist()],
            "center_mm": {"x_mm": round(float(cx) / w_img * w_mm, 3),
                          "y_mm": round(float(cy) / h * h_mm, 3)},
            "depth_mm": round(depth_mm, 2),
            "seg_arc_mm": round(arc_mm, 2),
            "depth_ratio": round(depth_ratio, 3),
            "paper_fill": round(float(paper_fill), 2),
            "confidence": min(0.95, 0.5 + 0.5 * paper_fill),
            "source": "paper_validated",
        })

    warning = None
    if len(notches) > MAX_GROOVES:
        notches.sort(key=lambda g: g.get("confidence", 0), reverse=True)
        notches = notches[:MAX_GROOVES]
        warning = f"检测到{len(notches)}个凹槽，已保留最显著的{MAX_GROOVES}个"

    _log.info("Paper-validated notches: %d found", len(notches))
    return notches, warning


# ═══════════════════════════════════════════════════════════════════════
#  CV outline refinement
# ═══════════════════════════════════════════════════════════════════════

def _refine_outline(binary, w_mm, h_mm, ppm, target_vertices=12):
    """Refine binary mask to polygon — single moderate epsilon preserving all geometry.

    Uses ONE moderately-fine polygon approximation (NOT coarse+fine dual-epsilon):
    - Single eps=0.0008*peri preserves rounded corners (r≥2mm) and all notch features
    - Collinear removal cleans straight-edge noise
    - Vertex dedup removes near-duplicates

    Shadow burrs (one-sided artifacts) are NOT removed here — they are cleaned
    later by _merge_front_back_outlines using front/back consensus:
    if either the front or back doesn't have a burr at the corresponding physical
    location (accounting for horizontal flip), that location is burr-free.

    target_vertices is kept for API compatibility but NOT FORCED.
    """
    h_img, w_img = binary.shape[:2]
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return []

    largest = max(contours, key=cv2.contourArea)
    peri = cv2.arcLength(largest, True)

    # Single epsilon: ~0.0008*peri ≈ 0.15–0.3mm resolution
    # Preserves rounded corners (≥2mm radius) and all real notch features
    eps = peri * 0.0008
    poly = cv2.approxPolyDP(largest, eps, True)
    pts = poly.reshape(-1, 2).astype(np.float32)
    _log.info("Refine: %d vertices (eps=%.4f*peri ≈ %.2fmm)", len(pts), 0.0008, eps / ppm)

    # Remove collinear vertices (straight-edge noise, angle_tol=5°)
    pts = _remove_collinear_vertices(pts, angle_tol_deg=5.0)

    # Deduplicate near-identical vertices (within 0.15mm)
    pts = _dedup_vertices(pts, min_dist_px=max(ppm * 0.15, 3.0))

    # Clockwise order
    if len(pts) >= 3 and cv2.contourArea(pts.reshape(-1, 1, 2)) < 0:
        pts = pts[::-1]

    _log.info("Refine outline: %d vertices final", len(pts))
    return [{"x_mm": round(px / w_img * w_mm, 3), "y_mm": round(py / h_img * h_mm, 3)}
            for px, py in pts.tolist()]


def _dedup_vertices(pts, min_dist_px):
    """Remove near-duplicate vertices, keeping the first of each cluster."""
    if len(pts) <= 3:
        return pts
    result = [pts[0]]
    for pt in pts[1:]:
        last = result[-1]
        dist = np.sqrt((pt[0] - last[0]) ** 2 + (pt[1] - last[1]) ** 2)
        if dist >= min_dist_px:
            result.append(pt)
    # Check if last and first are too close (polygon closure)
    if len(result) >= 3:
        d = np.sqrt((result[-1][0] - result[0][0]) ** 2 +
                     (result[-1][1] - result[0][1]) ** 2)
        if d < min_dist_px:
            result.pop()
    return np.array(result)


def _remove_collinear_vertices(pts, angle_tol_deg=3.0):
    """Remove vertices that are collinear (angle ≈ 180°).

    PCB edges must be straight lines. A vertex on a straight edge is noise.
    Only true corners (sharp bends) should remain.

    Args:
        pts: numpy array of shape (N, 2), polygon vertices in order.
        angle_tol_deg: tolerance in degrees for "almost 180°".

    Returns:
        Filtered numpy array with collinear vertices removed.
    """
    if len(pts) < 4:
        return pts

    n = len(pts)
    keep = [True] * n
    for i in range(n):
        a = pts[(i - 1) % n]
        b = pts[i]
        c = pts[(i + 1) % n]

        # Vectors from b
        v1 = a - b
        v2 = c - b

        len1 = np.linalg.norm(v1)
        len2 = np.linalg.norm(v2)
        if len1 < 1e-6 or len2 < 1e-6:
            keep[i] = False
            continue

        cos_angle = np.clip(np.dot(v1, v2) / (len1 * len2), -1.0, 1.0)
        angle_deg = np.degrees(np.arccos(cos_angle))

        # Angle near 180° means collinear → remove middle vertex
        if angle_deg >= (180.0 - angle_tol_deg):
            keep[i] = False

    result = pts[np.array(keep)]
    if len(result) < 3:
        return pts  # keep at least a triangle

    _log.info("_remove_collinear: %d → %d vertices (angle_tol=%.1f°)",
              len(pts), len(result), angle_tol_deg)
    return result


def refine_outline_geometry(outline_mm):
    """Refine PCB outline while PRESERVING notch/groove features.

    Strategy (in-place refinement, NOT full reconstruction):
    - Points near bbox corners  → replaced by a consistent-radius arc
    - Points along bbox edges   → projected onto straight edge lines
    - Points inside (notch/groove) → kept UNCHANGED (physical features)
    - Edge lines are made symmetric when opposite edges are similar

    Args:
        outline_mm: list of {"x_mm": ..., "y_mm": ...} dicts, ordered polygon.

    Returns:
        Refined outline_mm list (same feature structure, cleaner geometry).
    """
    if len(outline_mm) < 4:
        return outline_mm

    pts = np.array([[p["x_mm"], p["y_mm"]] for p in outline_mm])
    n = len(pts)

    # ── 1. Bounding box → edge positions ──
    min_x, min_y = pts.min(axis=0)
    max_x, max_y = pts.max(axis=0)
    w = max_x - min_x
    h = max_y - min_y
    short = min(w, h)
    if short < 0.5:
        return outline_mm

    top_y, bot_y = float(min_y), float(max_y)
    left_x, right_x = float(min_x), float(max_x)

    # ── 2. Symmetry on edge positions ──
    cx = (left_x + right_x) / 2
    cy = (top_y + bot_y) / 2
    sym_tol = short * 0.10
    if abs(abs(cy - top_y) - abs(bot_y - cy)) < sym_tol:
        half_h = max(abs(cy - top_y), abs(bot_y - cy))
        top_y, bot_y = cy - half_h, cy + half_h
    if abs(abs(cx - left_x) - abs(right_x - cx)) < sym_tol:
        half_w = max(abs(cx - left_x), abs(right_x - cx))
        left_x, right_x = cx - half_w, cx + half_w

    # ── 3. Detect unified corner radius ──
    cr_search = short * 0.15
    sharp_corners = [
        (right_x, top_y), (right_x, bot_y),
        (left_x, bot_y), (left_x, top_y),
    ]
    corner_radii = []
    for sc_x, sc_y in sharp_corners:
        dists = []
        for i in range(n):
            dx = abs(pts[i, 0] - sc_x)
            dy = abs(pts[i, 1] - sc_y)
            if dx < cr_search and dy < cr_search:
                d = math.sqrt(dx**2 + dy**2)
                if d > 0.05:
                    dists.append(d)
        if dists:
            # The consensus merge intersects two slightly-offset rounded
            # corners, producing "cut" corners whose closest point to the sharp
            # corner sits near the tangent point (d ≈ r).  The old
            # median/(sqrt(2)-1) estimator assumed a clean 90° arc and badly
            # over-estimated r for such cut corners (e.g. 0.7 vs true 0.3),
            # which made the corner region swallow real edge points and project
            # them onto an oversized arc — shrinking the board width.
            # min(d) is a conservative lower bound: ≈ r for cut corners, and a
            # mild (harmless) under-estimate for clean arcs — a slightly sharper
            # corner never shrinks the bbox because the tangent points stay on
            # the bounding-box edges.
            r_est = float(min(dists))
            corner_radii.append(min(r_est, short * 0.45))

    r = float(np.median(corner_radii)) if corner_radii else 0.0
    if r < short * 0.02:
        r = 0.0

    # ── 4. Classify each point ──
    def _dist_to_bbox(px, py):
        """Distance from point to nearest bbox side (0 = on boundary)."""
        return min(abs(px - left_x), abs(px - right_x),
                   abs(py - top_y), abs(py - bot_y))

    def _nearest_corner(px, py):
        """Index of nearest sharp corner, or -1 if far from all.
        Threshold is tied to the corner radius r (arc points lie within ~r of
        the sharp corner), so straight-edge points beyond the tangent point are
        never mistaken for corner points."""
        best, best_d = -1, max(r * 1.6, short * 0.03)
        for ci, (scx, scy) in enumerate(sharp_corners):
            d = math.hypot(px - scx, py - scy)
            if d < best_d:
                best, best_d = ci, d
        return best

    # Adaptive edge tolerance: separate detection noise from real notch/groove
    # features. Deviations are typically BIMODAL — a tight cluster near 0 (edge
    # jitter, ~0.02mm) and a deeper cluster (physical notches, ~0.3mm). Split at
    # the largest gap so grooves are never mistaken for noise, even when they
    # outnumber clean-edge points (e.g. recessed top AND bottom edges).
    deviations = []
    for i in range(n):
        if _nearest_corner(pts[i, 0], pts[i, 1]) < 0:
            deviations.append(_dist_to_bbox(pts[i, 0], pts[i, 1]))
    devs = sorted(deviations)
    edge_tol = max(0.05, short * 0.01)  # conservative fallback
    noise = 0.0
    if len(devs) >= 2:
        gaps = [devs[i + 1] - devs[i] for i in range(len(devs) - 1)]
        gi = max(range(len(gaps)), key=lambda k: gaps[k])
        gap, lower_max, upper_min = gaps[gi], devs[gi], devs[gi + 1]
        # Significant gap with a tight noise cluster well below the features
        if gap > max(0.08, short * 0.012) and lower_max < 0.5 * upper_min:
            edge_tol = (lower_max + upper_min) / 2
            noise = lower_max
        else:
            noise = float(np.median(devs))
            edge_tol = max(2.0 * noise, 0.05, short * 0.01)

    # labels: 0..3 = corner region (TR, BR, BL, TL), 4=edge, 5=notch
    labels = []
    for i in range(n):
        px, py = pts[i]
        ci = _nearest_corner(px, py)
        if ci >= 0:
            labels.append(ci)
        elif _dist_to_bbox(px, py) <= edge_tol:
            labels.append(4)   # edge point
        else:
            labels.append(5)   # notch/groove → preserve

    n_notch = labels.count(5)
    _log.info("refine_outline_geometry: r=%.3fmm, noise=%.3fmm, tol=%.3fmm, "
              "%d pts → %d corner / %d edge / %d notch",
              r, noise, edge_tol, n,
              sum(1 for l in labels if l < 4), labels.count(4), n_notch)

    # ── 5. Refine IN PLACE: keep original point order & feature structure ──
    # Corner points  → projected radially onto the ideal arc (consistent radius)
    # Edge points    → projected onto the nearest straight bbox edge line
    # Notch points   → kept exactly as detected (physical groove features)
    # This preserves the outline's traversal direction and groove connectivity
    # (no re-ordering / no inserted arc points that could cross the polygon).
    arc_center = {
        0: (right_x - r, top_y + r),   # TR
        1: (right_x - r, bot_y - r),   # BR
        2: (left_x + r, bot_y - r),    # BL
        3: (left_x + r, top_y + r),    # TL
    }

    arc_span = {0: (-90, 0), 1: (0, 90), 2: (90, 180), 3: (180, 270)}

    def _project_arc(px, py, ci):
        """Radially project a corner point onto the ideal arc circle, clamping
        the angle to the corner's 90° span (a point past the tangent point lands
        on the tangent point, i.e. on the straight edge)."""
        acx, acy = arc_center[ci]
        dx, dy = px - acx, py - acy
        d = math.hypot(dx, dy)
        if d < 1e-6:
            return {"x_mm": round(px, 3), "y_mm": round(py, 3)}
        ang = math.degrees(math.atan2(dy, dx))  # -180..180
        a0, a1 = arc_span[ci]
        # Normalize into [a0-180, a0+180) so angles just outside either arc end
        # clamp to the NEAREST tangent point (never wrap to the opposite side).
        while ang < a0 - 180:
            ang += 360
        while ang >= a0 + 180:
            ang -= 360
        ang = max(a0, min(a1, ang))  # clamp to the arc span
        a = math.radians(ang)
        return {"x_mm": round(acx + r * math.cos(a), 3),
                "y_mm": round(acy + r * math.sin(a), 3)}

    def _project_edge(px, py):
        """Project an edge point onto the nearest bbox edge line."""
        d_l = abs(px - left_x)
        d_r = abs(px - right_x)
        d_t = abs(py - top_y)
        d_b = abs(py - bot_y)
        m = min(d_l, d_r, d_t, d_b)
        if m == d_t:
            return {"x_mm": round(px, 3), "y_mm": round(top_y, 3)}
        elif m == d_b:
            return {"x_mm": round(px, 3), "y_mm": round(bot_y, 3)}
        elif m == d_l:
            return {"x_mm": round(left_x, 3), "y_mm": round(py, 3)}
        else:
            return {"x_mm": round(right_x, 3), "y_mm": round(py, 3)}

    result = []
    for i in range(n):
        lab = labels[i]
        px, py = float(pts[i, 0]), float(pts[i, 1])
        if lab < 4 and r > 0:
            result.append(_project_arc(px, py, lab))
        elif lab == 4:
            result.append(_project_edge(px, py))
        else:
            # Notch/groove (or sharp corner with r==0) → keep unchanged
            result.append({"x_mm": round(px, 3), "y_mm": round(py, 3)})

    # Deduplicate consecutive identical points
    dedup = []
    for p in result:
        if dedup and abs(p["x_mm"] - dedup[-1]["x_mm"]) < 1e-4 and \
           abs(p["y_mm"] - dedup[-1]["y_mm"]) < 1e-4:
            continue
        dedup.append(p)
    if len(dedup) > 1 and abs(dedup[0]["x_mm"] - dedup[-1]["x_mm"]) < 1e-4 and \
       abs(dedup[0]["y_mm"] - dedup[-1]["y_mm"]) < 1e-4:
        dedup.pop()

    return dedup if len(dedup) >= 4 else outline_mm


def _classify_corners(pts, angle_range_near_90=(70, 110)):
    """Classify polygon vertices as right-angle corners vs rounded/irregular.

    Returns list of (x, y, angle_deg, corner_type) dicts.
    corner_type is one of: 'right_angle', 'rounded', 'obtuse', 'acute'.
    """
    if len(pts) < 3:
        return []
    n = len(pts)
    result = []
    for i in range(n):
        a = pts[(i - 1) % n]
        b = pts[i]
        c = pts[(i + 1) % n]

        v1, v2 = a - b, c - b
        l1, l2 = np.linalg.norm(v1), np.linalg.norm(v2)
        if l1 < 1e-6 or l2 < 1e-6:
            result.append({"x": float(b[0]), "y": float(b[1]),
                           "angle_deg": 0.0, "type": "degenerate"})
            continue

        cos_a = np.clip(np.dot(v1, v2) / (l1 * l2), -1.0, 1.0)
        angle_deg = np.degrees(np.arccos(cos_a))

        if angle_range_near_90[0] <= angle_deg <= angle_range_near_90[1]:
            ctype = "right_angle"
        elif angle_deg > angle_range_near_90[1]:
            ctype = "obtuse"
        else:
            ctype = "acute"

        result.append({"x": float(b[0]), "y": float(b[1]),
                       "angle_deg": round(angle_deg, 1), "type": ctype})
    return result


# ═══════════════════════════════════════════════════════════════════════
#  CV groove detection & validation (convexity defects)
# ═══════════════════════════════════════════════════════════════════════

def _validate_grooves(img, pcb_mask, outline_mm, vlm_grooves,
                      w_mm, h_mm, ppm):
    """Validate VLM grooves + CV convexity with curvature-based filtering.

    A real groove is a sharp inward indentation. We distinguish real grooves
    from gentle edge curvature by checking:
      - Segment arc length (≤25% of perimeter — edge curves are long)
      - Depth ratio (depth / arc_width ≥ 0.08)
      - Minimum absolute depth (≥0.25mm)
    """
    h, w_img = pcb_mask.shape[:2]

    # Get pixel contour
    contours, _ = cv2.findContours(pcb_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        warning = f"VLM detected {len(vlm_grooves)} grooves but CV found no PCB contour"
        return vlm_grooves[:MAX_GROOVES], warning if len(vlm_grooves) > MAX_GROOVES else None

    cnt = max(contours, key=cv2.contourArea)
    n_contour = len(cnt)
    perim = cv2.arcLength(cnt, True)

    hull = cv2.convexHull(cnt, returnPoints=False)
    if len(hull) < 4:
        if len(vlm_grooves) > MAX_GROOVES:
            return vlm_grooves[:MAX_GROOVES], f"检测到{len(vlm_grooves)}个凹槽(超过{MAX_GROOVES}上限)，可能存在误识别"
        return vlm_grooves, None

    defects = cv2.convexityDefects(cnt, hull)
    cv_grooves = []

    if defects is not None:
        min_depth_px = max(ppm * 0.12, 6.0)   # ≥0.12mm (was 0.20), min 6px
        max_seg_ratio = 0.35        # max 35% of perimeter (was 25%)
        max_seg_px = perim * max_seg_ratio

        # ── Gather, filter, and merge overlapping defects ──
        raw = []
        for i in range(defects.shape[0]):
            row = defects[i].flatten()
            s, e, f, d = int(row[0]), int(row[1]), int(row[2]), int(row[3])
            depth = d / 256.0
            if depth < min_depth_px:
                continue

            # Arc length of the indentation along contour
            fwd = (e - s) % n_contour
            bwd = (s - e) % n_contour
            seg_arc = min(fwd, bwd)
            if seg_arc > max_seg_px:
                _log.debug("Reject defect: seg_arc=%.0fpx > max=%.0fpx (%.0f%%)",
                           seg_arc, max_seg_px, seg_arc / perim * 100)
                continue

            depth_mm = depth / ppm
            arc_mm = seg_arc / ppm
            depth_ratio = depth_mm / max(arc_mm, 0.1)

            # Depth ratio: a real groove is at least 8% as deep as it is wide
            min_ratio = 0.04 if depth_mm > 0.5 else 0.06   # lowered from 0.06/0.08
            if depth_ratio < min_ratio:
                _log.debug("Reject defect: depth_ratio=%.3f < %.3f", depth_ratio, min_ratio)
                continue

            raw.append({"s": s, "e": e, "f": f, "depth": depth,
                        "depth_mm": depth_mm, "arc_mm": arc_mm,
                        "seg_arc": seg_arc, "ratio": depth_ratio})

        # Merge overlapping defects (share hull edge or are very close)
        merged = []
        used = set()
        for i, di in enumerate(raw):
            if i in used:
                continue
            cluster = [di]
            used.add(i)
            for j, dj in enumerate(raw):
                if j in used:
                    continue
                if (di["s"] == dj["s"] and di["e"] == dj["e"]) or \
                   (di["s"] == dj["e"] and di["e"] == dj["s"]):
                    cluster.append(dj)
                    used.add(j)
                elif abs(di["f"] - dj["f"]) < 30:
                    cluster.append(dj)
                    used.add(j)
            best = max(cluster, key=lambda x: x["depth_mm"])
            merged.append(best)

        _log.info("Groove CV: %d raw defects → %d merged clusters", len(raw), len(merged))

        for idx, di in enumerate(merged):
            s, e, f, depth_mm = di["s"], di["e"], di["f"], di["depth_mm"]

            # Collect contour points between s and e
            if s < e:
                groove_pts = cnt[s:e+1, 0, :].tolist()
            else:
                groove_pts = cnt[s:, 0, :].tolist() + cnt[:e+1, 0, :].tolist()

            if len(groove_pts) < 4:
                continue

            # Simplify polygon to 3-6 points
            segments = np.array(groove_pts, dtype=np.float32).reshape(-1, 1, 2)
            poly_eps = max(ppm * 0.15, 5.0)
            simplified = cv2.approxPolyDP(segments, poly_eps, True).reshape(-1, 2)
            if len(simplified) < 3:
                simplified = np.array(groove_pts[:min(6, len(groove_pts))])

            cx = int(np.mean([p[0] for p in groove_pts]))
            cy = int(np.mean([p[1] for p in groove_pts]))

            cv_grooves.append({
                "id": f"cv_groove_{idx+1:02d}",
                "groove_type": "groove",
                "polygon": [{"x_mm": round(px / w_img * w_mm, 3),
                             "y_mm": round(py / h * h_mm, 3)}
                           for px, py in simplified.tolist()],
                "center_mm": {"x_mm": round(cx / w_img * w_mm, 3),
                              "y_mm": round(cy / h * h_mm, 3)},
                "depth_mm": round(depth_mm, 2),
                "seg_arc_mm": round(di["arc_mm"], 2),
                "depth_ratio": round(di["ratio"], 3),
                "confidence": min(0.9, depth_mm / 3.0),
                "source": "cv_curvature",
            })

    # Merge: VLM grooves first, CV grooves as supplement
    merged_grooves = list(vlm_grooves)
    existing_regions = [_groove_region(g, w_mm, h_mm) for g in vlm_grooves]
    for cvg in cv_grooves:
        cvg_region = _groove_region(cvg, w_mm, h_mm)
        overlap = any(_region_overlap(cvg_region, er) > 0.3 for er in existing_regions)
        if not overlap:
            merged_grooves.append(cvg)
            existing_regions.append(cvg_region)

    # Validate count
    warning = None
    if len(merged_grooves) > MAX_GROOVES:
        merged_grooves.sort(key=lambda g: g.get("confidence", 0), reverse=True)
        merged_grooves = merged_grooves[:MAX_GROOVES]
        warning = f"检测到超过{MAX_GROOVES}个凹槽/凸起（可能存在误识别），已保留最显著的{MAX_GROOVES}个"
    elif len(merged_grooves) == MAX_GROOVES:
        warning = f"检测到{MAX_GROOVES}个凹槽/凸起，请人工确认是否正确"

    _log.info("Groove validation: VLM=%d CV=%d → merged=%d",
              len(vlm_grooves), len(cv_grooves), len(merged_grooves))
    return merged_grooves, warning


def _groove_region(groove, w_mm, h_mm):
    """Get bounding region of a groove for overlap checking."""
    xs = [p["x_mm"] for p in groove.get("polygon", [])]
    ys = [p["y_mm"] for p in groove.get("polygon", [])]
    if not xs:
        return (0, 0, 0, 0)
    return (min(xs), min(ys), max(xs)-min(xs), max(ys)-min(ys))


def _region_overlap(a, b):
    """Compute IoU of two bounding regions."""
    ax1, ay1, aw, ah = a; ax2, ay2 = ax1+aw, ay1+ah
    bx1, by1, bw, bh = b; bx2, by2 = bx1+bw, by1+bh
    ix = max(0, min(ax2,bx2)-max(ax1,bx1))
    iy = max(0, min(ay2,by2)-max(ay1,by1))
    inter = ix*iy; union = aw*ah + bw*bh - inter
    return inter/union if union>0 else 0


# ═══════════════════════════════════════════════════════════════════════
#  CV hole detection
# ═══════════════════════════════════════════════════════════════════════

def _refine_holes_cv(img, mask, vlm_holes, w_mm, h_mm, ppm):
    """Refine VLM holes with CV validation."""
    h,w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Find dark regions inside PCB mask
    dark = cv2.bitwise_not(cv2.adaptiveThreshold(gray,255,
                          cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 10))
    dark = cv2.bitwise_and(dark, mask)

    holes = []
    for vh in vlm_holes:
        poly = vh.get("polygon",[])
        if len(poly)<3: continue
        # Check if region is actually dark (hole should be dark)
        pts = np.array([[[int(p["x_mm"]/w_mm*w), int(p["y_mm"]/h_mm*h)]]
                        for p in poly], np.int32)
        roi = np.zeros((h,w), np.uint8)
        cv2.fillPoly(roi, [pts], 255)
        dark_in_roi = cv2.bitwise_and(dark, roi)
        fill_ratio = np.sum(dark_in_roi) / max(np.sum(roi), 1)
        conf = vh.get("confidence", 0.7) * (0.5 + 0.5*fill_ratio)
        holes.append({**vh, "confidence": round(conf, 3), "fill_ratio": round(fill_ratio, 3)})
    return holes


def _detect_cv_holes(img, mask, w_mm, h_mm, ppm, existing_ids):
    """Detect additional holes via CV that VLM may have missed."""
    h,w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Look for dark blobs inside PCB
    masked = cv2.bitwise_and(gray, mask)
    _, th = cv2.threshold(masked, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    th = cv2.bitwise_and(th, mask)

    # Remove tiny noise
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    th = cv2.morphologyEx(th, cv2.MORPH_OPEN, kernel, 1)

    contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    holes = []
    idx = len(existing_ids)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        min_area = (ppm * 1.5) ** 2  # 1.5mm minimum
        max_area = (ppm * 20) ** 2   # 20mm maximum
        if area < min_area or area > max_area:
            continue

        # Approximate shape
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02*peri, True)
        circularity = 4*math.pi*area/(peri*peri) if peri>0 else 0

        M = cv2.moments(cnt)
        if M["m00"] == 0: continue
        cx, cy = M["m10"]/M["m00"], M["m01"]/M["m00"]

        hole_type = "round" if circularity > 0.7 else "slot" if circularity > 0.3 else "irregular"
        pts = approx.reshape(-1,2).tolist()
        idx += 1
        holes.append({
            "id": f"cv_hole_{idx:02d}",
            "hole_type": hole_type,
            "center": {"x_mm": round(cx/w*w_mm,3), "y_mm": round(cy/h*h_mm,3)},
            "polygon": [{"x_mm": round(px/w*w_mm,3), "y_mm": round(py/h*h_mm,3)}
                       for px,py in pts],
            "confidence": round(min(0.8, 0.5+0.5*circularity), 3),
            "area_px": area,
            "circularity": round(circularity, 3),
            "source": "cv",
        })
    return holes


# ═══════════════════════════════════════════════════════════════════════
#  Utilities
# ═══════════════════════════════════════════════════════════════════════

def _to_b64(img):
    _, buf = cv2.imencode(".png", img)
    return base64.b64encode(buf).decode("ascii")


def _make_transparent(img, mask):
    """Create clean RGBA PNG with edge decontamination and alpha feathering.

    Verified pipeline (pcb_image_pipeline.md), step 4:
      1. Erode mask 1px to strip white-halo edges
      2. Distance transform → 2px soft alpha feather
      3. Colour decontamination (un-premultiply) for clean edges
      4. Output RGBA PNG
    """
    # 1. Erode to remove white-border halo (2px to strip residual edge burrs)
    eroded = cv2.erode(mask, np.ones((3, 3), np.uint8), iterations=2)

    # 2. Distance transform for soft alpha (2px feather)
    dist = cv2.distanceTransform(eroded, cv2.DIST_L2, 5)
    alpha = np.clip(dist / 2.0, 0, 1)

    # 3. Colour decontamination: remove background-tint from edge pixels
    img_f = img.astype(np.float32)
    a = alpha.astype(np.float32)[..., None]
    F = (img_f - (1 - a) * 255.0) / np.clip(a, 0.01, 1.0)
    F = np.clip(F, 0, 255).astype(np.uint8)

    # 4. Assemble RGBA
    rgba = cv2.cvtColor(F, cv2.COLOR_BGR2BGRA)
    rgba[:, :, 3] = (alpha * 255).astype(np.uint8)

    _, buf = cv2.imencode(".png", rgba)
    return buf.tobytes()


def _outline_to_mask(outline_mm, w_mm, h_mm, w_px, h_px):
    """Fill a refined outline polygon into a clean binary mask (no shadow burrs)."""
    mask = np.zeros((h_px, w_px), dtype=np.uint8)
    if not outline_mm or len(outline_mm) < 3:
        return mask
    pts = np.array([
        [round(p["x_mm"] / w_mm * w_px),
         round(p["y_mm"] / h_mm * h_px)]
        for p in outline_mm
    ], dtype=np.int32)
    cv2.fillPoly(mask, [pts], 255)
    return mask


def _mm_to_px(points, w_mm, h_mm, pw, ph):
    return [[int(p["x_mm"]/w_mm*pw), int(p["y_mm"]/h_mm*ph)] for p in points]


# ═══════════════════════════════════════════════════════════════════════
#  Black frame detection (perspective rectification)
# ═══════════════════════════════════════════════════════════════════════

def detect_black_frame(img_buf: bytes, target_aspect: float | None = None) -> dict:
    """Detect the dark rectangular frame border on white paper.

    If `target_aspect` is provided (e.g. frame_w_mm / frame_h_mm), it is used
    to score candidate contours — those matching the expected ratio are preferred.
    If None, aspect scoring is skipped and any large rectangular frame is returned.

    Returns {found, outline, aspect_ratio, avg_width_px, avg_height_px, ...}
    """
    nparr = np.frombuffer(img_buf, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        return {"found": False, "error": "Failed to decode image"}

    h, w = img.shape[:2]

    # ── Otsu + Adaptive hybrid threshold with morphological close ──
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    adaptive = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                     cv2.THRESH_BINARY_INV, 101, 20)
    thresh = cv2.bitwise_or(otsu, adaptive)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, 2)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return {"found": False, "error": "No contours found in image"}

    # When target_aspect is provided, score contours by both area and
    # aspect-ratio match to avoid picking a wrong large blob.
    ASPECT_TOLERANCE = 0.50  # allow up to 50% aspect error

    if target_aspect is not None and target_aspect > 0:
        scored = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 100:
                continue
            hull_tmp = cv2.convexHull(cnt)
            rw_tmp, rh_tmp = cv2.minAreaRect(hull_tmp)[1]
            ca = max(rw_tmp, rh_tmp) / max(min(rw_tmp, rh_tmp), 1)
            aspect_err = abs(ca - target_aspect) / target_aspect
            # Penalise aspect error heavily, reward area moderately
            aspect_score = 1.0 / (1.0 + aspect_err * 3.0)
            score = area * aspect_score
            scored.append((score, cnt, hull_tmp, aspect_err, ca))
        if scored:
            scored.sort(key=lambda x: x[0], reverse=True)
            best_score, largest, hull, aspect_err, best_aspect = scored[0]
            _log.info("Picked contour score=%.0f area=%.0f aspect_err=%.1f%% (n=%d)",
                      best_score, cv2.contourArea(largest), aspect_err * 100, len(scored))
        else:
            largest = max(contours, key=cv2.contourArea)
            hull = cv2.convexHull(largest)
    else:
        largest = max(contours, key=cv2.contourArea)
        hull = cv2.convexHull(largest)

    peri = cv2.arcLength(hull, True)
    corners = cv2.approxPolyDP(hull, 0.02 * peri, True)

    if len(corners) != 4:
        # Fallback: try minAreaRect on hull
        rect = cv2.minAreaRect(hull)
        corners = cv2.boxPoints(rect)
        corners = corners.astype(np.int32)

    # Compute aspect ratio from bounding rect for logging
    rect = cv2.minAreaRect(hull)
    rw, rh = rect[1]
    if target_aspect is None or target_aspect <= 0:
        best_aspect = max(rw, rh) / max(min(rw, rh), 1)

    outline = _order_corners(corners.reshape(-1, 2))

    aspect_ratio = max(rw, rh) / min(rw, rh) if min(rw, rh) > 0 else 1.0

    # Annotated image
    annotated = img.copy()
    cv2.drawContours(annotated, [outline], -1, (0, 255, 0), 3)
    for pt in outline:
        cv2.circle(annotated, tuple(pt), 8, (255, 0, 0), -1)

    return {
        "found": True,
        "outline": [[int(x), int(y)] for x, y in outline],
        "aspect_ratio": round(aspect_ratio, 4),
        "avg_width_px": round(rw, 1),
        "avg_height_px": round(rh, 1),
        "annotated_png_base64": _to_b64(annotated),
    }


def _order_corners(pts: np.ndarray) -> np.ndarray:
    """Order 4 points: top-left, top-right, bottom-right, bottom-left."""
    pts = pts.reshape(4, 2).astype(np.float32)
    rect = np.zeros((4, 2), dtype=np.float32)
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]   # top-left
    rect[2] = pts[np.argmax(s)]   # bottom-right
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]  # top-right
    rect[3] = pts[np.argmax(diff)]  # bottom-left
    return rect.astype(np.int32)


def calibrate_black_frame(img_buf: bytes, frame_w_mm: float,
                          frame_h_mm: float) -> dict:
    """Detect black frame and compute perspective rectification.
    Saves calibration data to disk for later retrieval.
    Returns dict with calibration_id, rectified_png_base64, etc.
    """
    img_buf = _fix_exif_orientation(img_buf)
    nparr = np.frombuffer(img_buf, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Failed to decode image")

    # Detect black frame — use known aspect ratio to help detection
    target_aspect = frame_w_mm / frame_h_mm if frame_h_mm > 0 else None
    frame_result = detect_black_frame(img_buf, target_aspect)
    if not frame_result.get("found"):
        raise ValueError("未检测到黑色方框 — 请确保黑色矩形框清晰可见")

    outline = frame_result["outline"]
    src_pts = np.array(outline, dtype=np.float32)

    # Use actual detected frame pixel dimensions for output resolution
    # Compute edge lengths from the 4-corner outline
    e0 = float(np.linalg.norm(src_pts[1] - src_pts[0]))  # edge p0→p1
    e1 = float(np.linalg.norm(src_pts[2] - src_pts[1]))  # edge p1→p2
    e2 = float(np.linalg.norm(src_pts[3] - src_pts[2]))  # edge p2→p3
    e3 = float(np.linalg.norm(src_pts[0] - src_pts[3]))  # edge p3→p0

    dim1 = (e0 + e2) / 2.0  # average of opposite sides
    dim2 = (e1 + e3) / 2.0  # average of opposite sides

    aspect = frame_w_mm / frame_h_mm
    # Match pixel dimensions to physical dimensions using known aspect ratio
    if abs(dim1 / dim2 - aspect) < abs(dim2 / dim1 - aspect):
        w_px, h_px = dim1, dim2
    else:
        w_px, h_px = dim2, dim1

    source_ppm = min(w_px / frame_w_mm, h_px / frame_h_mm)
    target_ppm = max(min(source_ppm, 60), 10)  # cap: 10–60 px/mm
    rw_img = max(int(frame_w_mm * target_ppm), 200)
    rh_img = max(int(frame_h_mm * target_ppm), 100)

    _log.info("Rectified output: %dx%d px (%.1f px/mm from source %.1f px/mm)",
              rw_img, rh_img, target_ppm, source_ppm)

    dst_pts = np.array([
        [0, 0],
        [rw_img - 1, 0],
        [rw_img - 1, rh_img - 1],
        [0, rh_img - 1]
    ], dtype=np.float32)

    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    rectified = cv2.warpPerspective(img, M, (rw_img, rh_img))

    # Pre-compute pixels_per_mm for use in transparent PCB extraction
    pixels_per_mm = rw_img / frame_w_mm

    # Define cal_id and WORK_ROOT early (needed for saving transparent PNG)
    cal_id = uuid.uuid4().hex  # full 32-char hex UUID
    ROOT = Path(__file__).resolve().parents[1]
    WORK_ROOT = Path(os.getenv("BATTERY_DESIGN_WORKDIR", ROOT / "work"))

    # ── Extract transparent PCB from rectified image (remove white paper + black frame) ──
    transparent_pcb_b64 = ""
    transparent_pcb_outline_mm = []
    try:
        pcb_mask, _ = _extract_pcb_hsv(rectified)
        if pcb_mask is not None and cv2.countNonZero(pcb_mask) > 100:
            # Extract outline in mm for contour comparison
            outline_mm = _refine_outline(pcb_mask, frame_w_mm, frame_h_mm, pixels_per_mm, target_vertices=16)
            transparent_pcb_outline_mm = outline_mm
            # Rebuild clean mask from refined outline (eliminates HSV shadow burrs)
            if len(outline_mm) >= 3:
                clean_mask = _outline_to_mask(outline_mm, frame_w_mm, frame_h_mm, rw_img, rh_img)
                if cv2.countNonZero(clean_mask) > 100:
                    pcb_mask = clean_mask
            transparent_pcb_bytes = _make_transparent(rectified, pcb_mask)
            transparent_pcb_b64 = base64.b64encode(transparent_pcb_bytes).decode("ascii")
            # Save transparent PNG to disk for VLM pad detection
            cal_dir_pre = WORK_ROOT / "calibrations" / cal_id
            cal_dir_pre.mkdir(parents=True, exist_ok=True)
            (cal_dir_pre / "transparent.png").write_bytes(transparent_pcb_bytes)
        else:
            _log.warning("Transparent PCB extraction failed — mask too small or None")
    except Exception:
        _log.warning("Transparent PCB extraction failed", exc_info=True)

    # Annotated original (no green box) — just show corner dots for calibration reference
    annotated = img.copy()
    for pt in src_pts.astype(np.int32):
        cv2.circle(annotated, tuple(pt), 8, (255, 0, 0), -1)

    # Save calibration under work/calibrations/{cal_id}/
    cal_data = {
        "id": cal_id,
        "frame_w_mm": frame_w_mm,
        "frame_h_mm": frame_h_mm,
        "width_mm": frame_w_mm,
        "height_mm": frame_h_mm,
        "pixels_per_mm": round(pixels_per_mm, 3),
        "rectified_w_px": rw_img,
        "rectified_h_px": rh_img,
        "outline_px": [[int(x), int(y)] for x, y in outline],
        "transparent_pcb_outline_mm": transparent_pcb_outline_mm,
    }

    cal_dir = WORK_ROOT / "calibrations" / cal_id
    cal_dir.mkdir(parents=True, exist_ok=True)
    with open(cal_dir / "calibration.json", "w", encoding="utf-8") as f:
        json.dump(cal_data, f, indent=2)
    _, rect_buf = cv2.imencode(".png", rectified)
    (cal_dir / "rectified.png").write_bytes(rect_buf.tobytes())

    _log.info("Black frame calibration: %s → %d×%d px, %.2f px/mm",
              cal_id, rw_img, rh_img, pixels_per_mm)

    return {
        "calibration_id": cal_id,
        "pixels_per_mm": round(pixels_per_mm, 3),
        "rectified_png_base64": _to_b64(rectified),
        "annotated_png_base64": _to_b64(annotated),
        "transparent_pcb_b64": transparent_pcb_b64,
        "transparent_pcb_outline_mm": transparent_pcb_outline_mm,
        "outline": [[int(x), int(y)] for x, y in outline],
        "width_mm": frame_w_mm,
        "height_mm": frame_h_mm,
    }
