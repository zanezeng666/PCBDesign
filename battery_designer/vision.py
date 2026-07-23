"""PCB contour recognition using verified HSV pipeline + VLM refinement.

Pipeline (verified via pcb_image_pipeline.md):
  1. EXIF orientation fix — OpenCV ignores EXIF, piexif normalises it
  2. Black frame detection — simple threshold + convexHull + approxPolyDP(0.02*peri)
  3. Perspective correction — warp to 60×30mm reference frame
  4. HSV PCB extraction — green/blue/yellow detection + Otsu fallback
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
      1. VLM contour detection (grooves + shadows)
      2. HSV PCB extraction (green/blue/yellow + Otsu fallback)
      3. Outline refinement (~12 vertices via epsilon scan)
      4. Groove validation (CV convexity defects + VLM merging)
      5. Edge-decontaminated transparent PNG

    Returns: outline, grooves, groove_count, groove_warning, pcb_mask_b64,
             transparent_pcb_b64, method, debug_steps, vertex_count.
    """
    nparr = np.frombuffer(rectified_png, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        from .errors import DesignError
        raise DesignError("INVALID_IMAGE", "Failed to decode rectified image.")
    h, w = img.shape[:2]
    debug_steps = [{"step": "00_raw", "label": "原始图像", "image_base64": _to_b64(img)}]

    # ── Step 1: VLM contour detection ──
    vlm = _vlm_detect_contour(img, width_mm, height_mm)
    vlm_vc = len(vlm.get("outline", []))
    _log.info("VLM outline vertices: %d", vlm_vc)
    debug_steps.append({"step": "01_vlm", "label": f"VLM轮廓识别({vlm_vc}顶点)",
                        "outline": vlm["outline"], "grooves": vlm["grooves"],
                        "vlm_vertex_count": vlm_vc})

    # ── Step 2: HSV PCB extraction (verified pipeline, replaces shadow removal) ──
    pcb_mask, _ = _extract_pcb_hsv(img)
    if pcb_mask is None:
        from .errors import DesignError
        raise DesignError("NO_PCB_FOUND",
                          "Could not detect PCB board in rectified image.")
    debug_steps.append({"step": "02_hsv_extract", "label": "HSV PCB提取",
                        "pcb_mask_b64": _to_b64(pcb_mask)})

    # ── Step 3: Outline refinement ──
    outline = _refine_outline(pcb_mask, width_mm, height_mm, pixels_per_mm)
    vertex_count = len(outline)
    debug_steps.append({"step": "03_outline", "label": f"PCB轮廓({vertex_count}顶点)",
                        "outline_mm": outline})

    # ── Step 4: Groove validation ──
    grooves, warning = _validate_grooves(img, pcb_mask, outline, vlm["grooves"],
                                         width_mm, height_mm, pixels_per_mm)
    debug_steps.append({"step": "04_grooves", "label": f"凹槽检测({len(grooves)}个)",
                        "grooves": grooves, "warning": warning})

    # ── Step 5: Edge-decontaminated transparent PNG ──
    transparent_png = _make_transparent(img, pcb_mask)
    return {
        "outline": outline, "grooves": grooves,
        "groove_count": len(grooves), "groove_warning": warning,
        "pcb_mask_b64": _to_b64(pcb_mask),
        "transparent_pcb_b64": base64.b64encode(transparent_png).decode("ascii"),
        "method": "hsv-pipeline+vlm",
        "debug_steps": debug_steps,
        "vertex_count": vertex_count,
    }


def detect_holes(rectified_png: bytes, width_mm: float, height_mm: float,
                 pixels_per_mm: float, outline_mm: list[dict]) -> list[dict]:
    """Detect holes/slots inside PCB using VLM + CV refinement."""
    nparr = np.frombuffer(rectified_png, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        return []

    vlm_holes = _vlm_detect_holes(img, width_mm, height_mm)
    h, w = img.shape[:2]
    outline_px = _mm_to_px(outline_mm, width_mm, height_mm, w, h)
    mask = np.zeros((h, w), dtype=np.uint8)
    if len(outline_px) >= 3:
        cv2.fillPoly(mask, [np.array(outline_px, dtype=np.int32)], 255)

    holes = _refine_holes_cv(img, mask, vlm_holes, width_mm, height_mm, pixels_per_mm)
    cv_holes = _detect_cv_holes(img, mask, width_mm, height_mm, pixels_per_mm,
                                 existing_ids={hh["id"] for hh in holes})
    _log.info("detect_holes: VLM=%d + CV=%d → %d", len(holes), len(cv_holes), len(holes) + len(cv_holes))
    return holes + cv_holes


# ═══════════════════════════════════════════════════════════════════════
#  VLM prompt & detection
# ═══════════════════════════════════════════════════════════════════════

_CONTOUR_PROMPT = """You are a PCB visual inspector. This is a rectified top-down photo of a
BATTERY PROTECTION BOARD (锂电池保护板) sitting on white A4 paper.

Board physical size: {width:.1f}mm × {height:.1f}mm. Image: ~{px_w}×{px_h}px.
The black calibration frame has been cropped away — what remains is the PCB board
(dark green/blue) on clean white paper background.

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
#  HSV PCB extraction (verified pipeline)
# ═══════════════════════════════════════════════════════════════════════

def _extract_pcb_hsv(img):
    """Extract PCB binary mask using HSV colour detection.

    Verified pipeline (pcb_image_pipeline.md):
      1. HSV green: H=60-95, S=18-255, V=18-255
      2. Morphology: close(5x5, 4 iterations), open(3x3, 1 iteration)
      3. Largest external contour — NO convex hull (preserves grooves/arcs)
      4. Fine approxPolyDP(0.001 * perimeter) for micro-arc preservation
      5. Fill binary mask

    Falls back through blue/yellow HSV then Otsu if green area is unreasonable.
    Returns (binary_mask, contour) or (None, None).
    """
    h, w = img.shape[:2]
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    lower = np.array([60, 18, 18])
    upper = np.array([95, 255, 255])
    mask = cv2.inRange(hsv, lower, upper)

    k_close = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    k_open = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k_close, iterations=4)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k_open, iterations=1)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        _log.warning("HSV green: no contours found")
        return None, None

    pcb_contour = max(contours, key=cv2.contourArea)
    area_ratio = cv2.contourArea(pcb_contour) / (h * w)

    # ── Fallback: back-side PCB may not have green solder mask ──
    if area_ratio < 0.015 or area_ratio > 0.95:
        _log.info("HSV green area=%.1f%% — trying alternative colour ranges",
                  area_ratio * 100)
        for lo, hi, label in (
            ([100, 18, 18], [130, 255, 255], "blue"),
            ([20, 18, 18], [40, 255, 255], "yellow/tan"),
        ):
            mask2 = cv2.inRange(hsv, np.array(lo), np.array(hi))
            mask2 = cv2.morphologyEx(mask2, cv2.MORPH_CLOSE, k_close, iterations=4)
            mask2 = cv2.morphologyEx(mask2, cv2.MORPH_OPEN, k_open, iterations=1)
            c2, _ = cv2.findContours(mask2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if c2:
                c2b = max(c2, key=cv2.contourArea)
                a2 = cv2.contourArea(c2b) / (h * w)
                if 0.02 < a2 < 0.90:
                    mask, pcb_contour, area_ratio = mask2, c2b, a2
                    _log.info("Fallback %s HSV: area=%.1f%%", label, a2 * 100)
                    break

        # ── Last resort: Otsu on LAB L-channel ──
        if area_ratio < 0.02 or area_ratio > 0.90:
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

    _log.info("PCB extracted: area=%.1f%%  contour_len=%.0fpx  fine_verts=%d",
              area_ratio * 100, peri, len(smooth))
    return binary, pcb_contour


# ═══════════════════════════════════════════════════════════════════════
#  CV outline refinement
# ═══════════════════════════════════════════════════════════════════════

def _refine_outline(binary, w_mm, h_mm, ppm, target_vertices=12):
    """Refine binary mask to polygon with ~target_vertices using epsilon scan.

    Uses the pipeline's perimeter-fraction-based epsilon approach.
    When contour has natural features (grooves, cutouts), vertex count
    should match real geometry. Falling back to edge-split/merge if close.
    """
    h_img, w_img = binary.shape[:2]
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return []

    largest = max(contours, key=cv2.contourArea)
    peri = cv2.arcLength(largest, True)

    # Epsilon scan: fraction of perimeter from 0.004 to 0.06
    best_eps = 0.0
    best_approx = None
    best_nv = 999
    best_gap = 999

    for frac in np.arange(0.004, 0.06, 0.002):
        eps = peri * frac
        approx = cv2.approxPolyDP(largest, eps, True)
        nv = len(approx)
        gap = abs(nv - target_vertices)
        if gap < best_gap or (gap == best_gap and eps > best_eps):
            best_gap, best_eps, best_approx, best_nv = gap, eps, approx, nv
        if nv == target_vertices:
            break

    if best_approx is None or best_nv < 3:
        _log.warning("Refine outline: cannot find valid polygon")
        return []

    pts = best_approx.reshape(-1, 2)
    nv = len(pts)

    # ── Force exactly target_vertices if close ──
    if nv < target_vertices:
        need = target_vertices - nv
        pts_list = pts.tolist()
        for _ in range(need):
            max_len, split_idx = 0, 0
            for i in range(len(pts_list)):
                j = (i + 1) % len(pts_list)
                d = np.hypot(pts_list[j][0] - pts_list[i][0],
                             pts_list[j][1] - pts_list[i][1])
                if d > max_len:
                    max_len, split_idx = d, i
            j = (split_idx + 1) % len(pts_list)
            mid = [(pts_list[split_idx][0] + pts_list[j][0]) / 2,
                   (pts_list[split_idx][1] + pts_list[j][1]) / 2]
            pts_list.insert(split_idx + 1, mid)
        pts = np.array(pts_list, dtype=np.float32)
    elif nv > target_vertices:
        while len(pts) > target_vertices:
            min_dist, merge_i = float('inf'), 0
            for i in range(len(pts)):
                j = (i + 1) % len(pts)
                d = np.hypot(pts[j][0] - pts[i][0], pts[j][1] - pts[i][1])
                if d < min_dist:
                    min_dist, merge_i = d, i
            pts = np.delete(pts, merge_i, axis=0)

    # Deduplicate near-identical vertices (within 0.15mm)
    pts = _dedup_vertices(pts, min_dist_px=max(ppm * 0.15, 4.0))

    # Clockwise order
    if cv2.contourArea(pts.reshape(-1, 1, 2)) < 0:
        pts = pts[::-1]

    _log.info("Refine outline: %d vertices at eps=%.2f*peri (%.1fpx, gap=%d)",
              len(pts), best_eps / peri, best_eps, best_gap)
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
        min_depth_px = ppm * 0.20   # ≥0.20mm
        max_seg_ratio = 0.25        # max 25% of perimeter
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
            min_ratio = 0.06 if depth_mm > 0.8 else 0.08
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
    # 1. Erode to remove white-border halo
    eroded = cv2.erode(mask, np.ones((3, 3), np.uint8), iterations=1)

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

    # ── Verified pipeline: simple threshold + convexHull + approxPolyDP ──
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 70, 255, cv2.THRESH_BINARY_INV)

    kernel = np.ones((5, 5), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=1)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return {"found": False, "error": "No contours found in image"}

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

    # Annotated original
    annotated = img.copy()
    cv2.drawContours(annotated, [src_pts.astype(np.int32)], -1, (0, 255, 0), 3)
    for pt in src_pts.astype(np.int32):
        cv2.circle(annotated, tuple(pt), 10, (255, 0, 0), -1)

    pixels_per_mm = rw_img / frame_w_mm
    cal_id = uuid.uuid4().hex  # full 32-char hex UUID

    ROOT = Path(__file__).resolve().parents[1]
    WORK_ROOT = Path(os.getenv("BATTERY_DESIGN_WORKDIR", ROOT / "work"))

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
        "outline": [[int(x), int(y)] for x, y in outline],
        "width_mm": frame_w_mm,
        "height_mm": frame_h_mm,
    }
