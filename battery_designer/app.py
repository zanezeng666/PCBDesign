from __future__ import annotations

import base64
import json
import logging
import os
import math
import re
import shutil
from uuid import uuid4
from pathlib import Path

import cv2
import numpy as np
import traceback

logger = logging.getLogger(__name__)

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from .catalog import IcCatalog
from .errors import DesignError
from .generator import DesignGenerator
from .kicad import KicadPipeline
from .models import DesignSpec, ValidationRecord
from .storage import ProjectStore
from .vlm_detection import detect_with_vlm as _vlm_detect
from .vlm_detection import detect_all_vlm as _vlm_detect_all
from .vlm_detection import detect_components as _detect_components
from .vlm_detection import verify_pad_crop as _verify_pad_crop
from .vision import extract_pcb as _extract_pcb, detect_holes as _detect_holes
from .vision import _make_transparent, refine_outline_geometry
from .vision import detect_black_frame as _detect_black_frame, calibrate_black_frame as _calibrate_black_frame
from .pcb_recognition import PCBRecognitionPipeline


ROOT = Path(__file__).resolve().parents[1]
WORK_ROOT = Path(os.getenv("BATTERY_DESIGN_WORKDIR", ROOT / "work"))
STATIC_ROOT = ROOT / "web"
DATA_ROOT = ROOT / "data"
store = ProjectStore(WORK_ROOT / "projects")
catalog = IcCatalog(ROOT / "data" / "ic_catalog", WORK_ROOT / "ic_cache")
pipeline = KicadPipeline()
generator = DesignGenerator(pipeline)

app = FastAPI(title="Battery Protection Board Designer", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.mount("/static", StaticFiles(directory=STATIC_ROOT), name="static")
app.mount("/data", StaticFiles(directory=DATA_ROOT), name="data")


@app.middleware("http")
async def add_no_cache_header(request: Request, call_next):
    """Disable browser caching for all responses during development.

    Without this, the browser caches old .js/.css/.html files and
    the user sees stale front-end code after backend fixes.
    """
    response = await call_next(request)
    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    return response


@app.exception_handler(DesignError)
async def design_error_handler(_, exc: DesignError):
    return JSONResponse(status_code=exc.status_code, content=exc.as_dict())


def _json_safe(value):
    """递归净化为可JSON序列化的结构。

    pydantic 的 value_error 类型错误会在 ctx 中携带真实的 ValueError 对象
    （{'ctx': {'error': ValueError(...)}}），直接 json.dumps 会抛
    'Object of type ValueError is not JSON serializable'，导致 500 掩盖真正的
    校验失败原因。这里把异常对象等不可序列化值统一转为字符串。
    """
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, BaseException):
        return str(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


@app.exception_handler(RequestValidationError)
async def validation_error_handler(_, exc: RequestValidationError):
    return JSONResponse(status_code=422, content={"error": {"code": "INPUT_VALIDATION", "message": "Input validation failed.", "details": {"errors": _json_safe(exc.errors())}}})


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Catch all unhandled exceptions and return JSON instead of HTML.

    Without this handler, FastAPI's default error response is an HTML page
    starting with "Internal Server Error", which breaks fetch().json() on
    the frontend (→ "Unexpected token 'I'" parse error).
    """
    tb_lines = traceback.format_exception(type(exc), exc, exc.__traceback__)
    return JSONResponse(
        status_code=500,
        content={
            "error": {
                "code": "INTERNAL_ERROR",
                "message": str(exc),
                "details": {
                    "type": type(exc).__name__,
                    "traceback": "".join(tb_lines[-4:]),  # last 4 frames
                },
            }
        },
    )


@app.get("/")
def index():
    return FileResponse(STATIC_ROOT / "index.html")



@app.get("/api/calibrations/with-recognition")
def list_calibrations_with_recognition():
    """List calibration records that have PCB recognition results.

    A record qualifies when it has both pcb_outline.json (from extract-pcb)
    and terminal-candidates.json (from detect-terminals). Used by the
    standalone Step-3 test page so a full re-recognition is not needed.
    """
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
                import datetime
                mtime = outline_path.stat().st_mtime
                entry["created"] = datetime.datetime.fromtimestamp(mtime).strftime("%Y-%m-%d %H:%M")
            except Exception:
                entry["created"] = ""
            items.append(entry)
    return {"calibrations": items}


@app.get("/api/calibrations/{calibration_id}/recognition")
def get_calibration_recognition(calibration_id: str):
    """Load saved PCB recognition results (outline + terminal candidates).

    Returns the outline polygon and detected pad candidates so the Step-3
    test page can build a DesignSpec without re-running recognition.

    Coordinate note: the saved outline is in full-frame mm, but the pad
    candidates are PCB-RELATIVE (detect-terminals crops the transparent PCB
    to its bounding box before VLM identification). We offset the pad
    coordinates back into the full-frame space here so outline + terminals
    share one coordinate system (required by DesignSpec.point_in_polygon).
    """
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
        # 只有当焦盘确实是 PCB 相对坐标时才加偏移：新版 detect-terminals 会裁剪
        # 透明PCB并记录 origin="pcb_top_left"；旧版数据不裁剪、无该字段，焦盘
        # 本来就是全幅坐标，若再加偏移会被错误地推出轮廓外。
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


@app.get("/api/health")
def health():
    return {"status": "ok", "kicad": pipeline.diagnose(), "resolver_configured": bool(catalog.resolver_endpoint)}


@app.get("/api/calibrations/{calibration_id}/rectified.png")
def get_rectified_image(calibration_id: str):
    """Serve the rectified PNG image for preview in the upload zone."""
    if len(calibration_id) != 32 or any(c not in "0123456789abcdef" for c in calibration_id):
        raise DesignError("INVALID_CALIBRATION_ID", "The calibration id is invalid.")
    rectified_path = WORK_ROOT / "calibrations" / calibration_id / "rectified.png"
    if not rectified_path.exists():
        raise DesignError("CALIBRATION_NOT_FOUND", "Rectified image not found.", {"calibration_id": calibration_id})
    return FileResponse(rectified_path, media_type="image/png")


@app.get("/api/ic/resolve")
def resolve_ic(model: str):
    # 使用查询参数而非路径参数：型号/丝印可能含 "/"（如 3M1B/N607），
    # 作为路径段会被 URL 路由拆分导致 404。
    device = catalog.resolve(model)
    result = device.as_dict()
    # 判断用户输入是否为丝印（而非真实MPN）；丝印可能为多行（如 3M1B/N607，第二行是日期码）
    from .catalog import normalize_mpn, first_marking_line
    requested_full = normalize_mpn(model)
    requested_first = normalize_mpn(first_marking_line(model))
    marking_line = normalize_mpn(device.marking.get("first_line", ""))
    is_marking = bool(marking_line) and requested_full != normalize_mpn(device.full_mpn) and (
        requested_full == marking_line or requested_first == marking_line
    )
    result["resolved_from"] = "marking" if is_marking else "mpn_or_alias"
    result["input_was_marking"] = is_marking
    return result


@app.post("/api/cell/lookup")
async def cell_lookup(request: Request):
    """Look up cell parameters by manufacturer + model using AI.

    Input: {"manufacturer": "Samsung", "model": "INR18650-25R"}
    Output: cell parameters (capacity, voltages, currents, chemistry, etc.)
    """
    body = await request.json()
    manufacturer = str(body.get("manufacturer", "")).strip()
    model = str(body.get("model", "")).strip()
    if not model:
        raise DesignError("INPUT_VALIDATION", "电芯型号不能为空")

    # 缓存文件名消毒："/"、"\\" 是路径分隔符，必须替换，避免误建子目录
    cell_id = f"{manufacturer}_{model}".replace(" ", "_").replace("/", "_").replace("\\", "_")
    cache_path = WORK_ROOT / "cell_cache" / f"{cell_id}.json"
    if cache_path.exists():
        return json.loads(cache_path.read_text(encoding="utf-8"))

    # 先按原始型号查询；若型号含 "/" 且失败，用归一化写法（"/"→"-"）重试。
    # AI 训练数据对连字符写法覆盖更全，例如亿纬 INR18650/26V → INR18650-26V 才能命中。
    params = _ai_cell_lookup(manufacturer, model)
    if params is None and "/" in model:
        alt_model = model.replace("/", "-")
        logger.info("cell lookup: retry with normalized model '%s' -> '%s'", model, alt_model)
        params = _ai_cell_lookup(manufacturer, alt_model)

    if params is None:
        raise DesignError("CELL_LOOKUP_FAILED", "AI 未能获取该电芯的参数信息，请检查型号是否正确")

    # 保留用户输入的原始型号（AI 可能返回归一化写法，如 INR18650-26V）
    params["model"] = model
    params["lookup_model_input"] = {"manufacturer": manufacturer, "model": model}

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(params, ensure_ascii=False, indent=2), encoding="utf-8")
    return params


def _ai_cell_lookup(manufacturer: str, model: str) -> dict | None:
    """Call qwen text model to retrieve lithium cell parameters."""
    try:
        import dashscope
        from dashscope import Generation
    except ImportError:
        logger.warning("dashscope SDK not installed — cell lookup unavailable")
        return None

    from .vision import _get_api_key, _extract_json
    api_key = _get_api_key()
    if not api_key:
        logger.warning("DASHSCOPE_API_KEY not set — cell lookup unavailable")
        return None

    dashscope.api_key = api_key
    prompt = f"""你是锂电池电芯参数数据库。请提供以下电芯用于【保护板（BMS/PCM）设计】的关键参数：
厂商: {manufacturer or '未知'}
型号: {model}

只需要与保护板设计相关的参数（电压/电流/温度保护阈值、内阻、容量），不需要尺寸/重量/认证/运输/寿命等无关信息。
返回纯JSON对象（不要markdown），尽可能填满以下字段：
{{
  "manufacturer": "厂商标称",
  "model": "完整型号",
  "chemistry": "Li-ion 或 LiPo 或 LiFePO4 或 NCA/NCM/LCO 等（决定保护电压阈值）",
  "form_factor": "如 18650 / 21700 / 26650 / 软包",
  "nominal_capacity_mah": 数值,
  "min_capacity_mah": 数值,
  "nominal_voltage_v": 数值,
  "charge_cutoff_voltage_v": 数值,
  "discharge_cutoff_voltage_v": 数值,
  "standard_charge_current_a": 数值,
  "max_charge_current_a": 数值,
  "standard_discharge_current_a": 数值,
  "max_continuous_discharge_a": 数值,
  "max_pulse_discharge_a": 数值,
  "internal_resistance_mohm": 数值,
  "operating_temp_charge_c": [最低, 最高],
  "operating_temp_discharge_c": [最低, 最高],
  "notes": "简要补充说明（注明哪些字段是估计值，保持简短）"
}}

要求：
- 优先采用官方数据手册典型值；手册中不确定的字段，基于同系列/同规格电芯给出合理估计值即可
- 确实无法确定的个别字段填 null 即可，不要因为部分字段未知就返回 error
- 型号中的 "/" 与 "-" 等价（如 INR18650/26V 即 INR18650-26V），两种写法都要尝试识别
- 只要能确认该型号电芯存在（例如能从型号解读出尺寸18650和容量26V≈2600mAh），就必须返回参数对象，不要返回 error
- 仅当型号完全无法识别、明显不存在时，才返回 {{"error": "型号未找到"}}
- 只返回JSON，不要其他文字"""

    try:
        resp = Generation.call(
            model="qwen-plus",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=1024,
            result_format="message",
        )
    except Exception as exc:
        logger.error("Cell lookup API error: %s", exc)
        return None

    if resp.status_code != 200:
        logger.error("Cell lookup API status: %s", getattr(resp, 'code', '?'))
        return None

    try:
        raw = resp.output.choices[0].message.content
        if isinstance(raw, list):
            raw = "".join(p.get("text", "") for p in raw if isinstance(p, dict))
    except Exception as exc:
        logger.error("Cell lookup response parse: %s", exc)
        return None

    parsed = _extract_json(raw)
    if parsed is None or "error" in parsed:
        return None

    # Validate essential fields
    if not parsed.get("nominal_capacity_mah"):
        return None

    parsed["lookup_source"] = "ai"
    parsed["lookup_model_input"] = {"manufacturer": manufacturer, "model": model}
    return parsed


def _estimate_corner_radius(poly_pts, x1, y1, x2, y2):
    """Estimate corner radius from polygon with symmetry constraint.

    PCB pads are symmetric rounded rectangles — all 4 corners should have
    the same radius. For each bbox corner, the closest polygon point on the
    arc is at distance r*(sqrt(2)-1) from the corner.

    Symmetry repair: use median of per-corner radii to reject outliers
    caused by physical damage (e.g. chipped copper on one corner).
    """
    import math
    w = x2 - x1
    h = y2 - y1
    short = min(w, h)
    if short < 2 or len(poly_pts) < 4:
        return max(1, int(short * 0.1))

    corners = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
    radii = []
    search_r = short * 0.45  # only consider points near the corner

    for cx, cy in corners:
        min_d = float('inf')
        for px, py in poly_pts:
            if abs(px - cx) > search_r or abs(py - cy) > search_r:
                continue
            d = math.hypot(px - cx, py - cy)
            if d < min_d:
                min_d = d
        if min_d < float('inf') and min_d > 0:
            r = min_d / (math.sqrt(2) - 1)
            radii.append(r)

    if not radii:
        return max(1, int(short * 0.1))

    # Symmetry constraint: PCB pads have identical radii at all 4 corners.
    # Use median to reject outliers from damaged corners (e.g. chipped copper).
    radii_sorted = sorted(radii)
    n = len(radii_sorted)
    if n >= 3:
        median_r = radii_sorted[n // 2] if n % 2 == 1 else (
            radii_sorted[n // 2 - 1] + radii_sorted[n // 2]) / 2
        # Filter out corners deviating >40% from median (likely damaged)
        good = [r for r in radii if 0.6 * median_r <= r <= 1.4 * median_r]
        if good:
            r = sum(good) / len(good)
        else:
            r = median_r
    else:
        r = sum(radii) / len(radii)

    return max(1, min(int(round(r)), short // 2))


def _draw_rounded_rect(img, x1, y1, x2, y2, radius, color, thickness):
    """Draw a rounded rectangle with the given corner radius."""
    w = x2 - x1
    h = y2 - y1
    r = max(1, min(radius, min(w, h) // 2))
    try:
        cv2.roundRect(img, (x1, y1), (w, h), (r, r), color, thickness)
    except AttributeError:
        # Fallback for OpenCV < 4.6
        if thickness < 0:
            cv2.rectangle(img, (x1 + r, y1), (x2 - r, y2), color, -1)
            cv2.rectangle(img, (x1, y1 + r), (x2, y2 - r), color, -1)
            cv2.circle(img, (x1 + r, y1 + r), r, color, -1)
            cv2.circle(img, (x2 - r, y1 + r), r, color, -1)
            cv2.circle(img, (x1 + r, y2 - r), r, color, -1)
            cv2.circle(img, (x2 - r, y2 - r), r, color, -1)
        else:
            cv2.rectangle(img, (x1 + r, y1), (x2 - r, y2), color, thickness)
            cv2.rectangle(img, (x1, y1 + r), (x2, y2 - r), color, thickness)
            cv2.ellipse(img, (x1 + r, y1 + r), (r, r), 0, 180, 270, color, thickness)
            cv2.ellipse(img, (x2 - r, y1 + r), (r, r), 0, 270, 360, color, thickness)
            cv2.ellipse(img, (x2 - r, y2 - r), (r, r), 0, 0, 90, color, thickness)
            cv2.ellipse(img, (x1 + r, y2 - r), (r, r), 0, 90, 180, color, thickness)


# ── VLM debugging helpers ────────────────────────────────────────────

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


def _refine_positions_cv(result: dict, img_rgba, pixels_per_mm: float) -> dict:
    """Refine VLM pad positions using CV edge detection on the original image.

    VLM is good at identifying WHICH pads exist and their labels, but its
    coordinate precision is limited (~0.002 fractional = ~0.1mm on a 41mm crop).
    This function uses image processing to find the actual metallic boundary
    near each VLM-indicated position and shifts the polygon to match.

    Only position is adjusted — VLM's count, labels, and shape are preserved.
    """
    h_img, w_img = img_rgba.shape[:2]
    has_alpha = len(img_rgba.shape) == 3 and img_rgba.shape[2] == 4

    invalid_indices: set[int] = set()  # candidates to reject after loop

    for idx, cand in enumerate(result.get("candidates", [])):
        regions = cand.get("matched_regions", [])
        if not regions:
            continue
        region = regions[0]
        poly = region.get("polygon", [])
        if len(poly) < 3:
            continue

        # Polygon bbox in pixel coords
        xs = [p["x_mm"] * pixels_per_mm for p in poly]
        ys = [p["y_mm"] * pixels_per_mm for p in poly]
        pad_w = max(xs) - min(xs)
        pad_h = max(ys) - min(ys)
        # ── Label-aware ROI padding ──
        # P+/P-/B+/B-: VLM often traces silkscreen text 1.0–1.5 mm away from
        # the actual metal pad → wide search (80% + 2.0mm).
        # ID/TH/T/NTC/N: moderate search (50% + 1.0mm) — VLM sometimes
        # misplaces them on nearby silkscreen text but the real metal pad
        # is within ~2mm.
        # Others: tight search (30% + 0.5mm).
        label = cand.get("label", "")
        label_up = label.upper()
        if label_up in ("P+", "P-", "B+", "B-"):
            pad_px = int(max(pad_w, pad_h) * 0.80) + int(2.0 * pixels_per_mm)
        elif label_up in ("ID", "TH", "T", "NTC", "N"):
            pad_px = int(max(pad_w, pad_h) * 0.50) + int(1.0 * pixels_per_mm)
        else:
            pad_px = int(max(pad_w, pad_h) * 0.30) + int(0.5 * pixels_per_mm)

        rx1 = max(0, int(min(xs)) - pad_px)
        ry1 = max(0, int(min(ys)) - pad_px)
        rx2 = min(w_img, int(max(xs)) + pad_px + 1)
        ry2 = min(h_img, int(max(ys)) + pad_px + 1)
        if rx2 - rx1 < 10 or ry2 - ry1 < 10:
            continue

        roi = img_rgba[ry1:ry2, rx1:rx2].copy()

        # PCB region mask (alpha >= 128 = solid board).  Non-PCB pixels are
        # excluded from metallic detection so grooves/notches don't pollute it.
        if has_alpha:
            a = roi[:, :, 3]
            pcb_region = (a >= 128).astype(np.uint8) * 255
        else:
            pcb_region = np.full(roi.shape[:2], 255, dtype=np.uint8)

        # Metallic pads are LOW-SATURATION (silver/tin/gold) whereas the green
        # solder mask is highly saturated.  A plain Otsu threshold on grayscale
        # fails here: the dark groove/notch pixels drag the Otsu threshold down
        # so the whole green board gets classified as "bright", the blob becomes
        # the entire PCB and its center collapses to the ROI center (≈ the VLM
        # estimate) — i.e. no refinement at all.  Saturation separates metal
        # (S≈28) from solder mask (S≈224) cleanly; the loose V floor only
        # rejects truly-black parts (component bodies, V<40).
        # Use tighter S threshold (80 vs 100) to reduce background bleed at
        # board edges where white paper has similarly low saturation.
        # Also cap V at 230 to exclude pure-white paper/polygon (V≈240-255,
        # S≈0-10), which otherwise passes the low-saturation check.
        hsv = cv2.cvtColor(roi[:, :, :3], cv2.COLOR_BGR2HSV)
        # V cap raised to 245 (was 230): some metallic pads exhibit V≈225-240
        # under certain lighting, especially tin/solder surfaces.  White paper
        # (V≈245-255) is still safely excluded.
        metallic = cv2.inRange(hsv, np.array([0, 0, 40]), np.array([180, 80, 245]))
        binary = cv2.bitwise_and(metallic, pcb_region)

        # Clean up noise
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)
        # Fill holes (text on pad creates dark holes that shift centroid)
        flood = binary.copy()
        h_f, w_f = flood.shape
        mask_ff = np.zeros((h_f + 2, w_f + 2), np.uint8)
        cv2.floodFill(flood, mask_ff, (0, 0), 255)
        holes = cv2.bitwise_not(flood)
        binary = binary | holes

        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            # ── Wider fallback search ──
            # VLM may have placed the polygon on silkscreen text adjacent to
            # the actual metal pad.  Expand the ROI by 1.5× and retry once.
            fallback_px = int(pad_px * 1.5)
            frx1 = max(0, int(min(xs)) - fallback_px)
            fry1 = max(0, int(min(ys)) - fallback_px)
            frx2 = min(w_img, int(max(xs)) + fallback_px + 1)
            fry2 = min(h_img, int(max(ys)) + fallback_px + 1)
            if frx2 - frx1 > 20 and fry2 - fry1 > 20:
                froi = img_rgba[fry1:fry2, frx1:frx2].copy()
                if has_alpha:
                    fa = froi[:, :, 3]
                    fpcb = (fa >= 128).astype(np.uint8) * 255
                else:
                    fpcb = np.full(froi.shape[:2], 255, dtype=np.uint8)
                fhsv = cv2.cvtColor(froi[:, :, :3], cv2.COLOR_BGR2HSV)
                fmetallic = cv2.inRange(fhsv, np.array([0, 0, 40]), np.array([180, 80, 245]))
                fbinary = cv2.bitwise_and(fmetallic, fpcb)
                fkernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
                fbinary = cv2.morphologyEx(fbinary, cv2.MORPH_OPEN, fkernel, iterations=1)
                fbinary = cv2.morphologyEx(fbinary, cv2.MORPH_CLOSE, fkernel, iterations=2)
                fcontours, _ = cv2.findContours(fbinary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if fcontours:
                    # Found metal in wider search — reuse the expanded ROI context
                    binary, contours = fbinary, fcontours
                    rx1, ry1, rx2, ry2 = frx1, fry1, frx2, fry2
                    roi_area = (rx2 - rx1) * (ry2 - ry1)
                    logger.info("CV refine: %s metal found in wider fallback (+%.0f%%)",
                                cand.get('label', '?'), (fallback_px / max(pad_px, 1) - 1) * 100)
            if not contours:
                logger.info("CV refine: %s NO metallic contour found in ROI — keeping VLM position as-is",
                            cand.get('label', '?'))
                continue

        # Find the largest contour (should be the metallic pad)
        best = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(best)
        poly_area = pad_w * pad_h
        roi_area = (rx2 - rx1) * (ry2 - ry1)

        # ── VLM hallucination / silkscreen pseudo-pad rejection ──
        # If the CV-detected metal contour is far smaller than the VLM-claimed
        # polygon, the region is NOT a real terminal pad.  This catches:
        #  • VLM detecting a silkscreen label outline (rectangular, pad-like)
        #    as a "pad" when only a tiny via dot sits inside
        #  • Pure hallucinations where no metal exists at all
        #
        # Geometric-inferred pads are exempt: they fill a known pattern gap and
        # may land outside the board where no metal is visible.
        is_geometric = any("geometric" in (r.get("source", "") or "")
                           for r in (cand.get("matched_regions") or []))
        area_ratio = area / max(poly_area, 0.01)
        label = cand.get("label", "")

        # ID/TH/T/NTC/N are real auxiliary pads — never reject them.
        # VLM often places their polygon on silkscreen text instead of metal,
        # so CV may find zero metal in the ROI.  Keep the VLM position as-is.
        _AUX_LABELS = {"ID", "TH", "T", "NTC", "N"}

        if not is_geometric and label.upper() not in _AUX_LABELS:
            # Two-tier rejection (only for main power pads):
            # Tier 1: <5%  → pure hallucination (no metal at all)
            # Tier 2: <15% AND polygon > 4 mm² → silkscreen pseudo-pad
            if area_ratio < 0.05:
                logger.warning("CV refine: %s area %.0f < 5%% of poly %.0f — rejecting VLM hallucination",
                               label, area, poly_area)
                invalid_indices.add(idx)
                continue
            if area_ratio < 0.15 and poly_area > 4.0:
                logger.warning(
                    "CV refine: %s metal/poly=%.1f%% (poly=%.1f mm²) — "
                    "silkscreen pseudo-pad (large poly but little metal), rejecting",
                    label, area_ratio * 100, poly_area,
                )
                invalid_indices.add(idx)
                continue

        # Safety: contour must be 25%–400% of VLM polygon area
        if area < poly_area * 0.25 or area > poly_area * 4.0:
            logger.info("CV refine: %s contour area %.0f outside range [%.0f, %.0f], skip",
                        cand.get('label','?'), area, poly_area*0.25, poly_area*4.0)
            continue

        # Safety: if metallic mask fills >80% of ROI, background is bleeding
        # into the detection (happens at board edges or near large metal planes),
        # so the contour center is unreliable — skip.
        metal_pct = float(cv2.countNonZero(binary)) / float(roi_area) if roi_area > 0 else 1.0
        if metal_pct > 0.80:
            logger.info("CV refine: %s metal_mask=%.0f%% of ROI (background bleed), skip",
                        cand.get('label','?'), metal_pct * 100)
            continue
        # If <10% of expected pad area is metallic, the pad is barely
        # visible in this ROI — skip CV refinement but keep the candidate.
        if area < poly_area * 0.10:
            logger.info("CV refine: %s contour area %.0f < 10%% of poly_area %.0f, skip",
                        cand.get('label','?'), area, poly_area)
            continue

        # Use minAreaRect for geometric center (more robust than centroid
        # because it's not affected by asymmetric brightness within the pad)
        rect = cv2.minAreaRect(best)
        cx_px = rect[0][0] + rx1
        cy_px = rect[0][1] + ry1

        # Estimate corner radius from contour vs minAreaRect corners
        # For a rounded rect, contour points near each box corner are inset
        # by r*(sqrt(2)-1) from the sharp corner.
        import math as _math
        box_pts = cv2.boxPoints(rect)  # 4 corners of minAreaRect (in ROI coords)
        contour_pts = best.reshape(-1, 2).astype(float)
        cr_radii = []
        for bp in box_pts:
            dists = np.sqrt((contour_pts[:, 0] - bp[0])**2 +
                            (contour_pts[:, 1] - bp[1])**2)
            min_d = float(dists.min())
            if min_d > 0.5:  # at least 0.5px inset (not a sharp corner)
                cr_radii.append(min_d / (_math.sqrt(2) - 1))
        if len(cr_radii) >= 3:
            cr_sorted = sorted(cr_radii)
            cr_med = cr_sorted[len(cr_sorted) // 2]
            cr_good = [r for r in cr_radii if 0.5 * cr_med <= r <= 1.5 * cr_med]
            cr_px = sum(cr_good) / len(cr_good) if cr_good else cr_med
            cand["_cv_corner_radius_px"] = cr_px

        # VLM polygon center
        cx_vlm = sum(xs) / len(xs)
        cy_vlm = sum(ys) / len(ys)

        # Shift = actual center - VLM center
        dx = cx_px - cx_vlm
        dy = cy_px - cy_vlm
        shift_px = (dx ** 2 + dy ** 2) ** 0.5

        # Safety: max shift is a fraction of pad diagonal.
        # CV should refine, not relocate — but when the CV finds a large,
        # high-confidence metal contour (area_ratio ≥ 0.30) we can trust it
        # more and allow a larger shift.  This is critical for P+/P- pads
        # where VLM often places the polygon on silkscreen text 1.0–1.5mm
        # away from the actual metallic pad.
        img_max_dim = max(h_img, w_img)
        base_factor = 0.60 if area_ratio >= 0.30 else 0.30
        max_shift = min((pad_w ** 2 + pad_h ** 2) ** 0.5 * base_factor,
                        img_max_dim * 0.05)
        logger.info("CV refine: %s VLM=(%.1f,%.1f) CV=(%.1f,%.1f) shift=%.1fpx max=%.1fpx",
                    cand.get('label','?'), cx_vlm, cy_vlm, cx_px, cy_px, shift_px, max_shift)
        if shift_px > max_shift:
            continue

        # Apply shift (position only, preserve VLM shape)
        dx_mm = round(dx / pixels_per_mm, 3)
        dy_mm = round(dy / pixels_per_mm, 3)
        shift_mm = round((dx_mm ** 2 + dy_mm ** 2) ** 0.5, 3)
        cand["_cv_shift_mm"] = shift_mm  # store BEFORE polygon shift for alignment
        for pt in poly:
            pt["x_mm"] = round(pt["x_mm"] + dx_mm, 3)
            pt["y_mm"] = round(pt["y_mm"] + dy_mm, 3)
        center = region.get("center", {})
        if center:
            center["x_mm"] = round(center.get("x_mm", 0) + dx_mm, 3)
            center["y_mm"] = round(center.get("y_mm", 0) + dy_mm, 3)
        bbox = region.get("bbox", {})
        if bbox:
            bbox["x_mm"] = round(bbox.get("x_mm", 0) + dx_mm, 3)
            bbox["y_mm"] = round(bbox.get("y_mm", 0) + dy_mm, 3)
        vp = cand.get("visible_position")
        if vp:
            vp["x_mm"] = round(vp.get("x_mm", 0) + dx_mm, 3)
            vp["y_mm"] = round(vp.get("y_mm", 0) + dy_mm, 3)

        logger.info("CV refine: %s shifted (%.2f, %.2f)mm [%.1fpx]",
                    cand.get("label", "?"), dx_mm, dy_mm, shift_px)

        # −− Store CV-detected tight metal bbox (world mm) for downstream crop−windows −−
        # VLM polygon may be over−sized; the CV contour is the true metal boundary.
        # minAreaRect gives an axis−aligned bounding rectangle of the actual metal.
        cv_w_mm = rect[1][0] / pixels_per_mm
        cv_h_mm = rect[1][1] / pixels_per_mm
        cand["_cv_metal_bbox"] = {
            "x_mm": round(cx_px / pixels_per_mm, 3),
            "y_mm": round(cy_px / pixels_per_mm, 3),
            "w_mm": round(cv_w_mm, 3),
            "h_mm": round(cv_h_mm, 3),
        }

    # ── Remove hallucinated candidates ──
    if invalid_indices:
        original = result.get("candidates", [])
        kept = [c for i, c in enumerate(original) if i not in invalid_indices]
        rejected = [original[i]["label"] for i in sorted(invalid_indices) if i < len(original)]
        logger.warning("CV refine: rejecting %d hallucinated pad(s): %s",
                       len(invalid_indices), rejected)
        result["candidates"] = kept

    return result


def _clamp_pads_to_board(result, pcb_w_mm: float, pcb_h_mm: float):
    """Clamp every candidate's polygon vertices to stay within PCB bounds.

    Geometrically-inferred pads (or pads adjusted by `_align_pad_groups`)
    may land outside the board.  This function truncates the polygon so no
    vertex extends beyond [0, pcb_w]×[0, pcb_h], then recomputes the
    effective centre.

    Pads whose polygon collapses to a degenerate shape (zero area) after
    clamping are removed entirely — they were entirely outside the PCB.
    """
    if pcb_w_mm <= 0 or pcb_h_mm <= 0:
        return

    candidates = result.get("candidates", [])
    rejected: list[int] = []

    for ci, cand in enumerate(candidates):
        regions = cand.get("matched_regions", [])
        region_indices_to_remove: list[int] = []
        for ri, region in enumerate(regions):
            poly = region.get("polygon") or []
            if len(poly) < 3:
                continue
            clamped_any = False
            for v in poly:
                ox = v.get("x_mm")
                oy = v.get("y_mm")
                if ox is not None:
                    cx = max(0.0, min(pcb_w_mm, float(ox)))
                    if abs(cx - float(ox)) > 1e-4:
                        clamped_any = True
                    v["x_mm"] = round(cx, 3)
                if oy is not None:
                    cy = max(0.0, min(pcb_h_mm, float(oy)))
                    if abs(cy - float(oy)) > 1e-4:
                        clamped_any = True
                    v["y_mm"] = round(cy, 3)
            if not clamped_any:
                continue

            # Recompute effective centre from clamped polygon
            xs = [v["x_mm"] for v in poly]
            ys = [v["y_mm"] for v in poly]
            cx = round(sum(xs) / len(xs), 3)
            cy = round(sum(ys) / len(ys), 3)

            # Check for degenerate polygon: zero area after clamping means
            # the entire pad was outside the PCB and got squashed to a line/point.
            w_clamped = round(max(xs) - min(xs), 3)
            h_clamped = round(max(ys) - min(ys), 3)
            if w_clamped < 0.05 or h_clamped < 0.05:
                region_indices_to_remove.append(ri)
                logger.warning(
                    "PCB clamp: %s DEGENERATE polygon (%.3f×%.3fmm) at (%.2f,%.2f) — "
                    "pad was entirely outside PCB, discarding",
                    cand.get("label", "?"), w_clamped, h_clamped, cx, cy,
                )
                continue

            region["center"] = {"x_mm": cx, "y_mm": cy}
            bbox = region.get("bbox", {})
            if bbox:
                bbox["x_mm"] = round(min(xs), 3)
                bbox["y_mm"] = round(min(ys), 3)
                bbox["width_mm"] = w_clamped
                bbox["height_mm"] = h_clamped
            vp = cand.get("visible_position", {})
            if vp:
                vp["x_mm"] = cx
                vp["y_mm"] = cy
            logger.info("PCB clamp: %s polygon truncated to board edge (clamped %.3f×%.3fmm)",
                        cand.get("label", "?"), w_clamped, h_clamped)

        # Remove degenerate regions
        if region_indices_to_remove:
            for ri in reversed(region_indices_to_remove):
                del regions[ri]
            cand["matched_regions"] = regions

        # If all regions were removed, mark candidate for rejection
        if not cand.get("matched_regions"):
            rejected.append(ci)

    # Remove candidates that lost all their regions
    if rejected:
        for ci in reversed(rejected):
            label = candidates[ci].get("label", "?")
            logger.warning("PCB clamp: removing %s — all polygon regions collapsed outside PCB",
                          label)
            del candidates[ci]
        result["candidates"] = candidates


def _align_pad_groups(result, pixels_per_mm=1.0, pcb_w_mm=0.0, pcb_h_mm=0.0):
    """Enforce alignment, uniform size, and even spacing among pad groups.

    PCB layout conventions:
      - Pads in a column (similar X) are vertically aligned → same X center.
      - Pads in a row (similar Y) are horizontally aligned → same Y center.
      - All pads in a group share the SAME width & height.
      - 3+ pads in a group: check for uniform spacing; if roughly even,
        snap to perfectly even spacing.
      - 2 pads: just align center axis + unify size.
      - Ungrouped pads: left untouched.

    The 'regular' majority determines the truth; outliers are corrected.
    """
    import statistics

    candidates = result.get("candidates", [])
    if len(candidates) < 2:
        return result

    # ── Gather pad info ──
    pads: list[dict] = []
    for cand in candidates:
        regions = cand.get("matched_regions", [])
        region = regions[0] if regions else cand.get("visible_region", {})
        poly = region.get("polygon") or []
        center = region.get("center", {})
        if len(poly) < 3 or not center:
            # Fallback: try to reconstruct from width_mm / height_mm
            cx = center.get("x_mm")
            cy = center.get("y_mm")
            cand_w = cand.get("width_mm") or cand.get("visible_region", {}).get("bbox", {}).get("width_mm")
            cand_h = cand.get("height_mm") or cand.get("visible_region", {}).get("bbox", {}).get("height_mm")
            if cx is not None and cy is not None and cand_w and cand_h:
                hw, hh = cand_w / 2, cand_h / 2
                poly = [
                    {"x_mm": cx - hw, "y_mm": cy - hh},
                    {"x_mm": cx + hw, "y_mm": cy - hh},
                    {"x_mm": cx + hw, "y_mm": cy + hh},
                    {"x_mm": cx - hw, "y_mm": cy + hh},
                ]
            else:
                continue
        xs = [p["x_mm"] for p in poly]
        ys = [p["y_mm"] for p in poly]
        w = max(xs) - min(xs)
        h = max(ys) - min(ys)
        pads.append({
            "cand": cand, "region": region, "poly": poly,
            "cx": center.get("x_mm", 0), "cy": center.get("y_mm", 0),
            "w": w, "h": h,
        })

    if len(pads) < 2:
        return result

    # ── Tolerance for grouping ──
    typical = statistics.median([p["w"] for p in pads] + [p["h"] for p in pads])
    if typical < 0.01:
        return result
    tol = typical * 0.6  # generous tolerance for grouping

    # ── Find aligned groups ──
    used: set[int] = set()
    groups: list[tuple[str, list[int]]] = []

    for i in range(len(pads)):
        if i in used:
            continue
        # Vertical group: similar X center (column)
        v_grp = [j for j in range(len(pads))
                 if j not in used and abs(pads[j]["cx"] - pads[i]["cx"]) < tol]
        if len(v_grp) >= 2:
            groups.append(("v", v_grp))
            used.update(v_grp)
            continue
        # Horizontal group: similar Y center (row)
        h_grp = [j for j in range(len(pads))
                 if j not in used and abs(pads[j]["cy"] - pads[i]["cy"]) < tol]
        if len(h_grp) >= 2:
            groups.append(("h", h_grp))
            used.update(h_grp)

    # ── Process each group ──
    for axis, indices in groups:
        grp = [pads[i] for i in indices]
        n = len(grp)

        # 1) Unified dimensions (median)
        med_w = statistics.median([p["w"] for p in grp])
        med_h = statistics.median([p["h"] for p in grp])

        # 2) Unified alignment coordinate
        #    Use median of the group's actual detected positions on the
        #    alignment axis.  (Previously, 2-pad groups were snapped to
        #    PCB center, but that broke cases where aligned pads are
        #    offset from center — e.g. side-by-side terminal columns or
        #    ID+TH pairs near an edge.)
        if axis == "v":
            med_align = statistics.median([p["cx"] for p in grp])
        else:
            med_align = statistics.median([p["cy"] for p in grp])

        # 3) For 3+ pads: check uniform spacing along the other axis
        target_positions = None  # per-pad target along free axis
        if n >= 3:
            # Sort by free-axis position
            if axis == "v":
                ordered = sorted(grp, key=lambda p: p["cy"])
                coords = [p["cy"] for p in ordered]
            else:
                ordered = sorted(grp, key=lambda p: p["cx"])
                coords = [p["cx"] for p in ordered]

            gaps = [coords[i+1] - coords[i] for i in range(n - 1)]
            if len(gaps) == 0:
                continue
            med_gap = statistics.median(gaps)

            # ═══ Enforce minimum spacing: pads within a group MUST NOT overlap ═══
            # The uniform pad size along the free axis is med_h (vertical group)
            # or med_w (horizontal group).  Spacing between centres must be at
            # least the pad size so that adjacent pads do not overlap.
            pad_size_along_free = med_h if axis == "v" else med_w
            if med_gap < pad_size_along_free:
                # Unrealistically tight spacing — likely caused by an outlier
                # dragging the median down.  Fall back to the maximum gap to
                # preserve the larger, more plausible spacing, but never let
                # it drop below the pad size.
                max_gap = max(gaps)
                if max_gap >= pad_size_along_free:
                    med_gap = max_gap
                else:
                    med_gap = pad_size_along_free
                logger.info(
                    "Align(%s): median gap %.3f < pad size %.3f → bumped to %.3f",
                    axis, statistics.median(gaps), pad_size_along_free, med_gap,
                )

            # For 4+ pads in a column/row, try splitting at the spatial
            # midpoint into two natural halves (e.g. top vs bottom for a
            # vertical group). This uses pure spatial proximity — no labels.
            # Each half gets independent uniform spacing.
            clusters = [ordered]
            if n >= 4:
                mid = (coords[0] + coords[-1]) / 2
                top_half = [p for p in ordered
                            if (p["cy"] if axis == "v" else p["cx"]) < mid]
                bot_half = [p for p in ordered
                            if (p["cy"] if axis == "v" else p["cx"]) >= mid]
                if len(top_half) >= 2 and len(bot_half) >= 2:
                    clusters = [top_half, bot_half]

            # For each cluster, enforce uniform spacing independently
            target_positions = {}
            for cluster in clusters:
                cn = len(cluster)
                if cn >= 2:
                    if axis == "v":
                        cl_ordered = sorted(cluster, key=lambda p: p["cy"])
                        cl_coords = [p["cy"] for p in cl_ordered]
                    else:
                        cl_ordered = sorted(cluster, key=lambda p: p["cx"])
                        cl_coords = [p["cx"] for p in cl_ordered]
                    if cn >= 3:
                        cl_gaps = [cl_coords[i+1] - cl_coords[i] for i in range(cn - 1)]
                        cl_med_gap = statistics.median(cl_gaps)

                        # ═══ Cluster-level minimum spacing ═══
                        # Pads within a cluster must not overlap.  The uniform
                        # pad size along the free axis sets the absolute floor.
                        cl_pad_size = med_h if axis == "v" else med_w
                        cl_min_gap = cl_pad_size + 0.15  # at least 0.15 mm visual gap
                        if cl_med_gap < cl_min_gap:
                            # Try max gap first; if it fits the cluster span, use it.
                            cl_max_gap = max(cl_gaps)
                            span = cl_coords[-1] - cl_coords[0]
                            max_fit = span / (cn - 1)
                            if cl_max_gap >= cl_min_gap and cl_max_gap <= max_fit:
                                cl_med_gap = cl_max_gap
                            elif max_fit >= cl_pad_size:
                                # Can't fit with ideal gap, but at least avoid overlap
                                cl_med_gap = max_fit
                            else:
                                # Physically impossible to fit cn pads without overlap.
                                # Keep original positions — uniform spacing would lie.
                                logger.warning(
                                    "Align cluster(%s): can't fit %d pads in span %.2f "
                                    "(pad=%.2f) — keeping original positions",
                                    axis, cn, span, cl_pad_size,
                                )
                                cl_med_gap = 0  # skip uniform spacing

                        if cl_med_gap > 0:
                            start = cl_coords[0]
                            for idx_p, p in enumerate(cl_ordered):
                                target_positions[id(p)] = start + idx_p * cl_med_gap
                    elif cn == 2:
                        actual_mid = (cl_coords[0] + cl_coords[1]) / 2
                        half_gap = abs(cl_coords[0] - cl_coords[1]) / 2
                        first = cl_ordered[0]
                        second = cl_ordered[1]
                        target_positions[id(first)] = actual_mid - half_gap
                        target_positions[id(second)] = actual_mid + half_gap

        # 3b) For exactly 2 pads on the free axis:
        #     Symmetrize about the ACTUAL detected midpoint (optimal fit).
        #     This is the mathematically optimal position: minimizes total
        #     squared displacement from actual detected positions.
        #     Each pad moves toward/away from the actual midpoint along the
        #     free axis only, preserving the detected gap.
        if n == 2:
            if axis == "h":
                # Horizontal pair: free axis is X
                actual_mid = (grp[0]["cx"] + grp[1]["cx"]) / 2
                half_gap = abs(grp[0]["cx"] - grp[1]["cx"]) / 2
                left = grp[0] if grp[0]["cx"] < grp[1]["cx"] else grp[1]
                right = grp[1] if grp[0]["cx"] < grp[1]["cx"] else grp[0]
                target_positions = {
                    id(left): actual_mid - half_gap,
                    id(right): actual_mid + half_gap,
                }
                logger.info("Align(h): 2-pad symmetric about actual mid X=%.3f, gap=%.3f",
                            actual_mid, half_gap * 2)
            elif axis == "v":
                # Vertical pair: free axis is Y
                actual_mid = (grp[0]["cy"] + grp[1]["cy"]) / 2
                half_gap = abs(grp[0]["cy"] - grp[1]["cy"]) / 2
                top = grp[0] if grp[0]["cy"] < grp[1]["cy"] else grp[1]
                bottom = grp[1] if grp[0]["cy"] < grp[1]["cy"] else grp[0]
                target_positions = {
                    id(top): actual_mid - half_gap,
                    id(bottom): actual_mid + half_gap,
                }
                logger.info("Align(v): 2-pad symmetric about actual mid Y=%.3f, gap=%.3f",
                            actual_mid, half_gap * 2)

        # 4) Estimate corner radius: prefer CV contour-based radius (accurate),
        #    fall back to polygon-based estimation, then default.
        import math
        corner_radii_mm = []
        for p in grp:
            # Try CV contour-based radius first (stored during CV refinement)
            cv_cr_px = p["cand"].get("_cv_corner_radius_px")
            if cv_cr_px is not None and cv_cr_px > 0.5:
                corner_radii_mm.append(cv_cr_px / pixels_per_mm)
                continue
            # Fallback: estimate from original polygon
            poly = p["poly"]
            if len(poly) < 4:
                continue
            xs_p = [pt["x_mm"] for pt in poly]
            ys_p = [pt["y_mm"] for pt in poly]
            bx1, by1 = min(xs_p), min(ys_p)
            bx2, by2 = max(xs_p), max(ys_p)
            short = min(bx2 - bx1, by2 - by1)
            if short < 0.1:
                continue
            search_r = short * 0.45
            corners = [(bx1, by1), (bx2, by1), (bx2, by2), (bx1, by2)]
            radii = []
            for ccx, ccy in corners:
                min_d = float('inf')
                for pt in poly:
                    if abs(pt["x_mm"] - ccx) > search_r or abs(pt["y_mm"] - ccy) > search_r:
                        continue
                    d = math.hypot(pt["x_mm"] - ccx, pt["y_mm"] - ccy)
                    if d < min_d:
                        min_d = d
                if 0 < min_d < float('inf'):
                    radii.append(min_d / (math.sqrt(2) - 1))
            if len(radii) >= 3:
                radii_s = sorted(radii)
                med_r = radii_s[len(radii_s) // 2]
                good = [r for r in radii if 0.6 * med_r <= r <= 1.4 * med_r]
                corner_radii_mm.append(sum(good) / len(good) if good else med_r)
            elif radii:
                corner_radii_mm.append(sum(radii) / len(radii))

        # Group unified corner radius (median)
        grp_radius_mm = round(statistics.median(corner_radii_mm), 3) if corner_radii_mm else round(min(med_w, med_h) * 0.1, 3)
        # Clamp to valid range
        grp_radius_mm = max(0.05, min(grp_radius_mm, min(med_w, med_h) / 2))

        # ── CV-refined position preservation ──
        # If CV refine shifted any pad in this group by ≥ 0.25 mm, its
        # position is more trustworthy than VLM's raw coordinate.  Those
        # pads become group anchors — the remaining pads snap to their
        # median, NOT the overall median (which is still polluted by
        # unrefined VLM positions).
        cv_refined_cxs: list[float] = []
        for p in grp:
            shift_mm = p["cand"].get("_cv_shift_mm", 0)
            if shift_mm >= 0.25:
                # Use the CURRENT center (already shifted by CV refine)
                cv_refined_cxs.append(p["cx"])

        if cv_refined_cxs:
            # Recompute alignment anchor from CV-refined positions only
            cv_med = statistics.median(cv_refined_cxs)
            logger.info(
                "Align(%s): using CV-refined anchor X=%.3f (from %d pads) "
                "instead of group median %.3f",
                axis, cv_med, len(cv_refined_cxs), med_align,
            )
            med_align = cv_med

        # 5) Apply corrections to each pad in the group
        #    Position logic:
        #    - Alignment axis: snap to group median (pads are aligned)
        #    - Free axis: keep each pad's own CV-refined center (true position)
        #      OR uniform spacing if 3+ pads are evenly distributed
        #    Shape: reconstruct a perfect symmetric rectangle (med_w × med_h)
        #    centered at the new position. This ensures all pads in the group
        #    have IDENTICAL polygon shapes, eliminating asymmetry from damage.
        hw = med_w / 2
        hh = med_h / 2
        for p in grp:
            # Determine new center (anchor = CV-refined geometric center)
            if axis == "v":
                new_cx = med_align
                new_cy = target_positions.get(id(p), p["cy"]) if target_positions else p["cy"]
            else:
                new_cx = target_positions.get(id(p), p["cx"]) if target_positions else p["cx"]
                new_cy = med_align

            # Reconstruct polygon as perfect symmetric rectangle
            # centered at (new_cx, new_cy) with unified dimensions
            new_poly = [
                {"x_mm": round(new_cx - hw, 3), "y_mm": round(new_cy - hh, 3)},
                {"x_mm": round(new_cx + hw, 3), "y_mm": round(new_cy - hh, 3)},
                {"x_mm": round(new_cx + hw, 3), "y_mm": round(new_cy + hh, 3)},
                {"x_mm": round(new_cx - hw, 3), "y_mm": round(new_cy + hh, 3)},
            ]
            # Replace polygon in-place
            p["poly"].clear()
            p["poly"].extend(new_poly)

            # Update center
            center = p["region"].get("center", {})
            center["x_mm"] = round(new_cx, 3)
            center["y_mm"] = round(new_cy, 3)

            # Update bbox
            bbox = p["region"].get("bbox", {})
            if bbox:
                bbox["x_mm"] = round(new_cx - hw, 3)
                bbox["y_mm"] = round(new_cy - hh, 3)
                bbox["width_mm"] = round(med_w, 3)
                bbox["height_mm"] = round(med_h, 3)

            # Update visible_position
            vp = p["cand"].get("visible_position")
            if vp:
                vp["x_mm"] = round(new_cx, 3)
                vp["y_mm"] = round(new_cy, 3)

            # Update candidate-level dimensions and unified corner radius
            p["cand"]["width_mm"] = round(med_w, 3)
            p["cand"]["height_mm"] = round(med_h, 3)
            p["cand"]["corner_radius_mm"] = grp_radius_mm

            logger.info("Align(%s): %s → center=(%.3f,%.3f) size=%.3fx%.3f",
                        axis, p["cand"].get("label", "?"),
                        new_cx, new_cy, med_w, med_h)

    return result


@app.post("/api/vision/detect-terminals")
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


@app.post("/api/vision/verify-pad-regions")
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


def _cv_find_metallic_pads(transparent_png: bytes, width_mm: float, height_mm: float,
                           pixels_per_mm: float) -> list[dict]:
    """Find metallic solder pads on transparent PCB using CV thresholding.

    Metallic pads are bright (high V) and low saturation (silver/tin/gold).
    Returns list of dicts with center_mm, polygon_mm, area_mm2.
    """
    nparr = np.frombuffer(transparent_png, np.uint8)
    img_rgba = cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)
    if img_rgba is None or len(img_rgba.shape) < 3 or img_rgba.shape[2] != 4:
        return []

    h, w = img_rgba.shape[:2]
    alpha = img_rgba[:, :, 3]
    bgr = img_rgba[:, :, :3]

    # PCB content mask (where alpha > 0)
    pcb_mask = (alpha > 30).astype(np.uint8) * 255

    # Convert to HSV for metallic detection
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)

    # Metallic pads: bright (V > 120) and low saturation (S < 100)
    # This captures silver/tin colored solder pads on the PCB
    metallic_mask = cv2.inRange(hsv, np.array([0, 0, 120]), np.array([180, 100, 255]))

    # Only within PCB area
    metallic_mask = cv2.bitwise_and(metallic_mask, pcb_mask)

    # Morphological cleanup: close gaps within pads, remove tiny noise
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    metallic_mask = cv2.morphologyEx(metallic_mask, cv2.MORPH_CLOSE, kernel, iterations=3)
    metallic_mask = cv2.morphologyEx(metallic_mask, cv2.MORPH_OPEN, kernel, iterations=1)

    # Find contours of metallic regions
    contours, _ = cv2.findContours(metallic_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    pads = []
    min_area_px = (pixels_per_mm * 0.8) ** 2   # min ~0.64mm²
    max_area_px = (pixels_per_mm * 20.0) ** 2   # max 20mm × 20mm

    for cnt in contours:
        area_px = cv2.contourArea(cnt)
        if area_px < min_area_px or area_px > max_area_px:
            continue

        # No aspect ratio filter — battery protection board pads can be elongated strips

        # Compute center in mm
        M = cv2.moments(cnt)
        if M["m00"] == 0:
            continue
        cx_px = M["m10"] / M["m00"]
        cy_px = M["m01"] / M["m00"]
        cx_mm = cx_px / pixels_per_mm
        cy_mm = cy_px / pixels_per_mm

        # Use minAreaRect to get a clean rotated rectangle (pads are rounded rects)
        rect = cv2.minAreaRect(cnt)  # ((cx,cy), (w,h), angle)
        box_pts = cv2.boxPoints(rect)  # 4 corner points
        polygon_mm = [
            {"x_mm": round(float(pt[0]) / pixels_per_mm, 3),
             "y_mm": round(float(pt[1]) / pixels_per_mm, 3)}
            for pt in box_pts
        ]

        # Bounding box (axis-aligned) for display
        bx, by, bw, bh = cv2.boundingRect(cnt)
        pads.append({
            "center_mm": (round(cx_mm, 3), round(cy_mm, 3)),
            "polygon_mm": polygon_mm,
            "area_mm2": round(area_px / (pixels_per_mm ** 2), 2),
            "bbox_mm": {
                "x_mm": round(bx / pixels_per_mm, 3),
                "y_mm": round(by / pixels_per_mm, 3),
                "width_mm": round(bw / pixels_per_mm, 3),
                "height_mm": round(bh / pixels_per_mm, 3),
            },
        })

    # Sort by X position for consistent ordering
    pads.sort(key=lambda p: p["center_mm"][0])
    return pads


def _match_vlm_to_cv(candidates: list[dict], cv_pads: list[dict]):
    """Match VLM candidates to CV-detected metallic pads using optimal assignment.

    VLM tells us HOW MANY pads there are and their approximate positions.
    CV gives precise locations of metallic regions.
    We find the best 1-to-1 assignment: each VLM candidate → one unique CV pad.
    Uses greedy global-optimal: pick the best (closest) pair first, repeat.
    """
    if not candidates or not cv_pads:
        return

    # Build distance matrix: candidates × cv_pads
    n_cand = len(candidates)
    n_pads = len(cv_pads)

    # Compute adaptive match threshold from spatial extent of all points
    all_xs = [c.get("visible_region", {}).get("center", {}).get("x_mm", 0) for c in candidates]
    all_ys = [c.get("visible_region", {}).get("center", {}).get("y_mm", 0) for c in candidates]
    for pad in cv_pads:
        all_xs.append(pad["center_mm"][0])
        all_ys.append(pad["center_mm"][1])
    if all_xs and all_ys:
        spatial_spread = max(max(all_xs) - min(all_xs), max(all_ys) - min(all_ys), 1.0)
        max_match_dist = spatial_spread * 0.25  # 25% of spatial extent
    else:
        max_match_dist = 15.0

    # Compute all (distance, cand_idx, pad_idx) pairs
    pairs = []
    for ci, cand in enumerate(candidates):
        vr = cand.get("visible_region", {})
        vlm_center = vr.get("center", {})
        vlm_x = vlm_center.get("x_mm", 0)
        vlm_y = vlm_center.get("y_mm", 0)
        for pi, pad in enumerate(cv_pads):
            px, py = pad["center_mm"]
            dist = ((px - vlm_x) ** 2 + (py - vlm_y) ** 2) ** 0.5
            pairs.append((dist, ci, pi))

    # Sort by distance (best matches first)
    pairs.sort(key=lambda x: x[0])

    # Greedy assignment: pick best pair, mark both used, repeat
    used_cands = set()
    used_pads = set()
    assignments = {}  # cand_idx → pad_idx

    for dist, ci, pi in pairs:
        if ci in used_cands or pi in used_pads:
            continue
        if dist > max_match_dist:  # too far, not a valid match
            break
        assignments[ci] = pi
        used_cands.add(ci)
        used_pads.add(pi)
        if len(assignments) == n_cand:
            break

    # Apply assignments: replace VLM coords with CV-precise coords
    for ci, pi in assignments.items():
        cand = candidates[ci]
        pad = cv_pads[pi]
        vr = cand.get("visible_region", {})
        vr["center"] = {"x_mm": pad["center_mm"][0], "y_mm": pad["center_mm"][1]}
        vr["polygon"] = pad["polygon_mm"]
        vr["bbox"] = pad["bbox_mm"]
        vr["source"] = "cv_refined"
        if cand.get("visible_position"):
            cand["visible_position"] = {"x_mm": pad["center_mm"][0], "y_mm": pad["center_mm"][1]}
        if cand.get("matched_regions"):
            cand["matched_regions"] = [vr]


# ── Qwen VLM + CV 轮廓识别端点 ────────────────────────────────────────────

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


@app.post("/api/vision/extract-pcb")
def api_extract_pcb(calibration_id: str = Form(...),
                    other_calibration_id: str = Form(default=""),
                    other_outline_json: str = Form(default="")):
    """Extract PCB outline from white A4 paper with shadow removal.

    If other_calibration_id is provided (other PCB side), the outlines are
    cross-validated using mask consensus to remove one-sided edge artifacts
    (e.g. burrs visible only in one photo).

    If other_outline_json is provided directly, it takes priority over
    reading pcb_outline.json from disk (avoids race condition in parallel calls).

    The returned outline replaces the raw CV result with the consensus contour.
    """
    png, w_mm, h_mm, ppm = _load_calibration(calibration_id)
    result = _extract_pcb(png, w_mm, h_mm, ppm)

    # The mask returned by _extract_pcb is derived from the HSV extraction and
    # correctly excludes grooves/notches (white paper visible through them).
    # The consensus merge below simplifies the outline polygon and loses that
    # groove detail, so fillPoly on the merged outline fills the groove opaque
    # and the white photo background shows through in the preview.  Keep the
    # original mask to punch the grooves back through the rebuilt display masks.
    orig_mask = None
    if result.get("pcb_mask_b64"):
        try:
            orig_mask = cv2.imdecode(
                np.frombuffer(base64.b64decode(result["pcb_mask_b64"]), np.uint8),
                cv2.IMREAD_GRAYSCALE)
        except Exception:
            orig_mask = None

    # ── Cross-validate with other side (if available) ──
    consensus_outline = None
    consensus_msg = ""
    other_outline = None
    other_w_mm, other_h_mm, other_ppm = w_mm, h_mm, 0.0

    # Priority 1: directly provided outline JSON (avoids file race condition)
    if other_outline_json:
        try:
            od = json.loads(other_outline_json)
            if isinstance(od, dict):
                other_outline = od.get("outline", [])
                other_w_mm = float(od.get("frame_w_mm", w_mm))
                other_h_mm = float(od.get("frame_h_mm", h_mm))
                other_ppm = float(od.get("pixels_per_mm", 0.0))
            elif isinstance(od, list):
                other_outline = od
        except Exception:
            logger.warning("Failed to parse other_outline_json", exc_info=True)

    # Priority 2: read from other calibration's pcb_outline.json on disk
    if (not other_outline or len(other_outline) < 3) and other_calibration_id:
        other_dir = WORK_ROOT / "calibrations" / other_calibration_id
        other_outline_path = other_dir / "pcb_outline.json"
        other_meta_path = other_dir / "calibration.json"
        if other_outline_path.exists() and other_meta_path.exists():
            try:
                other_data = json.loads(other_outline_path.read_text(encoding="utf-8"))
                other_outline = other_data.get("outline", [])
                other_meta = json.loads(other_meta_path.read_text(encoding="utf-8"))
                other_w_mm = float(other_meta.get("frame_w_mm", w_mm))
                other_h_mm = float(other_meta.get("frame_h_mm", h_mm))
                # Prefer the ppm stored with the outline (the value actually used
                # to build it); fall back to the calibration metadata.
                other_ppm = float(other_data.get("pixels_per_mm", 0.0) or
                                  other_meta.get("pixels_per_mm", 0.0))
            except Exception:
                logger.warning("extract-pcb: failed to read other side outline",
                               exc_info=True)

    # ── Merge if valid other outline available ──
    if isinstance(other_outline, list) and len(other_outline) >= 3:
        try:
            merge_w = max(w_mm, other_w_mm)
            merge_h = max(h_mm, other_h_mm)

            # 构建正面结果供交叉校验
            front_result = {
                "outline": result["outline"],
                "pixels_per_mm": ppm,
                "width_mm": w_mm,
                "height_mm": h_mm,
                "rectified_png_b64": result.get("rectified_png_b64", ""),
            }
            back_result = {
                "outline": other_outline,
                "pixels_per_mm": other_ppm,
                "width_mm": other_w_mm,
                "height_mm": other_h_mm,
                "rectified_png_b64": "",
            }

            cross_result = PCBRecognitionPipeline.cross_validate_front_back(
                front_result, back_result
            )
            consensus_outline = cross_result["outline"]
            result["outline"] = consensus_outline
            result["transparent_pcb_b64"] = cross_result["transparent_pcb_b64"]
            consensus_msg = f" (正反面对照纠正, {len(consensus_outline)}顶点)"

            logger.info("extract-pcb consensus: %s corrected via other side",
                        calibration_id)
        except Exception:
            logger.warning("extract-pcb consensus failed for %s",
                           calibration_id, exc_info=True)

    # ── Geometry refinement: straighten edges, unify corners, enforce symmetry ──
    try:
        raw_outline = result["outline"]
        refined = refine_outline_geometry(raw_outline)
        if len(refined) >= 3:
            result["outline"] = refined
            result["outline_raw"] = raw_outline  # preserve original for reference
            logger.info("extract-pcb: outline refined %d → %d vertices (geometry)",
                        len(raw_outline), len(refined))
            # Rebuild transparent PNG from refined outline
            try:
                nparr3 = np.frombuffer(png, np.uint8)
                img3 = cv2.imdecode(nparr3, cv2.IMREAD_COLOR)
                h3, w3 = img3.shape[:2]
                outline_px3 = np.array([
                    [round(p["x_mm"] / w_mm * w3),
                     round(p["y_mm"] / h_mm * h3)]
                    for p in refined
                ], dtype=np.int32)
                new_mask3 = np.zeros((h3, w3), dtype=np.uint8)
                cv2.fillPoly(new_mask3, [outline_px3], 255)
                # Punch grooves back through (same reason as consensus rebuild).
                if orig_mask is not None and orig_mask.shape == new_mask3.shape:
                    new_mask3 = cv2.bitwise_and(new_mask3, orig_mask)
                result["pcb_mask_b64"] = base64.b64encode(
                    cv2.imencode(".png", new_mask3)[1]).decode("ascii")
                result["transparent_pcb_b64"] = base64.b64encode(
                    _make_transparent(img3, new_mask3)).decode("ascii")
            except Exception:
                logger.warning("extract-pcb: refined transparent rebuild failed",
                               exc_info=True)
    except Exception:
        logger.warning("extract-pcb: geometry refinement failed", exc_info=True)

    # ── Save to disk ──
    directory = WORK_ROOT / "calibrations" / calibration_id
    (directory / "pcb_outline.json").write_text(
        json.dumps({"outline": result["outline"], "grooves": result["grooves"],
                     "pixels_per_mm": ppm}, ensure_ascii=False, indent=2),
        encoding="utf-8")

    # Save transparent PNG to disk (updated with paper shadow removal)
    if result.get("transparent_pcb_b64"):
        (directory / "transparent.png").write_bytes(base64.b64decode(result["transparent_pcb_b64"]))

    if consensus_msg:
        result["consensus_msg"] = consensus_msg

    return result


@app.post("/api/vision/contour-match")
def api_contour_match(front_calibration_id: str = Form(...),
                      back_calibration_id: str = Form(...)):
    """Run Edge Chamfer Distance contour matching between front and back PCB outlines.

    This is the same algorithm used by /api/simulate, exposed as a standalone
    endpoint so the upload-mode flow can also get Chamfer-based cross-validation.
    """
    try:
        front_meta = json.loads(
            (WORK_ROOT / "calibrations" / front_calibration_id / "calibration.json")
            .read_text(encoding="utf-8"))
        back_meta = json.loads(
            (WORK_ROOT / "calibrations" / back_calibration_id / "calibration.json")
            .read_text(encoding="utf-8"))
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=f"Calibration not found: {e.filename}")

    # Prefer the latest extract-pcb outline (already consensus-corrected if
    # both sides were available), fall back to the calibration-time raw outline.
    front_ol = front_meta.get("transparent_pcb_outline_mm", [])
    back_ol = back_meta.get("transparent_pcb_outline_mm", [])
    front_pcb_path = WORK_ROOT / "calibrations" / front_calibration_id / "pcb_outline.json"
    back_pcb_path = WORK_ROOT / "calibrations" / back_calibration_id / "pcb_outline.json"
    try:
        if front_pcb_path.exists():
            front_pcb = json.loads(front_pcb_path.read_text(encoding="utf-8"))
            if isinstance(front_pcb.get("outline"), list) and len(front_pcb["outline"]) >= 3:
                front_ol = front_pcb["outline"]
        if back_pcb_path.exists():
            back_pcb = json.loads(back_pcb_path.read_text(encoding="utf-8"))
            if isinstance(back_pcb.get("outline"), list) and len(back_pcb["outline"]) >= 3:
                back_ol = back_pcb["outline"]
    except Exception:
        pass  # Fall back to calibration-time outlines

    if not isinstance(front_ol, list) or len(front_ol) < 3:
        raise HTTPException(status_code=400, detail="Front outline data missing or incomplete")
    if not isinstance(back_ol, list) or len(back_ol) < 3:
        raise HTTPException(status_code=400, detail="Back outline data missing or incomplete")

    w_mm = max(front_meta.get("frame_w_mm", 40.0), back_meta.get("frame_w_mm", 40.0))
    h_mm = max(front_meta.get("frame_h_mm", 25.0), back_meta.get("frame_h_mm", 25.0))

    steps = [
        {"side": "front", "calibration_success": True,
         "transparent_pcb_outline_mm": front_ol,
         "frame_w_mm": w_mm, "frame_h_mm": h_mm,
         "pixels_per_mm": front_meta.get("pixels_per_mm", 0.0)},
        {"side": "back", "calibration_success": True,
         "transparent_pcb_outline_mm": back_ol,
         "frame_w_mm": w_mm, "frame_h_mm": h_mm,
         "pixels_per_mm": back_meta.get("pixels_per_mm", 0.0)},
    ]
    result = _compare_front_back_contours(steps)
    return result


@app.post("/api/vision/detect-holes")
def api_detect_holes(calibration_id: str = Form(...),
                     outline_json: str = Form(default="")):
    """Detect holes/slots and edge grooves inside the PCB area.

    If outline_json is empty, loads from pcb_outline.json (saved by extract-pcb).
    """
    png, w_mm, h_mm, ppm = _load_calibration(calibration_id)

    if outline_json:
        outline_data = json.loads(outline_json)
        outline_mm = outline_data.get("outline", outline_data) if isinstance(outline_data, dict) else outline_data
    else:
        pcb_outline_path = WORK_ROOT / "calibrations" / calibration_id / "pcb_outline.json"
        if pcb_outline_path.exists():
            outline_data = json.loads(pcb_outline_path.read_text(encoding="utf-8"))
            outline_mm = outline_data.get("outline", [])
        else:
            # Fallback: use calibration outline
            metadata = json.loads((WORK_ROOT / "calibrations" / calibration_id / "calibration.json").read_text(encoding="utf-8"))
            outline_mm = metadata.get("outline", [])

    if not outline_mm or len(outline_mm) < 3:
        raise DesignError("OUTLINE_NOT_FOUND",
                          "PCB outline not found. Run extract-pcb first.")

    holes = _detect_holes(png, w_mm, h_mm, ppm, outline_mm)
    directory = WORK_ROOT / "calibrations" / calibration_id
    (directory / "holes.json").write_text(
        json.dumps({"holes": holes}, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"holes": holes, "hole_count": len(holes)}


@app.post("/api/vision/detect-all")
def api_detect_all(calibration_id: str = Form(...), side: str = Form(...)):
    """Unified detection: PCB outline + holes + pads (VLM + CV).

    Combines:
      - vision.extract_pcb()  for outline & grooves
      - vlm_detection.detect_all_vlm() for pad terminals
    """
    if side not in ("front", "back"):
        raise DesignError("INVALID_BOARD_SIDE", "side must be front or back")

    png, w_mm, h_mm, ppm = _load_calibration(calibration_id)

    # Run contour extraction
    contour = _extract_pcb(png, w_mm, h_mm, ppm)

    # Run pad detection via VLM
    pad_result = _vlm_detect_all(png, w_mm, h_mm, side)

    # Merge — CV-refined outline takes priority over VLM-only outline
    result = {
        **pad_result,
        **contour,
        "combined_method": "qwen-vlm+cv_unified",
    }

    directory = WORK_ROOT / "calibrations" / calibration_id
    (directory / "detection-all.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


@app.post("/api/vision/detect-components")
def api_detect_components(calibration_id: str = Form(...), side: str = Form(...)):
    """Detect electronic components on the PCB using the original rectified image.

    Uses the unmodified rectified PNG (not cropped/transparent) to preserve
    full visual context for reading tiny silkscreen text on ICs, MOSFETs,
    resistors, capacitors, etc.

    Returns:
        - components: list of detected components with type/silkscreen/package/position
        - inferred_ic: auto-inferred IC model from detected IC silkscreen
        - inferred_mos: auto-inferred MOS model from detected MOSFET silkscreen
    """
    if side not in ("front", "back"):
        raise DesignError("INVALID_BOARD_SIDE", "side must be front or back")

    # Load calibration metadata and rectified PNG (original image, not transparent)
    png, w_mm, h_mm, ppm = _load_calibration(calibration_id)

    # Call VLM component detection on the original rectified image
    result = _detect_components(png, w_mm, h_mm, side)

    # Save result
    directory = WORK_ROOT / "calibrations" / calibration_id
    (directory / "components.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    return result


@app.post("/api/projects", status_code=201)
def create_project(spec: DesignSpec):
    alignment = _validate_photo_pair(spec)
    if alignment is not None:
        spec.photo_capture.alignment_error_mm = alignment
    device = catalog.resolve(spec.protection_ic)
    generator.preflight(spec, device)
    project_id, directory = store.create({"spec": spec.model_dump(mode="json"), "device": device.as_dict(), "approved": False})
    photo_dir = directory / "photos"
    for side, calibration_id in (("front", spec.photo_capture.front_calibration_id), ("back", spec.photo_capture.back_calibration_id)):
        if calibration_id:
            shutil.copytree(WORK_ROOT / "calibrations" / calibration_id, photo_dir / side)
    return {
        "id": project_id,
        "status": "created",
        "port_topology": spec.port_topology,
        "photo_alignment_error_mm": spec.photo_capture.alignment_error_mm,
        "device": device.as_dict(),
        "directory": str(directory),
    }


def _validate_photo_pair(spec: DesignSpec) -> float | None:
    capture = spec.photo_capture
    if not capture.front_calibration_id and not capture.back_calibration_id:
        return None
    if not capture.front_calibration_id:
        raise DesignError("FRONT_PHOTO_REQUIRED", "A front calibration is required when photo calibration ids are used.")

    def load(calibration_id: str) -> dict:
        path = WORK_ROOT / "calibrations" / calibration_id / "calibration.json"
        if not path.exists():
            raise DesignError("CALIBRATION_NOT_FOUND", "Photo calibration record not found.", {"calibration_id": calibration_id})
        return json.loads(path.read_text(encoding="utf-8"))

    load(capture.front_calibration_id)
    if capture.back_calibration_id:
        load(capture.back_calibration_id)
    return None


@app.get("/api/projects/{project_id}")
def get_project(project_id: str):
    return store.read(project_id)


@app.post("/api/projects/{project_id}/preview")
def generate_preview(project_id: str):
    project = store.read(project_id)
    spec = DesignSpec.model_validate(project["spec"])
    device = catalog.resolve(project["device"]["full_mpn"])
    result = generator.generate_preview(spec, device, store.directory(project_id))
    store.update(project_id, status="preview_ready", preview=result)
    return result


@app.post("/api/projects/{project_id}/approve-candidate")
def approve_candidate(project_id: str):
    return store.update(project_id, approved=True, approval_notice="Candidate sample build approved; not approved for mass production.")


@app.post("/api/projects/{project_id}/manufacturing")
def generate_manufacturing(project_id: str):
    project = store.read(project_id)
    spec = DesignSpec.model_validate(project["spec"])
    device = catalog.resolve(project["device"]["full_mpn"])
    result = generator.generate_manufacturing(spec, device, store.directory(project_id), approved=bool(project.get("approved")))
    store.update(project_id, status="manufacturing_ready", manufacturing=result)
    return result


@app.post("/api/projects/{project_id}/validation")
def record_validation(project_id: str, record: ValidationRecord):
    if not record.passed:
        raise DesignError("HARDWARE_VALIDATION_FAILED", "All required hardware tests must pass before validation.")
    project = store.read(project_id)
    device = catalog.resolve(project["device"]["full_mpn"])
    promoted = catalog.promote(device, record.model_dump(mode="json"))
    store.write_json(project_id, "hardware-validation.json", record.model_dump(mode="json"))
    return store.update(project_id, hardware_validated=True, validation=record.model_dump(mode="json"), device=promoted.as_dict())


@app.get("/api/projects/{project_id}/artifacts/{artifact_path:path}")
def artifact(project_id: str, artifact_path: str):
    return FileResponse(store.artifact(project_id, artifact_path))


# ── Simulate / Black Frame Preview ───────────────────────────────────────

def _resample_polygon_perimeter(pts, n):
    """Resample polygon to exactly n evenly-spaced points along its perimeter."""
    if len(pts) < 2:
        return pts
    # Compute cumulative edge lengths
    seg_lens = []
    for i in range(len(pts)):
        j = (i + 1) % len(pts)
        seg_lens.append(np.hypot(pts[j][0] - pts[i][0], pts[j][1] - pts[i][1]))
    total_len = sum(seg_lens)
    if total_len < 1e-9:
        return np.array([pts[0]] * n)

    cum = [0.0]
    for sl in seg_lens[:-1]:  # exclude closing segment
        cum.append(cum[-1] + sl)

    result = []
    for k in range(n):
        t = k / n * total_len
        # Find segment index
        idx = 0
        for i in range(len(cum)):
            if cum[i] <= t:
                idx = i
        seg_t = (t - cum[idx]) / seg_lens[idx] if seg_lens[idx] > 0 else 0
        j = (idx + 1) % len(pts)
        x = pts[idx][0] + seg_t * (pts[j][0] - pts[idx][0])
        y = pts[idx][1] + seg_t * (pts[j][1] - pts[idx][1])
        result.append([x, y])
    return np.array(result, dtype=np.float32)


def _compare_front_back_contours(steps):
    """Compare front and back PCB outlines — they must be identical (same physical board).

    Since back is just the front PCB flipped, we merge both into a refined contour.
    Returns a dict with: ok, message, mismatch_area_pct, merged_outline_mm (optional).
    """
    outlines = {}
    for s in steps:
        if s.get("calibration_success") and s.get("transparent_pcb_outline_mm"):
            outlines[s["side"]] = s["transparent_pcb_outline_mm"]
    if len(outlines) < 2:
        return {"ok": False, "message": "前后两面标定数据不完整，无法对比轮廓", "mismatch_area_pct": 0}

    front_ol = outlines["front"]
    back_ol = outlines["back"]
    if not isinstance(front_ol, list) or len(front_ol) == 0:
        return {"ok": False, "message": "正面轮廓数据为空", "mismatch_area_pct": 0}
    if not isinstance(back_ol, list) or len(back_ol) == 0:
        return {"ok": False, "message": "背面轮廓数据为空", "mismatch_area_pct": 0}

    # Extract per-side PPM calibration for inter-camera scale normalization.
    # Different cameras can yield slightly different px/mm values (e.g.
    # front=47.6, back=48.4 → 1.6% systematic error).  Passing both PPM
    # values lets the merge function unify the coordinate systems.
    front_ppm = 0.0
    back_ppm = 0.0
    for s in steps:
        if s.get("side") == "front":
            front_ppm = s.get("pixels_per_mm", 0.0)
        elif s.get("side") == "back":
            back_ppm = s.get("pixels_per_mm", 0.0)

    AREA_THRESHOLD_PCT = 10.0  # 10% tolerance

    # ── Merge front + back into refined consensus outline ──
    w_mm = steps[0].get("frame_w_mm", 40.0) if steps else 40.0
    h_mm = steps[0].get("frame_h_mm", 25.0) if steps else 25.0

    front_result = {
        "outline": front_ol,
        "pixels_per_mm": front_ppm,
        "width_mm": w_mm,
        "height_mm": h_mm,
    }
    back_result = {
        "outline": back_ol,
        "pixels_per_mm": back_ppm,
        "width_mm": w_mm,
        "height_mm": h_mm,
    }
    cross_result = PCBRecognitionPipeline.cross_validate_front_back(
        front_result, back_result
    )
    merged_outline = cross_result["outline"]
    f_area = cross_result["front_area_mm2"]
    b_area = cross_result["back_area_mm2"]
    merged_area = cross_result["consensus_area_mm2"]

    if f_area <= 0 or b_area <= 0:
        return {"ok": False, "message": "无法计算PCB面积", "mismatch_area_pct": 0}

    # ── Consensus-based area validation ──
    # All three areas (f_area, b_area, merged_area) come from the same
    # mask-drawing pipeline at the same SCALE.  This eliminates the bias
    # introduced by comparing raw contour areas against mask-derived
    # consensus areas.
    consensus_dev_front = (abs(f_area - merged_area) / merged_area * 100.0
                           if merged_area > 0 else 999)
    consensus_dev_back = (abs(b_area - merged_area) / merged_area * 100.0
                          if merged_area > 0 else 999)
    consensus_dev = max(consensus_dev_front, consensus_dev_back)

    logger.info(
        "Front/back merge: front=%dpts(mask=%.0fmm2) back=%dpts(mask=%.0fmm2) "
        "-> merged=%dpts(%.0fmm2), consensus_dev=%.1f%% (f=%.1f%% b=%.1f%%)",
        len(front_ol), f_area, len(back_ol), b_area,
        len(merged_outline), merged_area,
        consensus_dev, consensus_dev_front, consensus_dev_back)

    result = {
        "mismatch_area_pct": round(consensus_dev, 1),
        "merged_outline_mm": merged_outline,
        "merged_area_mm2": round(merged_area, 1),
    }

    if consensus_dev > AREA_THRESHOLD_PCT:
        result["ok"] = False
        result["message"] = (f"正反面轮廓与共识偏差 {consensus_dev:.1f}%，"
                             "已合并生成参考轮廓")
    else:
        result["ok"] = True
        result["message"] = (f"正反面轮廓与共识匹配通过（偏差 {consensus_dev:.1f}%），"
                             "已合并为精确轮廓")

    return result


@app.post("/api/simulate")
def simulate(
    frame_w_mm: float = Form(40.0),
    frame_h_mm: float = Form(25.0),
):
    """Auto-test: run black frame detection + calibration on input/front.jpg and input/back.jpg.

    Called by the frontend on page load to validate the photos without requiring
    the user to upload them manually.
    """
    input_dir = ROOT / "input"
    steps = []
    total = 0

    for side, fname in [("front", "front.jpg"), ("back", "back.jpg")]:
        total += 1
        img_path = input_dir / fname
        if not img_path.exists():
            steps.append({
                "side": side, "frame_detected": False, "aspect_ok": False,
                "calibration_success": False,
                "calibration_error_msg": f"图片文件不存在: {img_path}",
                "file": fname,
            })
            continue

        img_buf = img_path.read_bytes()

        try:
            target_aspect = frame_w_mm / frame_h_mm if frame_h_mm > 0 else None
            frame_result = _detect_black_frame(img_buf, target_aspect)
            frame_detected = frame_result.get("found", False)
            detected_aspect = frame_result.get("aspect_ratio", 0)
            detected_w_px = frame_result.get("avg_width_px", 0)
            detected_h_px = frame_result.get("avg_height_px", 0)

            cal_success = False
            cal_data = {}
            orientation_hint = None
            cal_w_mm = frame_w_mm
            cal_h_mm = frame_h_mm
            aspect_error_pct = 0.0

            if frame_detected and detected_aspect > 0:
                # Use exact user-provided frame dimensions (from web UI).
                # The black frame has a known physical size — do NOT auto-adapt.
                cal_w_mm = frame_w_mm
                cal_h_mm = frame_h_mm
                expected_aspect = cal_w_mm / cal_h_mm if cal_h_mm > 0 else 0
                if expected_aspect > 0:
                    aspect_error_pct = abs(detected_aspect - expected_aspect) / expected_aspect * 100
                else:
                    aspect_error_pct = 0
                aspect_ok = aspect_error_pct <= 25  # allow up to 25% for perspective skew

                try:
                    cal_data = _calibrate_black_frame(img_buf, cal_w_mm, cal_h_mm)
                    cal_success = True
                except Exception as e:
                    cal_data = {"calibration_error_msg": str(e)}

            elif frame_detected:
                # Frame found but aspect_ratio is 0 (should not happen)
                aspect_ok = False
                aspect_error_pct = 100.0
                expected_aspect = frame_h_mm / frame_w_mm
            else:
                aspect_ok = False
                aspect_error_pct = 100.0
                expected_aspect = frame_h_mm / frame_w_mm

            step = {
                "side": side,
                "frame_detected": frame_detected,
                "aspect_ok": aspect_ok,
                "calibration_success": cal_success,
                "detected_aspect_ratio": round(detected_aspect, 4),
                "expected_aspect_ratio": round(expected_aspect, 4),
                "aspect_error_pct": round(aspect_error_pct, 1),
                "orientation_hint": orientation_hint,
                "file": fname,
                "image": fname,
                "frame_w_mm": cal_w_mm,
                "frame_h_mm": cal_h_mm,
                "detected_w_px": round(detected_w_px, 1),
                "detected_h_px": round(detected_h_px, 1),
            }
            if cal_success:
                step.update({
                    "calibration_id": cal_data["calibration_id"],
                    "pixels_per_mm": cal_data["pixels_per_mm"],
                    "rectified_w_mm": cal_w_mm,
                    "rectified_h_mm": cal_h_mm,
                    "confidence": 0.95,
                    "rectified_png_base64": cal_data["rectified_png_base64"],
                    "transparent_pcb_base64": cal_data.get("transparent_pcb_b64", ""),
                    "transparent_pcb_outline_mm": cal_data.get("transparent_pcb_outline_mm", []),
                })
            else:
                step["calibration_error_msg"] = cal_data.get("calibration_error_msg", "标定失败")

            steps.append(step)
        except Exception as e:
            steps.append({
                "side": side, "frame_detected": False, "aspect_ok": False,
                "calibration_success": False,
                "calibration_error_msg": str(e),
                "orientation_hint": None, "file": fname, "image": fname,
                "frame_w_mm": frame_w_mm,
                "frame_h_mm": frame_h_mm,
                "detected_w_px": 0,
                "detected_h_px": 0,
            })

    all_ok = all(s.get("calibration_success", False) for s in steps)
    # ── Front/back contour comparison ──
    contour_match = _compare_front_back_contours(steps)
    # Use adjusted frame dimensions from the first successful step
    adj_w = steps[0].get("frame_w_mm", frame_w_mm) if steps else frame_w_mm
    adj_h = steps[0].get("frame_h_mm", frame_h_mm) if steps else frame_h_mm
    return {
        "success": all_ok,
        "total_images": total,
        "frame_w_mm": adj_w,
        "frame_h_mm": adj_h,
        "steps": steps,
        "contour_match": contour_match,
    }


@app.post("/api/vision/preview-black-frame")
def preview_black_frame(
    file: UploadFile = File(...),
    frame_w_mm: float = Form(40.0),
    frame_h_mm: float = Form(25.0),
):
    """Detect the black frame in an uploaded photo and return an annotated preview."""
    img_buf = file.file.read()
    target_aspect = frame_w_mm / frame_h_mm if frame_h_mm > 0 else None
    result = _detect_black_frame(img_buf, target_aspect)
    return result


@app.post("/api/vision/calibrate-black-frame")
def calibrate_black_frame_endpoint(
    file: UploadFile = File(...),
    frame_w_mm: float = Form(40.0),
    frame_h_mm: float = Form(25.0),
):
    """Detect black frame, compute perspective rectification, save calibration.

    Returns calibration_id, rectified PNG, pixel-per-mm scale, etc.
    """
    img_buf = file.file.read()
    result = _calibrate_black_frame(img_buf, frame_w_mm, frame_h_mm)
    return result


@app.get("/api/mos/list")
def list_mos():
    """Return available MOSFET MPNs for the IC catalog dropdown."""
    from .mos import list_available_mpns
    return [{"model": mpn} for mpn in list_available_mpns()]


# ── Optimization Dashboard ──────────────────────────────────────────────
@app.get("/optimization")
async def optimization_dashboard():
    """Serve the self-optimization debug dashboard."""
    return FileResponse(STATIC_ROOT / "optimization_dashboard.html")


@app.get("/api/optimization/summary")
async def optimization_summary():
    """Return optimization history summary for the dashboard."""
    opt_dir = DATA_ROOT / "optimization"
    viz_dir = DATA_ROOT / "visualization"

    rounds: list[dict] = []
    round_pattern = re.compile(r"report_round_(\d+)_(front|back)\.json")

    round_files: dict[int, dict[str, Path]] = {}
    if opt_dir.exists():
        for f in sorted(opt_dir.iterdir()):
            m = round_pattern.match(f.name)
            if m:
                rnd = int(m.group(1))
                side = m.group(2)
                round_files.setdefault(rnd, {})[side] = f

    for rnd in sorted(round_files.keys()):
        files = round_files[rnd]
        if "front" not in files and "back" not in files:
            continue

        front_data = None
        back_data = None
        try:
            if "front" in files:
                front_data = json.loads(files["front"].read_text(encoding="utf-8"))
            if "back" in files:
                back_data = json.loads(files["back"].read_text(encoding="utf-8"))
        except Exception:
            continue

        report = front_data or back_data or {}
        if isinstance(report, dict) and "composite_score" in report:
            comp = report["composite_score"]
            total = round(comp.get("total", 0), 1) if isinstance(comp, dict) else 0
        else:
            comp = {}
            total = 0

        front_score = 0
        back_score = 0
        if front_data and isinstance(front_data, dict):
            fc = front_data.get("composite_score", {})
            front_score = round(fc.get("total", 0), 1) if isinstance(fc, dict) else 0
        if back_data and isinstance(back_data, dict):
            bc = back_data.get("composite_score", {})
            back_score = round(bc.get("total", 0), 1) if isinstance(bc, dict) else 0

        iou_mean = 0
        for side_data in [front_data, back_data]:
            if side_data and isinstance(side_data, dict):
                iou = side_data.get("iou", {})
                if isinstance(iou, dict):
                    iou_mean = max(iou_mean, iou.get("mean", 0))

        matches_quality = {"excellent": 0, "good": 0, "fair": 0, "poor": 0}
        for side_data in [front_data, back_data]:
            if side_data and isinstance(side_data, dict):
                iou = side_data.get("iou", {})
                if isinstance(iou, dict) and "values" in iou:
                    for v in iou["values"]:
                        if v >= 0.8:
                            matches_quality["excellent"] += 1
                        elif v >= 0.6:
                            matches_quality["good"] += 1
                        elif v >= 0.3:
                            matches_quality["fair"] += 1
                        else:
                            matches_quality["poor"] += 1

        diagnosis_parts = []
        for label, side_data in [("Front", front_data), ("Back", back_data)]:
            if side_data and isinstance(side_data, dict):
                cd = side_data.get("center_offset", {})
                if isinstance(cd, dict):
                    mean_off = cd.get("mean_px", 0)
                    if mean_off > 5:
                        diagnosis_parts.append(f"{label} offset:{mean_off:.1f}px")
                rec = side_data.get("recall", {})
                if isinstance(rec, dict) and rec.get("value", 1) < 0.8:
                    diagnosis_parts.append(f"{label} recall:{rec['value']:.1%}")

        round_entry = {
            "round": rnd,
            "front_score": front_score,
            "back_score": back_score,
            "composite_score": total,
            "iou_mean": round(iou_mean, 3),
            "diagnosis": "; ".join(diagnosis_parts) if diagnosis_parts else "OK",
            "matches_quality": matches_quality,
        }

        params_file = opt_dir / f"params_round_{rnd:03d}.json"
        if params_file.exists():
            try:
                round_entry["params"] = json.loads(params_file.read_text(encoding="utf-8"))
            except Exception:
                round_entry["params"] = {}

        viz_files = []
        if viz_dir.exists():
            for side in ["front", "back"]:
                viz_file = viz_dir / f"round_{rnd:03d}_{side}.jpg"
                if viz_file.exists():
                    viz_files.append(f"data/visualization/round_{rnd:03d}_{side}.jpg")
        round_entry["viz_images"] = viz_files

        rounds.append(round_entry)

    best_score = max((r["composite_score"] for r in rounds), default=0)

    best_params = {}
    best_params_file = opt_dir / "best_params.json"
    if best_params_file.exists():
        try:
            best_params = json.loads(best_params_file.read_text(encoding="utf-8"))
        except Exception:
            pass

    return {
        "rounds": rounds,
        "best_score": best_score,
        "best_params": best_params,
        "total_rounds": len(rounds),
    }



def main() -> None:
    import uvicorn

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8000)
    args, _ = parser.parse_known_args()
    uvicorn.run("battery_designer.app:app", host="127.0.0.1", port=args.port, reload=False)


if __name__ == "__main__":
    main()
