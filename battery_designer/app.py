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
from .vision import extract_pcb as _extract_pcb, detect_holes as _detect_holes
from .vision import _make_transparent, refine_outline_geometry
from .vision import detect_black_frame as _detect_black_frame, calibrate_black_frame as _calibrate_black_frame


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

    for cand in result.get("candidates", []):
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
        # Padding: 30% of pad size + 5px margin (tight search area)
        pad_px = int(max(pad_w, pad_h) * 0.30) + 5

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
        hsv = cv2.cvtColor(roi[:, :, :3], cv2.COLOR_BGR2HSV)
        metallic = cv2.inRange(hsv, np.array([0, 0, 40]), np.array([180, 100, 255]))
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
            continue

        # Find the largest contour (should be the metallic pad)
        best = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(best)
        poly_area = pad_w * pad_h
        # Safety: contour must be 25%–400% of VLM polygon area
        if area < poly_area * 0.25 or area > poly_area * 4.0:
            logger.info("CV refine: %s contour area %.0f outside range [%.0f, %.0f], skip",
                        cand.get('label','?'), area, poly_area*0.25, poly_area*4.0)
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

        # Safety: max shift = 35% of pad diagonal,
        # capped at 2% of image max dimension (scales with PCB size)
        img_max_dim = max(h_img, w_img)
        max_shift = min((pad_w ** 2 + pad_h ** 2) ** 0.5 * 0.35,
                        img_max_dim * 0.02)
        logger.info("CV refine: %s VLM=(%.1f,%.1f) CV=(%.1f,%.1f) shift=%.1fpx max=%.1fpx",
                    cand.get('label','?'), cx_vlm, cy_vlm, cx_px, cy_px, shift_px, max_shift)
        if shift_px > max_shift:
            continue

        # Apply shift (position only, preserve VLM shape)
        dx_mm = round(dx / pixels_per_mm, 3)
        dy_mm = round(dy / pixels_per_mm, 3)
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

    return result


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
        poly = region.get("polygon", [])
        center = region.get("center", {})
        if len(poly) < 3 or not center:
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
        #    Alignment axis: use PCB center as symmetry anchor (pads are
        #    symmetric about board center by PCB layout convention).
        #    For 3+ pads: use median (self-consistent, more robust).
        if n == 2:
            if axis == "v" and pcb_w_mm > 0:
                med_align = pcb_w_mm / 2  # PCB center X
            elif axis == "h" and pcb_h_mm > 0:
                med_align = pcb_h_mm / 2  # PCB center Y
            elif axis == "v":
                med_align = statistics.median([p["cx"] for p in grp])
            else:
                med_align = statistics.median([p["cy"] for p in grp])
        else:
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
            med_gap = statistics.median(gaps)
            # If all gaps within 25% of median → enforce uniform spacing
            if med_gap > 0 and all(abs(g - med_gap) / med_gap < 0.25 for g in gaps):
                start = coords[0]
                target_positions = {}
                for idx_p, p in enumerate(ordered):
                    target_positions[id(p)] = start + idx_p * med_gap

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
def detect_terminals(calibration_id: str = Form(...), side: str = Form(...)):
    """Detect terminal candidates using VLM identification on cropped PCB.

    For transparent PCB: crop to PCB bounding box so the board fills the entire
    image. This eliminates the Y-offset problem caused by PCB floating in white
    space. VLM coordinates are then offset back to full-frame mm.
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
                    # Upscale 2x for better VLM recognition
                    upscaled = cv2.resize(composited, None, fx=2.0, fy=2.0,
                                          interpolation=cv2.INTER_CUBIC)
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

    # ── Group alignment correction (PCB layout symmetry) ──
    try:
        result = _align_pad_groups(result, pixels_per_mm,
                                     crop_w_mm, crop_h_mm)
    except Exception:
        logger.warning("Pad group alignment failed", exc_info=True)

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
        if dist > 15.0:  # too far, not a valid match
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
            consensus_outline, _, _, _ = _merge_front_back_outlines(
                result["outline"], other_outline, merge_w, merge_h,
                front_ppm=ppm, back_ppm=other_ppm)
            result["outline"] = consensus_outline
            consensus_msg = f" (正反面对照纠正, {len(consensus_outline)}顶点)"

            # ── Rebuild display mask & transparent PNG from consensus outline ──
            try:
                nparr2 = np.frombuffer(png, np.uint8)
                img2 = cv2.imdecode(nparr2, cv2.IMREAD_COLOR)
                h2, w2 = img2.shape[:2]
                if len(consensus_outline) >= 3:
                    outline_px = np.array([
                        [round(p["x_mm"] / w_mm * w2),
                         round(p["y_mm"] / h_mm * h2)]
                        for p in consensus_outline
                    ], dtype=np.int32)
                    new_mask = np.zeros((h2, w2), dtype=np.uint8)
                    cv2.fillPoly(new_mask, [outline_px], 255)
                    # Punch grooves back through (consensus merge simplifies the
                    # polygon and loses groove detail → white background).
                    if orig_mask is not None and orig_mask.shape == new_mask.shape:
                        new_mask = cv2.bitwise_and(new_mask, orig_mask)
                    result["pcb_mask_b64"] = base64.b64encode(
                        cv2.imencode(".png", new_mask)[1]).decode("ascii")
                    result["transparent_pcb_b64"] = base64.b64encode(
                        _make_transparent(img2, new_mask)).decode("ascii")
                    logger.info("extract-pcb consensus: rebuilt display from %d-vertex outline",
                                len(consensus_outline))
            except Exception:
                logger.warning("extract-pcb consensus: image rebuild failed",
                               exc_info=True)

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

    w_mm = max(front_meta.get("frame_w_mm", 60.0), back_meta.get("frame_w_mm", 60.0))
    h_mm = max(front_meta.get("frame_h_mm", 40.0), back_meta.get("frame_h_mm", 40.0))

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


def _remove_collinear_pts(pts, angle_tol_deg=3.5):
    """Remove vertices where angle ≈ 180° (collinear)."""
    if len(pts) < 4:
        return pts
    n = len(pts)
    keep = [True] * n
    for i in range(n):
        a, b, c = pts[(i - 1) % n], pts[i], pts[(i + 1) % n]
        v1, v2 = a - b, c - b
        l1, l2 = np.linalg.norm(v1), np.linalg.norm(v2)
        if l1 < 1e-6 or l2 < 1e-6:
            keep[i] = False
            continue
        cos_a = np.clip(np.dot(v1, v2) / (l1 * l2), -1.0, 1.0)
        if np.degrees(np.arccos(cos_a)) >= (180.0 - angle_tol_deg):
            keep[i] = False
    result = pts[np.array(keep)]
    return result if len(result) >= 3 else pts


def _contour_to_points(ol_mm):
    """Convert outline dict list to flat (x,y) numpy array."""
    pts = []
    for pt in ol_mm:
        if isinstance(pt, dict):
            pts.append([pt.get("x_mm", 0), pt.get("y_mm", 0)])
        elif hasattr(pt, '__iter__') and len(pt) >= 2:
            pts.append([float(pt[0]), float(pt[1])])
    return np.array(pts, dtype=np.float32) if pts else None


def _simplify_polygon_dp(pts_mm, epsilon_mm=2.0):
    """Douglas-Peucker polygon simplification — keep only main shape vertices.

    The raw HSV-mask contour has dozens/hundreds of vertices, many of which are
    shadow-noise artifacts.  A real PCB has a clean geometric outline with ~6–20
    significant corners.  This function strips the noise before Chamfer matching.

    epsilon_mm:  minimum allowed deviation in mm.  Default 2.0 mm ensures even
                 moderate shadow burrs (2-5mm deviation) are merged into the
                 main polygon edge, not preserved as separate vertices.
                 Adaptive floor = max(2.0, 1.2% of perimeter) scales for any size.
    """
    if len(pts_mm) < 4:
        return pts_mm
    peri_mm = cv2.arcLength(pts_mm.reshape(-1, 1, 2), True)
    # Adaptive epsilon: max(2.0mm floor, 1.2% of perimeter).
    # e.g. 200mm PCB → 2.4mm eps (aggressive against shadow burrs)
    # e.g.  80mm PCB → 2.0mm eps (still removes typical 2-5mm noise)
    eps = max(epsilon_mm, peri_mm * 0.012)
    simplified = cv2.approxPolyDP(
        pts_mm.reshape(-1, 1, 2), eps, True).reshape(-1, 2)
    # Remove collinear leftovers after DP (tight 3° tolerance)
    simplified = _remove_collinear_pts(simplified, angle_tol_deg=3.0)
    logger.info("DP simplify: %d → %d vertices (eps=%.2f mm, peri=%.0f mm)",
                len(pts_mm), len(simplified), eps, peri_mm)
    return simplified if len(simplified) >= 3 else pts_mm


def _prune_hull_burrs(pts_mm, min_depth_mm=1.5, max_width_mm=8.0):
    """Remove narrow deep dents (shadow burrs) from polygon vertices.

    After DP simplification, small-scale shadow burrs may still survive as
    a single deep-concave vertex on an otherwise straight polygon edge.
    These are characterised by: (1) vertex angle near 180° (shallow V),
    (2) large point-to-chord distance (deep dent), (3) narrow chord (small mouth).

    Strategy: for each vertex, compute the distance to the chord connecting
    its two neighbours.  If the chord is narrow and the vertex is far from it,
    the vertex is a burr and is removed.  Legitimate corners have either
    large chord widths or sharp angles that survive the filter.
    """
    if len(pts_mm) < 5:
        return pts_mm
    n = len(pts_mm)
    keep = [True] * n
    removed = 0

    for i in range(n):
        p_prev = pts_mm[(i - 1) % n]
        p_i = pts_mm[i]
        p_next = pts_mm[(i + 1) % n]

        # Chord vector and length
        chord = p_next - p_prev
        chord_len = float(np.linalg.norm(chord))
        if chord_len < 0.1:
            continue

        # Distance from vertex to chord line (point-line distance)
        # 2D cross product: |chord x (p_i - p_prev)| = |cx*dy - cy*dx|
        d = p_i - p_prev
        cross_abs = abs(float(chord[0] * d[1] - chord[1] * d[0]))
        dist = cross_abs / chord_len

        if dist < min_depth_mm or chord_len > max_width_mm:
            continue

        # Check angle at vertex: near-180° = shallow dent (burr-like)
        v1 = p_prev - p_i
        v2 = p_next - p_i
        len1 = float(np.linalg.norm(v1))
        len2 = float(np.linalg.norm(v2))
        if len1 < 1e-6 or len2 < 1e-6:
            keep[i] = False
            removed += 1
            continue
        cos_angle = max(-1.0, min(1.0, float(np.dot(v1, v2)) / (len1 * len2)))
        angle_deg = float(np.degrees(np.arccos(cos_angle)))

        # Vertex with angle > 155°: almost-straight edge with a dent → burr
        if angle_deg >= 155.0:
            keep[i] = False
            removed += 1

    if removed > 0:
        result = pts_mm[np.array(keep)]
        logger.info("Hull burr prune: removed %d burr vertices, %d → %d",
                    removed, n, len(result))
        return result if len(result) >= 3 else pts_mm
    return pts_mm


def _extract_straight_edges(pts_mm, min_length_mm=5.0):
    """Extract long straight edge segments from a DP-simplified polygon.

    Each segment is a pair of consecutive vertices. Segments shorter than
    min_length_mm are filtered out — short edges are typically shadow burrs,
    not real PCB physical edges. Returns metadata needed for alignment scoring:
    direction, normal, midpoint, and projection range.

    Returns:
        list of dict: [{p1, p2, length, direction, normal, midpoint}, ...]
    """
    n = len(pts_mm)
    if n < 3:
        return []
    edges = []
    for i in range(n):
        p1 = pts_mm[i]
        p2 = pts_mm[(i + 1) % n]
        vec = p2 - p1
        length = float(np.linalg.norm(vec))
        if length < min_length_mm:
            continue
        direction = vec / length
        # Perpendicular normal (CCW rotation of direction)
        normal = np.array([-direction[1], direction[0]], dtype=np.float64)
        midpoint = (p1 + p2) / 2.0
        edges.append({
            'p1': p1.copy(),
            'p2': p2.copy(),
            'length': length,
            'direction': direction,
            'normal': normal,
            'midpoint': midpoint,
        })
    return edges


def _score_straight_edge_align(front_edges, back_pts, dx_mm, dy_mm,
                                angle_cos_th=0.966, max_dist_mm=2.0,
                                min_overlap=0.3, min_length_mm=5.0):
    """Score a candidate (dx, dy) translation by straight-edge alignment quality.

    PCB physical edges are straight — unlike shadow-generated jagged edges.
    This function only trusts the long straight segments from DP simplification
    as ground truth for alignment.

    For each front straight edge, it looks for a parallel, overlapping, and
    nearby counterpart in the shifted back polygon.  The total score is the sum
    of (edge_length × overlap_ratio × distance_decay) over all matched edges.

    Returns:
        float: alignment score (higher = better).  0.0 if no edges match.
    """
    if not front_edges:
        return 0.0

    # Shift back and extract its straight edges
    shifted_back = back_pts.copy()
    shifted_back[:, 0] += dx_mm
    shifted_back[:, 1] += dy_mm
    back_edges = _extract_straight_edges(shifted_back, min_length_mm=min_length_mm)

    if not back_edges:
        return 0.0

    score = 0.0
    for fe in front_edges:
        best_quality = 0.0
        for be in back_edges:
            # 1. Direction similarity: cos_sim > angle_cos_th (±15°)
            cos_sim = abs(float(np.dot(fe['direction'], be['direction'])))
            if cos_sim < angle_cos_th:
                continue

            # 2. Perpendicular distance between parallel lines
            dist = abs(float(np.dot(fe['normal'], be['midpoint'] - fe['midpoint'])))
            if dist > max_dist_mm:
                continue

            # 3. Projection overlap along front edge direction
            # Project back endpoints onto front edge's infinite line
            proj_a1 = 0.0
            proj_a2 = fe['length']
            proj_b1 = float(np.dot(fe['direction'], be['p1'] - fe['p1']))
            proj_b2 = float(np.dot(fe['direction'], be['p2'] - fe['p1']))

            b_min, b_max = min(proj_b1, proj_b2), max(proj_b1, proj_b2)
            overlap_start = max(proj_a1, b_min)
            overlap_end = min(proj_a2, b_max)
            overlap_len = max(0.0, overlap_end - overlap_start)

            if overlap_len <= 0:
                continue

            overlap_ratio = overlap_len / fe['length']
            if overlap_ratio < min_overlap:
                continue

            # Quality: overlap × linear distance decay
            dist_decay = max(0.0, 1.0 - dist / max_dist_mm)
            quality = overlap_ratio * dist_decay

            if quality > best_quality:
                best_quality = quality

        if best_quality > 0:
            score += fe['length'] * best_quality

    return score


def _merge_front_back_outlines(front_ol, back_ol, w_mm, h_mm,
                              front_ppm=0.0, back_ppm=0.0):
    """Merge front + back PCB outlines via full 2D mask comparison.

    Key constraint:  back is the front PCB **left-right mirrored**
    (same board, flipped over — left⇄right, top=top, bottom=bottom).

    Strategy (full 2D consensus):
      1. Mirror back around its bbox x-center (correct physical flip).
      2. Align by STRAIGHT-EDGE MATCHING (translation only):
         PCB physical edges are straight — shadow/jagged edges are NOT trusted.
         Extract long straight segments (≥5mm) from DP-simplified polygons,
         then search (dx, dy) that maximises the total length-weighted quality
         of parallel + overlapping + nearby front⇆back edge pairs.
         Two-pass coarse→fine grid search for optimal translation.
         NOTE: consensus masks are drawn from the RAW (un-simplified) points so
         shallow notch/groove features survive; the 2mm DP simplification is used
         ONLY to obtain clean edges for alignment.  No PPM/scale rescaling is
         applied — both outlines are already in mm, and an IoU-based scale search
         would merely shrink the good outline to match the smaller one.
      3. INTERSECTION mask → removes one-sided PROTRUSION burrs.
      4. diff = fm XOR bm → the full 2D difference between the two masks.
      5. Hull = convex hull of intersection.
      6. diff ∩ hull → indentation-style burrs (one side's shadow artifact
         creating a "bite" that the other side fills).
      7. Connected-component analysis on diff in hull:
         - Small isolated components → shadow burrs → FILL them.
         - Large components → serious misalignment → log warning, keep.
      8. Extract clean polygon from filled result.
    """
    # RAW points are kept for consensus-mask drawing so that shallow notch /
    # groove features (0.3-0.5mm) are preserved.  The 2mm DP simplification
    # below would erase them, so it is applied ONLY to copies used to obtain
    # clean straight edges for alignment.
    front_raw = _contour_to_points(front_ol)
    back_raw = _contour_to_points(back_ol)

    # ---- simplify copies for robust straight-edge alignment ----
    # Raw HSV-mask contours have 50-200+ noisy vertices from shadow edges.
    # DP simplification keeps the ~6-20 geometric corners for edge matching,
    # but is NEVER used for the consensus mask (would destroy notches).
    front_pts = _simplify_polygon_dp(front_raw.copy()) if front_raw is not None else None
    back_pts = _simplify_polygon_dp(back_raw.copy()) if back_raw is not None else None

    if front_pts is None or back_pts is None:
        # One outline missing — return the available one with dummy areas
        ol = front_ol if front_pts is not None else (back_ol or [])
        pts = front_pts if front_pts is not None else back_pts
        if pts is not None and len(pts) >= 3:
            a = float(abs(cv2.contourArea(pts.reshape(-1, 1, 2).astype(np.float32))))
        else:
            a = 0.0
        return ol, a, a, a

    # ── High-resolution canvas (0.05mm/pixel) ──
    SCALE = 20.0  # px/mm → 0.05mm precision
    cw = int(w_mm * SCALE) + 1
    ch = int(h_mm * SCALE) + 1
    if cw < 10 or ch < 10:
        # Canvas too small — return front outline with contour-based area
        if len(front_pts) >= 3:
            a_f = float(abs(cv2.contourArea(front_pts.reshape(-1, 1, 2).astype(np.float32))))
        else:
            a_f = 0.0
        if len(back_pts) >= 3:
            a_b = float(abs(cv2.contourArea(back_pts.reshape(-1, 1, 2).astype(np.float32))))
        else:
            a_b = 0.0
        return front_ol, a_f, a_b, a_f

    # ── 1. Mirror back around its bbox x-center ──
    # Use the RAW bbox (simplification can shift extreme points slightly).
    # The same mirror is applied to BOTH the raw points (consensus mask) and
    # the simplified points (alignment).
    b_min_x, b_max_x = np.min(back_raw[:, 0]), np.max(back_raw[:, 0])
    back_cx = (b_min_x + b_max_x) / 2.0
    back_mirrored = back_pts.copy()            # simplified → alignment
    back_mirrored[:, 0] = 2.0 * back_cx - back_mirrored[:, 0]
    back_raw_mir = back_raw.copy()             # raw → consensus mask
    back_raw_mir[:, 0] = 2.0 * back_cx - back_raw_mir[:, 0]

    # ── 2. Align: centroid initial guess → coarse+fine grid search for best straight-edge match ──
    f_centroid = np.mean(front_pts, axis=0)
    b_centroid = np.mean(back_mirrored, axis=0)
    init_tx = f_centroid - b_centroid
    back_mirrored += init_tx
    back_raw_mir += init_tx

    # ── 2a. Extent matching: compensate systematic scale mismatch ──
    # The two photos may be at slightly different zoom, so the other side can
    # be uniformly larger/smaller than the current side.  Left uncorrected the
    # intersection would (a) shrink the result to the smaller side, and (b)
    # "cut" the corners (offset arcs intersect into a spurious large corner
    # radius that the downstream geometry refinement then over-rounds,
    # shrinking the board).  Scale the OTHER side about its centroid to match
    # the CURRENT side's bbox extent BEFORE the straight-edge search, so the
    # subsequent alignment re-optimises the translation on the scaled outline.
    # This preserves the current side's own dimensions (its calibration scale)
    # while still using the other side to recover notches and strip burrs.
    f_min = front_raw.min(axis=0); f_max = front_raw.max(axis=0)
    b_min = back_raw_mir.min(axis=0); b_max = back_raw_mir.max(axis=0)
    f_ext = f_max - f_min; b_ext = b_max - b_min
    if float(np.min(b_ext)) > 1e-6:
        sx = max(0.95, min(1.05, float(f_ext[0] / b_ext[0])))
        sy = max(0.95, min(1.05, float(f_ext[1] / b_ext[1])))
        if abs(sx - 1.0) > 0.001 or abs(sy - 1.0) > 0.001:
            bc = (b_min + b_max) / 2.0
            scale_vec = np.array([sx, sy])
            back_raw_mir = (back_raw_mir - bc) * scale_vec + bc
            back_mirrored = (back_mirrored - bc) * scale_vec + bc
            logger.info("Extent match: other side scaled by (%.4f, %.4f) "
                        "to match current extent", sx, sy)

    def _draw_mask(pts, dx_px=0, dy_px=0):
        mask = np.zeros((ch, cw), dtype=np.uint8)
        scaled = pts * SCALE
        pix = np.round(scaled).astype(np.int32)
        pix[:, 0] += dx_px
        pix[:, 1] += dy_px
        pix[:, 0] = np.clip(pix[:, 0], 0, cw - 1)
        pix[:, 1] = np.clip(pix[:, 1], 0, ch - 1)
        cv2.fillPoly(mask, [pix], 255)
        return mask

    # Draw front mask once (fixed reference) — from RAW points to keep notches.
    fm = _draw_mask(front_raw)

    # ── Straight-Edge Alignment ──
    # PCB physical edges are straight — shadow/jagged edges are NOT trusted.
    # We rely on DP-simplified long straight edges (≥5mm) as ground truth.
    # Score each (dx, dy) by total length of front straight edges that have
    # a parallel, overlapping, and nearby counterpart in the shifted back polygon.
    # Two-pass coarse→fine: (±5mm @ 0.5mm step) → (±1mm @ 0.1mm step)

    # Pre-extract front straight edges (fixed reference, computed once)
    front_edges = _extract_straight_edges(front_pts, min_length_mm=5.0)
    logger.info("Front straight edges: %d segments", len(front_edges))

    best_dx_px, best_dy_px = 0, 0
    best_score = -1.0
    centroid_score = -1.0

    search_passes = [(5.0, 0.5), (1.0, 0.1)]  # (range_mm, step_mm)

    for pass_idx, (sr_mm, ss_mm) in enumerate(search_passes):
        sr_px = int(sr_mm * SCALE)            # search radius in pixels
        ss_px = max(1, int(ss_mm * SCALE))     # step in pixels
        pass_best_score = -1.0

        for dy_px in range(best_dy_px - sr_px, best_dy_px + sr_px + 1, ss_px):
            for dx_px in range(best_dx_px - sr_px, best_dx_px + sr_px + 1, ss_px):
                dx_mm = dx_px / SCALE
                dy_mm = dy_px / SCALE
                score = _score_straight_edge_align(front_edges, back_mirrored,
                                                    dx_mm, dy_mm)
                if score > pass_best_score:
                    pass_best_score = score
                    best_dx_px = dx_px
                    best_dy_px = dy_px

        if pass_idx == 0:
            centroid_score = pass_best_score
        logger.info("Align pass %d (range=±%.1fmm step=%.1fmm): "
                    "best_edge_score=%.1f at (%.2f, %.2f) mm",
                    pass_idx + 1, sr_mm, ss_mm, pass_best_score,
                    best_dx_px / SCALE, best_dy_px / SCALE)

    # Apply optimal translation to BOTH simplified (alignment) and raw (mask)
    best_dx_mm = best_dx_px / SCALE
    best_dy_mm = best_dy_px / SCALE
    back_mirrored[:, 0] += best_dx_mm
    back_mirrored[:, 1] += best_dy_mm
    back_raw_mir[:, 0] += best_dx_mm
    back_raw_mir[:, 1] += best_dy_mm

    total_tx_mm = float(init_tx[0]) + best_dx_mm
    total_ty_mm = float(init_tx[1]) + best_dy_mm
    logger.info("Align: centroid edge_score=%.1f → optimal+(%.2f, %.2f) mm "
                "edge_score=%.1f (total=%.2f, %.2f) mm",
                centroid_score, best_dx_mm, best_dy_mm,
                pass_best_score, total_tx_mm, total_ty_mm)

    # ── 3. Draw final aligned back mask for consensus — from RAW points ──
    bm = _draw_mask(back_raw_mir)

    # ── 4. Base consensus: intersection + minimal CLOSE ──
    inter = cv2.bitwise_and(fm, bm)
    union = cv2.bitwise_or(fm, bm)
    union_area_mm2 = float(np.sum(union > 0)) / (SCALE * SCALE)

    # Fill sub-pixel alignment gaps (0.3mm kernel)
    close_kpx = max(3, int(0.3 * SCALE))
    if close_kpx % 2 == 0:
        close_kpx += 1
    k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_kpx, close_kpx))
    inter = cv2.morphologyEx(inter, cv2.MORPH_CLOSE, k_close)
    inter = cv2.bitwise_and(inter, union)  # clamp
    inter = cv2.bitwise_or(inter, cv2.bitwise_and(fm, bm))

    # ── 5. Full 2D difference: XOR of the two aligned masks ──
    diff = cv2.bitwise_xor(fm, bm)

    # ── 6. Compute convex hull of intersection ──
    #    Any difference pixel INSIDE the hull is one side's indentation
    #    that the other side fills → shadow burr candidate.
    inter_cnts, _ = cv2.findContours(inter, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not inter_cnts:
        return front_ol

    inter_cnt = max(inter_cnts, key=cv2.contourArea)
    hull_mask = np.zeros((ch, cw), dtype=np.uint8)
    hull_pts = cv2.convexHull(inter_cnt)
    cv2.fillPoly(hull_mask, [hull_pts], 255)

    # diff inside hull → indentations that one side fills → burr candidates
    diff_burrs = cv2.bitwise_and(diff, hull_mask)

    # ── 7. Connected-component analysis on burr candidates ──
    #    Small isolated components: shadow burr → fill
    #    Large components: possibly misalignment → warn, keep
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(diff_burrs, connectivity=4)

    burr_max_area_mm2 = max(union_area_mm2 * 0.015, 6.0)  # adaptive: ≥6mm², scales with board
    burr_max_area_px = int(burr_max_area_mm2 * SCALE * SCALE)
    fill_mask = np.zeros((ch, cw), dtype=np.uint8)
    small_burrs = 0
    large_diffs = 0

    for label_id in range(1, n_labels):  # skip background (0)
        area_px = stats[label_id, cv2.CC_STAT_AREA]
        if area_px <= burr_max_area_px:
            fill_mask[labels == label_id] = 255
            small_burrs += 1
        else:
            area_mm2 = area_px / (SCALE * SCALE)
            logger.warning("Large diff region (%.1fmm²) inside hull — possible misalignment", area_mm2)
            large_diffs += 1

    # ── 8. Fill burrs → result = inter + filled burr areas ──
    if small_burrs > 0:
        result = cv2.bitwise_or(inter, fill_mask)
        # Small CLOSE to smooth filled boundaries
        smooth_kpx = max(3, int(0.3 * SCALE))
        if smooth_kpx % 2 == 0:
            smooth_kpx += 1
        k_smooth = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (smooth_kpx, smooth_kpx))
        result = cv2.morphologyEx(result, cv2.MORPH_CLOSE, k_smooth)
        result = cv2.bitwise_and(result, union)
    else:
        result = inter

    # ── 8b. Morphological OPEN to trim thin protrusions ──
    # After consensus, narrow leftover protrusions (thin "spikes") may remain
    # on the boundary from single-side shadow artifacts.  Use a SMALL 0.5mm
    # kernel: the intersection already removes one-sided burrs, so this only
    # cleans up tiny intersection artifacts.  A large kernel (e.g. 2.0mm)
    # would round genuine sharp corners (r≈0.3mm) into ~1mm arcs, which the
    # downstream geometry refinement then mis-detects as large corner radii.
    open_kpx = max(3, int(0.5 * SCALE))
    if open_kpx % 2 == 0:
        open_kpx += 1
    k_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_kpx, open_kpx))
    result = cv2.morphologyEx(result, cv2.MORPH_OPEN, k_open)
    result = cv2.bitwise_and(result, union)  # clamp to union

    # ── 9. Extract clean polygon + hull burr prune ──
    poly = _extract_clean_polygon(result, SCALE, w_mm, h_mm, raw_pts=False)
    if len(poly) >= 5:
        poly = _prune_hull_burrs(poly)

    # ── Log metrics ──
    f_area = np.sum(fm > 0) / (SCALE * SCALE)
    b_area = np.sum(bm > 0) / (SCALE * SCALE)
    u_area = np.sum(union > 0) / (SCALE * SCALE)
    i_area = np.sum(cv2.bitwise_and(fm, bm) > 0) / (SCALE * SCALE)
    r_area = np.sum(result > 0) / (SCALE * SCALE)
    overlap_pct = (i_area / max(u_area, 1)) * 100.0

    logger.info(
        "Front/back 2D consensus: front=%.1fmm² back=%.1fmm², "
        "overlap=%.0f%%, burrs_filled=%d large_diffs=%d "
        "→ merged=%.1fmm², %d vertices",
        f_area, b_area, overlap_pct, small_burrs, large_diffs, r_area, len(poly))

    if len(poly) == 0:
        # Return raw areas from contourArea as fallback
        f_fallback = float(abs(cv2.contourArea(front_pts.reshape(-1, 1, 2).astype(np.float32))))
        b_fallback = float(abs(cv2.contourArea(back_pts.reshape(-1, 1, 2).astype(np.float32))))
        return front_ol, f_fallback, b_fallback, f_fallback

    outline = [{"x_mm": round(float(x), 3), "y_mm": round(float(y), 3)}
               for x, y in poly.tolist()]
    return outline, round(f_area, 1), round(b_area, 1), round(r_area, 1)


def _extract_clean_polygon(mask, SCALE, w_mm, h_mm, raw_pts=False):
    """Extract a clean, simplified polygon from a binary mask.

    Uses moderate epsilon + collinear removal to produce a clean polygon
    with minimal vertices while preserving rounded corners and notches.
    """
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return np.array([])
    cnt = max(cnts, key=cv2.contourArea)
    if raw_pts:
        return cnt.reshape(-1, 2) / SCALE
    peri = cv2.arcLength(cnt, True)
    eps = peri * 0.0008  # moderate: preserves rounded corners (r≥2mm) and notches
    poly = cv2.approxPolyDP(cnt, eps, True).reshape(-1, 2) / SCALE
    poly = _remove_collinear_pts(poly, angle_tol_deg=5.0)
    return poly


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
    # Uses mask-based consensus in a shared canvas (same SCALE, same mm
    # reference).  Returns areas computed from the mask — i.e. the SAME
    # processing pipeline for all three numbers, unlike raw contour areas
    # which come from a different computation path.
    w_mm = steps[0].get("frame_w_mm", 60.0) if steps else 60.0
    h_mm = steps[0].get("frame_h_mm", 40.0) if steps else 40.0
    merged_outline, f_area, b_area, merged_area = _merge_front_back_outlines(
        front_ol, back_ol, w_mm, h_mm, front_ppm, back_ppm)

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
    frame_w_mm: float = Form(60.0),
    frame_h_mm: float = Form(30.0),
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
    frame_w_mm: float = Form(60.0),
    frame_h_mm: float = Form(30.0),
):
    """Detect the black frame in an uploaded photo and return an annotated preview."""
    img_buf = file.file.read()
    target_aspect = frame_w_mm / frame_h_mm if frame_h_mm > 0 else None
    result = _detect_black_frame(img_buf, target_aspect)
    return result


@app.post("/api/vision/calibrate-black-frame")
def calibrate_black_frame_endpoint(
    file: UploadFile = File(...),
    frame_w_mm: float = Form(60.0),
    frame_h_mm: float = Form(30.0),
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
