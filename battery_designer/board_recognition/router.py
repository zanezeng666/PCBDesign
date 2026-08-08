"""PCB recognition API router.

All vision / PCB detection endpoints extracted from the original monolithic
``app.py``.  Mounted under the main FastAPI app via ``include_router``.
"""
from __future__ import annotations

import base64
import json
import logging

import cv2
import numpy as np
from fastapi import APIRouter, File, UploadFile, Form, HTTPException

from ..pad_detection.vlm_detection import detect_all_vlm as _vlm_detect_all
from .vision import (
    extract_pcb as _extract_pcb,
    detect_holes as _detect_holes,
    _make_transparent,
    refine_outline_geometry,
    detect_black_frame as _detect_black_frame,
    calibrate_black_frame as _calibrate_black_frame,
)
from ..core.config import WORK_ROOT
from .pipeline import PCBRecognitionPipeline
from .contour_compare import _compare_front_back_contours
from .overlay import _make_overlay_image

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/api/vision/extract-pcb")
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



@router.post("/api/vision/contour-match")
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



@router.post("/api/vision/detect-holes")
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



@router.post("/api/vision/detect-all")
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



@router.post("/api/simulate")
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



@router.post("/api/vision/preview-black-frame")
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



@router.post("/api/vision/calibrate-black-frame")
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


# ── Unified PCB Recognition ─────────────────────────────────────────────


@router.post("/api/vision/recognize-pcb")
async def recognize_pcb(
    front_image: UploadFile = File(..., description="正面 PCB 照片"),
    back_image: UploadFile = File(..., description="背面 PCB 照片"),
    frame_w_mm: float = Form(40.0, description="黑框宽度 (mm)"),
    frame_h_mm: float = Form(25.0, description="黑框高度 (mm)"),
):
    """统一 PCB 识别接口：输入正反面图片 + 黑框尺寸，输出校验后的图片和真实尺寸。

    流程：
      1. 对正/反面图片各执行完整 PCBRecognitionPipeline（方向检测→黑框检测→透视校正→HSV提取→纸色模型→透明PNG）
      2. 正反面交叉校验，生成共识轮廓
      3. 计算共识轮廓的真实物理尺寸（宽×高 mm）
      4. 对正面/背面分别生成叠加轮廓的校验图

    Returns:
        {
            "success": bool,
            "frame_w_mm": float,            # 黑框宽
            "frame_h_mm": float,            # 黑框高
            "pcb_width_mm": float,          # PCB 真实宽度（轮廓外接矩形宽）
            "pcb_height_mm": float,         # PCB 真实高度（轮廓外接矩形高）
            "pcb_outline": [...],           # 共识轮廓顶点 (mm)
            "outline_vertex_count": int,    # 轮廓顶点数
            "front": {
                "calibration_id": str,
                "pixels_per_mm": float,
                "outline": [...],
                "rectified_png_b64": str,   # 透视校正图
                "transparent_pcb_b64": str, # 透明 PCB 图
                "overlay_b64": str,         # 叠加轮廓校验图
                "calibration_success": bool,
                "error": str | None,
            },
            "back": {
                "calibration_id": str,
                "pixels_per_mm": float,
                "outline": [...],
                "rectified_png_b64": str,
                "transparent_pcb_b64": str,
                "overlay_b64": str,
                "calibration_success": bool,
                "error": str | None,
            },
            "consensus": {
                "outline": [...],           # 共识轮廓
                "transparent_pcb_b64": str, # 共识透明 PNG
                "front_area_mm2": float,
                "back_area_mm2": float,
                "consensus_area_mm2": float,
                "deviation_pct": float,     # 正反面与共识偏差
                "ok": bool,
                "message": str,
            },
        }
    """
    from io import BytesIO

    front_bytes = await front_image.read()
    back_bytes = await back_image.read()

    pl = PCBRecognitionPipeline()

    # ── 处理正面 ──
    front_data = None
    front_error = None
    try:
        front_data = pl.run(
            image_bytes=front_bytes,
            frame_width_mm=frame_w_mm,
            frame_height_mm=frame_h_mm,
        )
    except Exception as e:
        front_error = f"正面识别失败: {e}"
        logger.warning("recognize-pcb: front failed: %s", e, exc_info=True)

    # ── 处理背面 ──
    back_data = None
    back_error = None
    try:
        back_data = pl.run(
            image_bytes=back_bytes,
            frame_width_mm=frame_w_mm,
            frame_height_mm=frame_h_mm,
        )
    except Exception as e:
        back_error = f"背面识别失败: {e}"
        logger.warning("recognize-pcb: back failed: %s", e, exc_info=True)

    # ── 构建返回结构 ──
    def _side_result(data, error):
        if data is None:
            return {
                "calibration_id": "",
                "pixels_per_mm": 0.0,
                "outline": [],
                "rectified_png_b64": "",
                "transparent_pcb_b64": "",
                "overlay_b64": "",
                "calibration_success": False,
                "error": error,
            }
        outline = data["outline"]
        ppm = data["pixels_per_mm"]
        rectified_b64 = data.get("rectified_png_b64", "")
        transparent_b64 = data.get("transparent_pcb_b64", "")

        # 生成叠加轮廓校验图
        overlay_b64 = _make_overlay_image(rectified_b64, outline, ppm,
                                          frame_w_mm, frame_h_mm)
        return {
            "calibration_id": data["calibration_id"],
            "pixels_per_mm": round(ppm, 2),
            "outline": outline,
            "rectified_png_b64": rectified_b64,
            "transparent_pcb_b64": transparent_b64,
            "overlay_b64": overlay_b64,
            "calibration_success": True,
            "error": None,
        }

    front_result = _side_result(front_data, front_error)
    back_result = _side_result(back_data, back_error)

    # ── 交叉校验 ──
    consensus_data = {
        "outline": [],
        "transparent_pcb_b64": "",
        "front_area_mm2": 0.0,
        "back_area_mm2": 0.0,
        "consensus_area_mm2": 0.0,
        "deviation_pct": 0.0,
        "ok": False,
        "message": "正反面数据不完整，无法交叉校验",
    }

    pcb_width_mm = 0.0
    pcb_height_mm = 0.0
    consensus_outline = []

    if front_data and back_data:
        try:
            cross = PCBRecognitionPipeline.cross_validate_front_back(
                front_data, back_data, frame_w_mm, frame_h_mm,
            )
            consensus_outline = cross["outline"]
            consensus_transparent = cross.get("transparent_pcb_b64", "")
            f_area = cross.get("front_area_mm2", 0.0)
            b_area = cross.get("back_area_mm2", 0.0)
            c_area = cross.get("consensus_area_mm2", 0.0)

            deviation = 0.0
            if c_area > 0:
                dev_f = abs(f_area - c_area) / c_area * 100
                dev_b = abs(b_area - c_area) / c_area * 100
                deviation = max(dev_f, dev_b)

            ok = deviation <= 10.0  # 10% 容差
            if ok:
                msg = f"正反面轮廓匹配通过（偏差 {deviation:.1f}%），已合并为精确轮廓"
            else:
                msg = f"正反面轮廓偏差 {deviation:.1f}%，已合并生成参考轮廓"

            consensus_data = {
                "outline": consensus_outline,
                "transparent_pcb_b64": consensus_transparent,
                "front_area_mm2": round(f_area, 2),
                "back_area_mm2": round(b_area, 2),
                "consensus_area_mm2": round(c_area, 2),
                "deviation_pct": round(deviation, 1),
                "ok": ok,
                "message": msg,
            }
        except Exception as e:
            logger.warning("recognize-pcb: consensus failed: %s", e, exc_info=True)
            consensus_data["message"] = f"交叉校验失败: {e}"
    elif front_data:
        consensus_outline = front_data["outline"]
        consensus_data["message"] = "仅正面识别成功，使用正面轮廓"

    # ── 计算 PCB 真实尺寸（轮廓外接矩形） ──
    if consensus_outline and len(consensus_outline) >= 3:
        xs = [p["x_mm"] for p in consensus_outline]
        ys = [p["y_mm"] for p in consensus_outline]
        pcb_width_mm = round(max(xs) - min(xs), 3)
        pcb_height_mm = round(max(ys) - min(ys), 3)

    # ── 如果共识轮廓为空但有单面轮廓，用单面 ──
    if not consensus_outline:
        if front_result["outline"]:
            consensus_outline = front_result["outline"]
        elif back_result["outline"]:
            consensus_outline = back_result["outline"]
        if consensus_outline and len(consensus_outline) >= 3:
            xs = [p["x_mm"] for p in consensus_outline]
            ys = [p["y_mm"] for p in consensus_outline]
            pcb_width_mm = round(max(xs) - min(xs), 3)
            pcb_height_mm = round(max(ys) - min(ys), 3)

    success = front_result["calibration_success"] or back_result["calibration_success"]

    return {
        "success": success,
        "frame_w_mm": frame_w_mm,
        "frame_h_mm": frame_h_mm,
        "pcb_width_mm": pcb_width_mm,
        "pcb_height_mm": pcb_height_mm,
        "pcb_outline": consensus_outline,
        "outline_vertex_count": len(consensus_outline),
        "front": front_result,
        "back": back_result,
        "consensus": consensus_data,
    }





