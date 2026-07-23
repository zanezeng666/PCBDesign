from __future__ import annotations

import base64
import json
import logging
import os
import re
import shutil
from uuid import uuid4
from pathlib import Path

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
from .vision import extract_pcb as _extract_pcb, detect_holes as _detect_holes
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


@app.exception_handler(RequestValidationError)
async def validation_error_handler(_, exc: RequestValidationError):
    return JSONResponse(status_code=422, content={"error": {"code": "INPUT_VALIDATION", "message": "Input validation failed.", "details": {"errors": exc.errors()}}})


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


@app.get("/api/ic/resolve/{model}")
def resolve_ic(model: str):
    return catalog.resolve(model).as_dict()


@app.post("/api/vision/detect-terminals")
def detect_terminals(calibration_id: str = Form(...), side: str = Form(...)):
    """Detect terminal candidates on a rectified PCB image using qwen3.7-plus VLM."""
    if len(calibration_id) != 32 or any(character not in "0123456789abcdef" for character in calibration_id):
        raise DesignError("INVALID_CALIBRATION_ID", "The calibration id is invalid.")
    directory = WORK_ROOT / "calibrations" / calibration_id
    metadata_path = directory / "calibration.json"
    image_path = directory / "rectified.png"
    if not metadata_path.exists() or not image_path.exists():
        raise DesignError("CALIBRATION_NOT_FOUND", "Photo calibration record not found.", {"calibration_id": calibration_id})
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    image_bytes = image_path.read_bytes()
    width_mm = float(metadata["width_mm"])
    height_mm = float(metadata["height_mm"])

    result = _vlm_detect(image_bytes, width_mm, height_mm, side)
    result["method_used"] = "vlm"

    (directory / "terminal-candidates.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    if result.get("annotated_png_base64"):
        annotated = base64.b64decode(result["annotated_png_base64"])
        (directory / "terminal-annotated.png").write_bytes(annotated)
    return result


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
def api_extract_pcb(calibration_id: str = Form(...)):
    """Extract PCB outline from white A4 paper with shadow removal.

    Uses Qwen VLM to detect the PCB board on white paper background,
    then CV to remove photo shadows and refine the outline polygon.
    Also detects edge grooves/protrusions (≤6).
    """
    png, w_mm, h_mm, ppm = _load_calibration(calibration_id)
    result = _extract_pcb(png, w_mm, h_mm, ppm)

    directory = WORK_ROOT / "calibrations" / calibration_id
    (directory / "pcb_outline.json").write_text(
        json.dumps({"outline": result["outline"], "grooves": result["grooves"],
                     "pixels_per_mm": ppm}, ensure_ascii=False, indent=2),
        encoding="utf-8")

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
    # Use adjusted frame dimensions from the first successful step
    adj_w = steps[0].get("frame_w_mm", frame_w_mm) if steps else frame_w_mm
    adj_h = steps[0].get("frame_h_mm", frame_h_mm) if steps else frame_h_mm
    return {
        "success": all_ok,
        "total_images": total,
        "frame_w_mm": adj_w,
        "frame_h_mm": adj_h,
        "steps": steps,
    }


@app.post("/api/vision/preview-black-frame")
def preview_black_frame(file: UploadFile = File(...)):
    """Detect the black frame in an uploaded photo and return an annotated preview."""
    img_buf = file.file.read()
    result = _detect_black_frame(img_buf)
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
