from __future__ import annotations

import base64
import json
import os
import shutil
from uuid import uuid4
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
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
from .terminal_detection import detect_terminal_candidates
from .vision import calibrate_known_size, calibrate_photo, outline_alignment_error


ROOT = Path(__file__).resolve().parents[1]
WORK_ROOT = Path(os.getenv("BATTERY_DESIGN_WORKDIR", ROOT / "work"))
STATIC_ROOT = ROOT / "web"
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


@app.exception_handler(DesignError)
async def design_error_handler(_, exc: DesignError):
    return JSONResponse(status_code=exc.status_code, content=exc.as_dict())


@app.exception_handler(RequestValidationError)
async def validation_error_handler(_, exc: RequestValidationError):
    return JSONResponse(status_code=422, content={"error": {"code": "INPUT_VALIDATION", "message": "Input validation failed.", "details": {"errors": exc.errors()}}})


@app.get("/")
def index():
    return FileResponse(STATIC_ROOT / "index.html")


@app.get("/api/health")
def health():
    return {"status": "ok", "kicad": pipeline.diagnose(), "resolver_configured": bool(catalog.resolver_endpoint)}


@app.get("/api/ic/resolve/{model}")
def resolve_ic(model: str):
    return catalog.resolve(model).as_dict()


@app.post("/api/vision/calibrate")
async def calibrate(file: UploadFile = File(...), marker_size_mm: float = Form(...)):
    image = await file.read()
    if len(image) > 25 * 1024 * 1024:
        raise DesignError("IMAGE_TOO_LARGE", "Images are limited to 25 MB.")
    result = calibrate_photo(image, marker_size_mm)
    calibration_id = uuid4().hex
    directory = WORK_ROOT / "calibrations" / calibration_id
    directory.mkdir(parents=True)
    (directory / "original.bin").write_bytes(image)
    (directory / "rectified.png").write_bytes(result.rectified_png)
    (directory / "preview.png").write_bytes(result.preview_png)
    response = {"calibration_id": calibration_id, **result.response()}
    (directory / "calibration.json").write_text(json.dumps(response, ensure_ascii=False, indent=2), encoding="utf-8")
    return response


@app.post("/api/vision/calibrate-known-size")
async def calibrate_by_known_size(
    file: UploadFile = File(...), width_mm: float = Form(...), height_mm: float = Form(...)
):
    image = await file.read()
    if len(image) > 25 * 1024 * 1024:
        raise DesignError("IMAGE_TOO_LARGE", "Images are limited to 25 MB.")
    result = calibrate_known_size(image, width_mm, height_mm)
    calibration_id = uuid4().hex
    directory = WORK_ROOT / "calibrations" / calibration_id
    directory.mkdir(parents=True)
    (directory / "original.bin").write_bytes(image)
    (directory / "rectified.png").write_bytes(result.rectified_png)
    (directory / "preview.png").write_bytes(result.preview_png)
    response = {"calibration_id": calibration_id, **result.response()}
    (directory / "calibration.json").write_text(json.dumps(response, ensure_ascii=False, indent=2), encoding="utf-8")
    return response


@app.post("/api/vision/detect-terminals")
def detect_terminals(calibration_id: str = Form(...), side: str = Form(...)):
    if len(calibration_id) != 32 or any(character not in "0123456789abcdef" for character in calibration_id):
        raise DesignError("INVALID_CALIBRATION_ID", "The calibration id is invalid.")
    directory = WORK_ROOT / "calibrations" / calibration_id
    metadata_path = directory / "calibration.json"
    image_path = directory / "rectified.png"
    if not metadata_path.exists() or not image_path.exists():
        raise DesignError("CALIBRATION_NOT_FOUND", "Photo calibration record not found.", {"calibration_id": calibration_id})
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    result = detect_terminal_candidates(
        image_path.read_bytes(), float(metadata["width_mm"]), float(metadata["height_mm"]), side
    )
    (directory / "terminal-candidates.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    annotated = base64.b64decode(result["annotated_png_base64"])
    (directory / "terminal-annotated.png").write_bytes(annotated)
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

    front = load(capture.front_calibration_id)
    if not capture.back_calibration_id:
        return None
    back = load(capture.back_calibration_id)
    error = outline_alignment_error(front, back, capture.back_transform.value)
    if error > 0.5:
        raise DesignError(
            "PHOTO_ALIGNMENT_FAILED",
            "Front/back outlines do not align within 0.5 mm after the selected transform.",
            {"alignment_error_mm": round(error, 3), "back_transform": capture.back_transform.value},
        )
    return round(error, 3)


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


def main() -> None:
    import uvicorn

    uvicorn.run("battery_designer.app:app", host="127.0.0.1", port=8000, reload=False)


if __name__ == "__main__":
    main()
