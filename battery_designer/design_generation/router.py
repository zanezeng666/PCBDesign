"""Step 4 — Design generation API endpoints.

Moved from ``routers/project.py``.  Contains the 7 project CRUD endpoints:
- ``POST /api/projects`` — create project
- ``GET /api/projects/{project_id}`` — get project
- ``POST /api/projects/{project_id}/preview`` — generate preview
- ``POST /api/projects/{project_id}/approve-candidate``
- ``POST /api/projects/{project_id}/manufacturing``
- ``POST /api/projects/{project_id}/validation``
- ``GET /api/projects/{project_id}/artifacts/{artifact_path}``
"""

from __future__ import annotations

import json
import logging
import shutil

from fastapi import APIRouter
from fastapi.responses import FileResponse

from ..core.errors import DesignError
from ..core.models import DesignSpec, ValidationRecord
from ..core.config import WORK_ROOT
from ..core.singletons import store, catalog, generator

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/api/projects", status_code=201)
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


@router.get("/api/projects/{project_id}")
def get_project(project_id: str):
    return store.read(project_id)


@router.post("/api/projects/{project_id}/preview")
def generate_preview(project_id: str):
    project = store.read(project_id)
    spec = DesignSpec.model_validate(project["spec"])
    device = catalog.resolve(project["device"]["full_mpn"])
    result = generator.generate_preview(spec, device, store.directory(project_id))
    store.update(project_id, status="preview_ready", preview=result)
    return result


@router.post("/api/projects/{project_id}/approve-candidate")
def approve_candidate(project_id: str):
    return store.update(project_id, approved=True, approval_notice="Candidate sample build approved; not approved for mass production.")


@router.post("/api/projects/{project_id}/manufacturing")
def generate_manufacturing(project_id: str):
    project = store.read(project_id)
    spec = DesignSpec.model_validate(project["spec"])
    device = catalog.resolve(project["device"]["full_mpn"])
    result = generator.generate_manufacturing(spec, device, store.directory(project_id), approved=bool(project.get("approved")))
    store.update(project_id, status="manufacturing_ready", manufacturing=result)
    return result


@router.post("/api/projects/{project_id}/validation")
def record_validation(project_id: str, record: ValidationRecord):
    if not record.passed:
        raise DesignError("HARDWARE_VALIDATION_FAILED", "All required hardware tests must pass before validation.")
    project = store.read(project_id)
    device = catalog.resolve(project["device"]["full_mpn"])
    promoted = catalog.promote(device, record.model_dump(mode="json"))
    store.write_json(project_id, "hardware-validation.json", record.model_dump(mode="json"))
    return store.update(project_id, hardware_validated=True, validation=record.model_dump(mode="json"), device=promoted.as_dict())


@router.get("/api/projects/{project_id}/artifacts/{artifact_path:path}")
def artifact(project_id: str, artifact_path: str):
    return FileResponse(store.artifact(project_id, artifact_path))
