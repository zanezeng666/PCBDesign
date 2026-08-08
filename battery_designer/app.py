"""Battery Protection Board Designer — FastAPI application.

This module is intentionally thin: it creates the ``FastAPI`` app, configures
middleware / exception handlers, and mounts all feature routers.

Business logic lives in four functional packages matching the system's
four-step pipeline:
  * ``battery_designer.board_recognition``   — Step 1: PCB board recognition
  * ``battery_designer.pad_detection``       — Step 2: Pad / terminal detection
  * ``battery_designer.component_detection`` — Step 3: Component detection
  * ``battery_designer.design_generation``   — Step 4: Design generation
"""
from __future__ import annotations

import logging
import traceback

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from .core.errors import DesignError
from .core.config import STATIC_ROOT, DATA_ROOT

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
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
    """Disable browser caching for all responses during development."""
    response = await call_next(request)
    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    return response


# ---------------------------------------------------------------------------
# Exception handlers
# ---------------------------------------------------------------------------
@app.exception_handler(DesignError)
async def design_error_handler(_, exc: DesignError):
    return JSONResponse(status_code=exc.status_code, content=exc.as_dict())


def _json_safe(value):
    """Recursively coerce to JSON-serialisable structures."""
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
    """Catch all unhandled exceptions and return JSON instead of HTML."""
    tb_lines = traceback.format_exception(type(exc), exc, exc.__traceback__)
    return JSONResponse(
        status_code=500,
        content={
            "error": {
                "code": "INTERNAL_ERROR",
                "message": str(exc),
                "details": {
                    "type": type(exc).__name__,
                    "traceback": "".join(tb_lines[-4:]),
                },
            }
        },
    )


# ---------------------------------------------------------------------------
# Index
# ---------------------------------------------------------------------------
@app.get("/")
def index():
    return FileResponse(STATIC_ROOT / "index.html")


# ---------------------------------------------------------------------------
# Routers — mounted by functional package
# ---------------------------------------------------------------------------
from .board_recognition.router import router as _board_router
from .board_recognition.calibration_api import router as _calibration_router
from .pad_detection.router import router as _pad_router
from .component_detection.router import router as _component_router
from .design_generation.router import router as _project_router
from .core.router import router as _system_router

app.include_router(_board_router)
app.include_router(_calibration_router)
app.include_router(_pad_router)
app.include_router(_component_router)
app.include_router(_project_router)
app.include_router(_system_router)


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------
def main() -> None:
    import uvicorn
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8000)
    args, _ = parser.parse_known_args()
    uvicorn.run("battery_designer.app:app", host="127.0.0.1", port=args.port, reload=False)


if __name__ == "__main__":
    main()
