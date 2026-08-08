"""Centralised path constants used across all modules."""

from __future__ import annotations

import os
from pathlib import Path

# Project root (two levels up from this file: core/ → battery_designer/ → project root)
ROOT: Path = Path(__file__).resolve().parents[2]

# Working directory for generated projects (overridable via env var)
WORK_ROOT: Path = Path(os.getenv("BATTERY_DESIGN_WORKDIR", str(ROOT / "work")))

# Static assets directory (web frontend)
STATIC_ROOT: Path = ROOT / "web"

# Data directory (catalogs, calibration, optimization)
DATA_ROOT: Path = ROOT / "data"

# Calibration images directory
CALIB_DIR: Path = DATA_ROOT / "calibration"
