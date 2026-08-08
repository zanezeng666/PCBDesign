"""Global singleton instances — single source of truth.

Consolidates singleton initialisation that was previously duplicated across
``shared.py``, ``app.py``, ``routers/project.py``, ``routers/ic_catalog.py``,
and ``routers/system.py``.

Import path notes: ``IcCatalog``, ``KicadPipeline``, and ``DesignGenerator``
are imported from their current top-level locations.  These imports will be
updated as those modules are relocated to ``component_detection/`` and
``design_generation/`` in subsequent refactoring steps.
"""

from __future__ import annotations

from .config import ROOT, WORK_ROOT
from .storage import ProjectStore
from ..component_detection.catalog import IcCatalog
from ..design_generation.kicad import KicadPipeline
from ..design_generation.generator import DesignGenerator

# ── Singletons (instantiated once at import time) ────────────────────

store = ProjectStore(WORK_ROOT / "projects")
catalog = IcCatalog(ROOT / "data" / "ic_catalog", WORK_ROOT / "ic_cache")
pipeline = KicadPipeline()
generator = DesignGenerator(pipeline)
