"""Engine shared configuration — resolves KiCad paths across platforms."""
from __future__ import annotations

import os
import sys
from pathlib import Path


def _find_kicad_root() -> Path:
    """Resolve KiCad installation root from env or common paths.

    Priority:
    1. KICAD_PATH or KICAD_BIN env var
    2. Platform-specific well-known locations (newest version first)
    3. Raise FileNotFoundError
    """
    for env_var in ("KICAD_PATH", "KICAD_BIN"):
        val = os.getenv(env_var)
        if val:
            candidate = Path(val)
            if candidate.exists():
                # If pointing to bin/, go up one level to get the root
                return candidate.parent if candidate.is_dir() and candidate.name == "bin" else candidate

    candidates: list[Path] = []
    if sys.platform == "win32":
        for ver in ("9.0", "8.0", "7.0"):
            candidates.append(Path(f"C:\\Program Files\\KiCad\\{ver}"))
            candidates.append(Path(f"C:\\Program Files (x86)\\KiCad\\{ver}"))
    elif sys.platform == "darwin":
        candidates.append(Path("/Applications/KiCad/KiCad.app/Contents/Applications"))
    else:
        candidates.extend([Path("/usr/share/kicad"), Path("/usr/lib/kicad")])

    for c in candidates:
        if c.exists():
            return c

    raise FileNotFoundError(
        "KiCad not found. Set KICAD_PATH or KICAD_BIN environment variable, "
        "or install KiCad to a standard location."
    )


# ── resolved paths ──
KICAD_ROOT: Path = _find_kicad_root()
KICAD_BIN: Path = KICAD_ROOT / "bin"
KICAD_SHARE: Path = KICAD_ROOT / "share" / "kicad"
KICAD_SYMBOL_DIR: Path = KICAD_SHARE / "symbols"
KICAD_FOOTPRINT_DIR: Path = KICAD_SHARE / "footprints"

# kicad-cli executable (cross-platform)
if sys.platform == "win32":
    KICAD_CLI: Path = KICAD_BIN / "kicad-cli.exe"
else:
    KICAD_CLI: Path = KICAD_BIN / "kicad-cli"

# ── set env vars for skidl ──
os.environ["KICAD_SYMBOL_DIR"] = str(KICAD_SYMBOL_DIR)
os.environ["KICAD9_SYMBOL_DIR"] = str(KICAD_SYMBOL_DIR)
os.environ["KICAD8_SYMBOL_DIR"] = str(KICAD_SYMBOL_DIR)
os.environ["KICAD7_SYMBOL_DIR"] = str(KICAD_SYMBOL_DIR)
os.environ["KICAD6_SYMBOL_DIR"] = str(KICAD_SYMBOL_DIR)
os.environ["KICAD_FOOTPRINT_DIR"] = str(KICAD_FOOTPRINT_DIR)

# Add KiCad bin to PATH on Windows for DLL loading
if sys.platform == "win32":
    os.environ["PATH"] = str(KICAD_BIN) + os.pathsep + os.environ.get("PATH", "")
    try:
        os.add_dll_directory(str(KICAD_BIN))
    except (AttributeError, OSError):
        pass  # add_dll_directory not available in all Python versions


# ── project-internal paths ──
ENGINE_DIR: Path = Path(__file__).resolve().parent
CUSTOM_SYM_LIB: Path = ENGINE_DIR / "circuits" / "symbols" / "battery_protection.kicad_sym"
