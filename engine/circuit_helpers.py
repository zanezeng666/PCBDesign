"""Shared circuit building utilities — used by schematic.py, schematic_png.py, pcb2.py."""
from __future__ import annotations

import os
from pathlib import Path

from .config import CUSTOM_SYM_LIB, KICAD_CLI
from .logger import get_logger

import skidl
from skidl import Net, Part, KICAD9, SchLib

_log = get_logger(__name__)


def init_skidl() -> SchLib | None:
    """Initialize SKiDL with KiCad 9 and load custom symbol library."""
    skidl.set_default_tool(KICAD9)
    return SchLib(CUSTOM_SYM_LIB) if os.path.exists(CUSTOM_SYM_LIB) else None


def create_part(part_def: dict, custom_lib: SchLib | None) -> Part | None:
    """Create a SKiDL Part from a YAML part definition.

    Handles 4 categories: IC/MOSFET_DUAL (custom lib), RES, CAP, CONN.
    Falls back gracefully when a preferred symbol is not found.
    """
    pid = part_def["id"]
    pname = part_def["part_name"]
    cat = part_def["category"]
    fp = part_def["footprint"]

    if cat in ("IC", "MOSFET_DUAL") and custom_lib:
        p = Part(custom_lib, pname, footprint=fp)
    elif cat == "RES":
        try:
            p = Part("Device", "R_Small_US", footprint=fp)
        except Exception:
            p = Part("Device", "R", footprint=fp)
    elif cat == "CAP":
        try:
            p = Part("Device", "C_Small", footprint=fp)
        except Exception:
            p = Part("Device", "C", footprint=fp)
    elif cat == "CONN":
        try:
            p = Part("Connector_Generic", "Conn_01x04", footprint=fp)
        except Exception:
            p = Part("Connector", "Conn_01x04", footprint=fp)
    else:
        return None

    p.ref = pid
    p.value = part_def["value"]
    return p


def build_circuit_from_yaml(circuit: dict, custom_lib: SchLib | None) -> dict[str, Part]:
    """Create all Parts from a circuit YAML definition.

    Returns:
        dict mapping part IDs to SKiDL Part instances.
    """
    parts: dict[str, Part] = {}
    for part_def in circuit["parts"]:
        p = create_part(part_def, custom_lib)
        if p is not None:
            parts[part_def["id"]] = p
    return parts


def create_nets_from_yaml(circuit: dict, parts: dict[str, Part]) -> dict[str, Net]:
    """Create all nets from circuit definition and connect them to parts.

    Non-critical connection failures are silently skipped.
    """
    nets: dict[str, Net] = {}
    for net_def in circuit["nets"]:
        net = Net(net_def["name"])
        nets[net_def["name"]] = net
        for conn in net_def["connections"]:
            part_id = conn["part"]
            pin_name = str(conn["pin"])
            if part_id in parts:
                try:
                    parts[part_id][pin_name] += net
                except Exception:
                    pass
    return nets


def export_sch_png(sch_path: Path, png_path: Path, width: int = 1600) -> bool:
    """Export schematic SVG to PNG using kicad-cli + cairosvg.

    Returns True on success.
    """
    import subprocess

    svg_file = png_path.with_suffix(".svg")
    r = subprocess.run(
        [str(KICAD_CLI), "sch", "export", "svg", str(sch_path), "--output", str(svg_file)],
        capture_output=True, text=True,
    )
    if r.returncode != 0:
        _log.error(f"SVG export failed: {r.stderr[:200]}")
        return False

    import cairosvg
    cairosvg.svg2png(url=str(svg_file), write_to=str(png_path), output_width=width)
    return True


def export_sch_png_direct(sch_path: Path, png_path: Path) -> bool:
    """Export schematic directly to PNG using kicad-cli (no cairosvg required).

    Returns True on success.
    """
    import subprocess

    r = subprocess.run(
        [str(KICAD_CLI), "sch", "export", "png", str(sch_path),
         "--output", str(png_path), "--background", "opaque"],
        capture_output=True, text=True,
    )
    if r.returncode == 0:
        _log.info(f"PNG: {png_path} ({os.path.getsize(png_path)} bytes)")
        return True

    _log.error(f"PNG export failed: {r.stderr[:200]}")
    return False
