"""原理图 PNG 预览生成 — SKiDL auto_stub 模式"""
import os, sys, yaml
from pathlib import Path

from .circuit_helpers import (
    init_skidl, build_circuit_from_yaml, create_nets_from_yaml,
    export_sch_png, export_sch_png_direct,
)
from .logger import get_logger

import skidl

_log = get_logger(__name__)


def main(yaml_path, output_dir):
    with open(yaml_path, "r", encoding="utf-8") as f:
        circuit = yaml.safe_load(f)

    custom_lib = init_skidl()
    parts = build_circuit_from_yaml(circuit, custom_lib)
    create_nets_from_yaml(circuit, parts)

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    name = circuit.get("name", "sch").replace(" ", "_")
    sch_file = out / f"{name}.kicad_sch"
    png_file = out / "schematic.png"

    _log.info(f"生成原理图: {sch_file}")
    skidl.generate_schematic(
        filepath=str(sch_file),
        auto_stub=True,
        auto_stub_fallback="warn",
    )
    _log.info(f"OK: {sch_file}")

    _log.info(f"导出 PNG...")
    if not export_sch_png_direct(sch_file, png_file):
        export_sch_png(sch_file, png_file)  # fallback via SVG

    return str(sch_file) if sch_file.exists() else None


if __name__ == "__main__":
    yf = sys.argv[1] if len(sys.argv) > 1 else "engine/circuits/protections/dw01_1s.yaml"
    od = sys.argv[2] if len(sys.argv) > 2 else "output/dw01_sch"
    result = main(yf, od)
    print(f"\n{'DONE' if result else 'FAILED'}: {result}")
