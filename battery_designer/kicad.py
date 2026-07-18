from __future__ import annotations

import csv
import json
import os
import shutil
import subprocess
from pathlib import Path

from .catalog import DevicePackage
from .errors import DesignError
from .models import DesignSpec

_KICAD_BIN = Path(os.getenv("KICAD_BIN", r"C:\Program Files\KiCad\9.0\bin"))
_DLL_HANDLE = None
if os.name == "nt" and _KICAD_BIN.exists():
    os.environ["PATH"] = str(_KICAD_BIN) + os.pathsep + os.environ.get("PATH", "")
    _DLL_HANDLE = os.add_dll_directory(str(_KICAD_BIN))

import cairosvg


class KicadPipeline:
    def __init__(self, kicad_cli: Path | None = None):
        configured = os.getenv("KICAD_CLI")
        self.kicad_cli = Path(configured) if configured else (kicad_cli or Path(r"C:\Program Files\KiCad\9.0\bin\kicad-cli.exe"))

    def diagnose(self) -> dict:
        if not self.kicad_cli.exists():
            return {"available": False, "path": str(self.kicad_cli), "reason": "not found"}
        result = subprocess.run(
            [str(self.kicad_cli), "--version"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        return {
            "available": result.returncode == 0,
            "path": str(self.kicad_cli),
            "version": (result.stdout or "").strip(),
            "stderr": (result.stderr or "").strip(),
        }

    def build(self, spec: DesignSpec, device: DevicePackage, output_dir: Path) -> dict:
        if not device.template_dir:
            raise DesignError(
                "IC_TEMPLATE_NOT_READY",
                "The resolved IC has metadata but no reviewed KiCad template, so manufacturing files cannot be generated safely.",
                {"full_mpn": device.full_mpn, "status": device.status},
            )
        template = Path(device.template_dir)
        manifest_path = template / "template.json"
        if not manifest_path.exists():
            raise DesignError("TEMPLATE_MANIFEST_MISSING", "The KiCad template is missing template.json.")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        schematic_source = template / manifest["schematic"]
        pcb_source = template / manifest["pcb"]
        if not schematic_source.exists() or not pcb_source.exists():
            raise DesignError("TEMPLATE_FILES_MISSING", "The template schematic or PCB file is missing.")

        design_dir = output_dir / "kicad"
        if design_dir.exists():
            shutil.rmtree(design_dir)
        shutil.copytree(template, design_dir)
        schematic = design_dir / manifest["schematic"]
        pcb = design_dir / manifest["pcb"]
        reports = output_dir / "reports"
        manufacturing = output_dir / "manufacturing"
        gerber = manufacturing / "gerber"
        reports.mkdir(parents=True, exist_ok=True)
        gerber.mkdir(parents=True, exist_ok=True)

        # Template-specific geometry/routing adaptation is required before the
        # hard gates. The manifest adapter is an executable supplied with the
        # reviewed device pack; no generic unsafe fallback is allowed.
        adapter = design_dir / manifest.get("adapter", "")
        if not manifest.get("adapter") or not adapter.exists():
            raise DesignError("TEMPLATE_ADAPTER_MISSING", "The reviewed template has no geometry/routing adapter.")
        spec_path = design_dir / "design-input.json"
        spec_path.write_text(spec.model_dump_json(indent=2), encoding="utf-8")
        self._run([str(adapter), str(spec_path), str(pcb)], "TEMPLATE_ADAPT_FAILED")

        erc = reports / "erc.rpt"
        drc = reports / "drc.rpt"
        self._run([str(self.kicad_cli), "sch", "erc", "--exit-code-violations", "-o", str(erc), str(schematic)], "ERC_FAILED")
        self._run([str(self.kicad_cli), "pcb", "drc", "--exit-code-violations", "--all-track-errors", "-o", str(drc), str(pcb)], "DRC_FAILED")
        self._assert_no_unconnected(drc)

        preview = output_dir / "preview"
        preview.mkdir(exist_ok=True)
        self._run([str(self.kicad_cli), "sch", "export", "svg", "-o", str(preview), str(schematic)], "SCHEMATIC_SVG_FAILED")
        for side, layers in (("front", "F.Cu,F.Mask,F.Silkscreen,Edge.Cuts"), ("back", "B.Cu,B.Mask,B.Silkscreen,Edge.Cuts")):
            self._run([str(self.kicad_cli), "pcb", "export", "svg", "--layers", layers, "-o", str(preview / f"pcb_{side}.svg"), str(pcb)], "PCB_SVG_FAILED")
        for svg in preview.rglob("*.svg"):
            cairosvg.svg2png(url=str(svg), write_to=str(svg.with_suffix(".png")), output_width=1800)
        self._run([str(self.kicad_cli), "pcb", "export", "gerbers", "-o", str(gerber), str(pcb)], "GERBER_FAILED")
        self._run([str(self.kicad_cli), "pcb", "export", "drill", "-o", str(gerber), str(pcb)], "DRILL_FAILED")
        bom = manufacturing / "bom.csv"
        self._run([str(self.kicad_cli), "sch", "export", "bom", "-o", str(bom), str(schematic)], "BOM_FAILED")
        position = manufacturing / "positions.csv"
        self._run([str(self.kicad_cli), "pcb", "export", "pos", "--format", "csv", "--units", "mm", "--side", "both", "-o", str(position), str(pcb)], "POSITION_FAILED")
        _write_jlc_files(bom, position, manufacturing)
        return {"schematic": str(schematic), "pcb": str(pcb), "erc": str(erc), "drc": str(drc)}

    @staticmethod
    def _assert_no_unconnected(report: Path) -> None:
        content = report.read_text(encoding="utf-8", errors="replace").lower()
        if "unconnected_items" in content or "未连接" in content:
            raise DesignError("UNCONNECTED_ITEMS", "KiCad DRC reports unconnected items.", {"report": str(report)})

    @staticmethod
    def _run(command: list[str], code: str) -> subprocess.CompletedProcess:
        result = subprocess.run(command, capture_output=True, text=True, encoding="utf-8", errors="replace")
        if result.returncode != 0:
            raise DesignError(code, "External design command failed.", {"command": command, "stdout": result.stdout[-2000:], "stderr": result.stderr[-2000:]})
        return result


def _write_jlc_files(bom: Path, positions: Path, output_dir: Path) -> None:
    # Preserve KiCad originals and add deterministic JLCPCB field mappings.
    with bom.open("r", encoding="utf-8-sig", newline="") as source, (output_dir / "bom_jlc.csv").open("w", encoding="utf-8-sig", newline="") as target:
        rows = list(csv.reader(source))
        writer = csv.writer(target)
        writer.writerow(["Comment", "Designator", "Footprint", "LCSC Part #"])
        for row in rows[1:]:
            padded = row + [""] * 5
            writer.writerow([padded[1], padded[0], padded[2], ""])
    with positions.open("r", encoding="utf-8-sig", newline="") as source, (output_dir / "cpl_jlc.csv").open("w", encoding="utf-8-sig", newline="") as target:
        reader = csv.DictReader(source)
        writer = csv.writer(target)
        writer.writerow(["Designator", "Mid X", "Mid Y", "Layer", "Rotation"])
        for row in reader:
            writer.writerow([row.get("Ref", row.get("Designator", "")), row.get("PosX", row.get("Mid X", "")), row.get("PosY", row.get("Mid Y", "")), row.get("Side", row.get("Layer", "")), row.get("Rot", row.get("Rotation", ""))])
