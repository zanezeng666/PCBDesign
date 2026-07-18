from __future__ import annotations

import hashlib
import json
import zipfile
from datetime import datetime, timezone
from pathlib import Path

from .catalog import DevicePackage, validate_ic_for_design
from .errors import DesignError
from .kicad import KicadPipeline
from .models import DesignSpec
from .mos import MosfetSelection, select_mosfets
from .ocp import assess_overcurrent_target
from .preview import write_mechanical_previews


class DesignGenerator:
    def __init__(self, pipeline: KicadPipeline):
        self.pipeline = pipeline

    def preflight(self, spec: DesignSpec, device: DevicePackage) -> MosfetSelection:
        if not spec.outline.confirmed:
            raise DesignError("OUTLINE_NOT_CONFIRMED", "The photo-derived board outline must be confirmed before generation.")
        validate_ic_for_design(device, spec.battery.series_cells, spec.port_topology)
        return select_mosfets(spec.limits)

    def generate_preview(self, spec: DesignSpec, device: DevicePackage, project_dir: Path) -> dict:
        selection = self.preflight(spec, device)
        output = project_dir / "output"
        preview = output / "preview"
        reports = output / "reports"
        reports.mkdir(parents=True, exist_ok=True)
        artifacts = write_mechanical_previews(spec, preview)
        (output / "design-input.json").write_text(spec.model_dump_json(indent=2), encoding="utf-8")
        (reports / "ic-source.json").write_text(json.dumps(device.as_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
        (reports / "mos-selection.json").write_text(json.dumps(selection.as_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
        ocp = assess_overcurrent_target(spec, device, selection)
        (reports / "overcurrent-assessment.json").write_text(json.dumps(ocp, ensure_ascii=False, indent=2), encoding="utf-8")
        warnings = [] if device.status == "validated" else ["候选样板，不得直接批量生产", "ERC/DRC cannot prove electrical behavior"]
        if ocp.get("status") == "mismatch":
            warnings.append("过流动作目标与 IC/MOS 最坏情况计算不匹配")
        risk = {
            "classification": "candidate_sample" if device.status != "validated" else "validated_template",
            "mass_production_allowed": device.status == "validated" and bool(device.template_dir),
            "warnings": warnings,
        }
        (reports / "risk-report.json").write_text(json.dumps(risk, ensure_ascii=False, indent=2), encoding="utf-8")
        manifest = self._manifest(output, "preview")
        return {
            "stage": "preview_ready",
            "port_topology": spec.port_topology,
            "device": device.as_dict(),
            "mos_selection": selection.as_dict(),
            "overcurrent_assessment": ocp,
            "risk": risk,
            "artifacts": [str(path.relative_to(project_dir)).replace("\\", "/") for path in artifacts],
            "manifest": manifest,
        }

    def generate_manufacturing(self, spec: DesignSpec, device: DevicePackage, project_dir: Path, approved: bool) -> dict:
        if device.status != "validated" and not approved:
            raise DesignError("CANDIDATE_APPROVAL_REQUIRED", "Candidate templates require explicit sample-build approval.")
        selection = self.preflight(spec, device)
        output = project_dir / "output"
        build = self.pipeline.build(spec, device, output)
        (output / "reports" / "mos-selection.json").write_text(json.dumps(selection.as_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
        manifest = self._manifest(output, "manufacturing")
        package = project_dir / f"{_safe_name(spec.name)}-candidate-package.zip"
        self._zip(output, package)
        return {
            "stage": "manufacturing_ready",
            "classification": "candidate_sample" if device.status != "validated" else "validated_template",
            "package": str(package.relative_to(project_dir)).replace("\\", "/"),
            "build": build,
            "manifest": manifest,
        }

    @staticmethod
    def _manifest(output: Path, stage: str) -> dict:
        files = []
        for path in sorted(p for p in output.rglob("*") if p.is_file() and p.name != "manifest.json"):
            files.append({
                "path": str(path.relative_to(output)).replace("\\", "/"),
                "size": path.stat().st_size,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            })
        manifest = {
            "stage": stage,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "generator_version": "0.1.0",
            "files": files,
        }
        (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        return manifest

    @staticmethod
    def _zip(output: Path, destination: Path) -> None:
        with zipfile.ZipFile(destination, "w", zipfile.ZIP_DEFLATED) as archive:
            for path in sorted(p for p in output.rglob("*") if p.is_file()):
                archive.write(path, path.relative_to(output))


def _safe_name(name: str) -> str:
    value = "".join(character if character.isalnum() or character in "-_" else "_" for character in name)
    return value.strip("_") or "battery-board"
