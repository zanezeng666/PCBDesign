from __future__ import annotations

import hashlib
import json
import zipfile
from datetime import datetime, timezone
from pathlib import Path

from .catalog import DevicePackage, get_reference_mosfet_mpn, validate_ic_for_design
from .errors import DesignError
from .kicad import KicadPipeline
from .models import DesignSpec, ElectricalLimits
from .mos import MosfetSelection, derive_config_from_count, derive_electrical_limits, get_mosfet
from .ocp import evaluate_oc_protection
from .preview import CalibrationInfo, write_mechanical_previews


class DesignGenerator:
    def __init__(self, pipeline: KicadPipeline):
        self.pipeline = pipeline

    # ── preflight (no user-provided limits) ─────────────────────

    def preflight(self, spec: DesignSpec, device: DevicePackage) -> tuple[MosfetSelection, ElectricalLimits, dict]:
        """Validate inputs and auto-derive electrical characteristics.

        Derivation chain:
          battery_type → cell voltages
          mos_count + MOSFET spec → continuous / peak / OC trip
        """
        if not spec.outline.confirmed:
            raise DesignError("OUTLINE_NOT_CONFIRMED", "The photo-derived board outline must be confirmed before generation.")
        validate_ic_for_design(device, spec.battery.series_cells, spec.port_topology)

        # Derive: which MOSFET — 优先用用户指定型号，否则用 IC 数据手册推荐
        mosfet_mpn = spec.mos_mpn or get_reference_mosfet_mpn(device)
        mosfet = get_mosfet(mosfet_mpn)

        # Derive: electrical characteristics from mos_count
        selection = derive_config_from_count(spec.mos_count, mosfet)
        limits = derive_electrical_limits(
            selection,
            dischg_oc_detection_v=device.parameters.get("discharge_overcurrent_detection_v_typ", 0.25),
        )

        # Derive: OC protection report
        ocp = evaluate_oc_protection(
            selection,
            dischg_oc_detection_v_typ=device.parameters.get("discharge_overcurrent_detection_v_typ"),
            dischg_oc_detection_v_min_25c=device.parameters.get("discharge_overcurrent_detection_v_min_25c"),
            dischg_oc_detection_v_max_25c=device.parameters.get("discharge_overcurrent_detection_v_max_25c"),
        )

        return selection, limits, ocp

    # ── stages ──────────────────────────────────────────────────

    def generate_preview(self, spec: DesignSpec, device: DevicePackage, project_dir: Path) -> dict:
        selection, limits, ocp = self.preflight(spec, device)
        output = project_dir / "output"
        preview = output / "preview"
        reports = output / "reports"
        reports.mkdir(parents=True, exist_ok=True)

        calibrations = self._load_calibrations(spec, project_dir)
        artifacts = write_mechanical_previews(spec, preview, calibrations)
        design_input = spec.model_dump()
        design_input["derived_limits"] = limits.model_dump()
        (output / "design-input.json").write_text(json.dumps(design_input, ensure_ascii=False, indent=2), encoding="utf-8")
        (reports / "ic-source.json").write_text(json.dumps(device.as_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
        (reports / "mos-selection.json").write_text(json.dumps(selection.as_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
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
            "derived_limits": limits.model_dump(),
            "overcurrent_assessment": ocp,
            "risk": risk,
            "artifacts": [str(path.relative_to(project_dir)).replace("\\", "/") for path in artifacts],
            "manifest": manifest,
        }

    def generate_manufacturing(self, spec: DesignSpec, device: DevicePackage, project_dir: Path, approved: bool) -> dict:
        if device.status != "validated" and not approved:
            raise DesignError("CANDIDATE_APPROVAL_REQUIRED", "Candidate templates require explicit sample-build approval.")
        selection, limits, ocp = self.preflight(spec, device)
        output = project_dir / "output"
        build = self.pipeline.build(spec, device, output)
        (output / "reports" / "mos-selection.json").write_text(json.dumps(selection.as_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
        (output / "reports" / "overcurrent-assessment.json").write_text(json.dumps(ocp, ensure_ascii=False, indent=2), encoding="utf-8")
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

    # ── internal helpers ────────────────────────────────────────

    @staticmethod
    def _load_calibrations(
        spec: DesignSpec, project_dir: Path,
    ) -> dict[str, CalibrationInfo]:
        """Load calibration data from project photos directory."""
        calibrations: dict[str, CalibrationInfo] = {}
        photo_dir = project_dir / "photos"
        for side in ("front", "back"):
            side_dir = photo_dir / side
            if not side_dir.exists():
                continue
            cal_json = side_dir / "calibration.json"
            transparent = side_dir / "transparent.png"
            ppm = 0.0
            if cal_json.exists():
                try:
                    data = json.loads(cal_json.read_text(encoding="utf-8"))
                    ppm = float(data.get("pixels_per_mm", 0.0))
                except Exception:
                    pass
            calibrations[side] = CalibrationInfo(
                pixels_per_mm=ppm,
                transparent_png_path=transparent if transparent.exists() else None,
            )
        return calibrations

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
