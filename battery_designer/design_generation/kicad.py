from __future__ import annotations

import csv
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

from ..component_detection.catalog import DevicePackage
from ..core.errors import DesignError
from ..core.models import DesignSpec

# ── KiCad path resolution (cross-platform) ──
def _resolve_kicad_bin() -> Path:
    """Find KiCad bin directory from common platform-specific locations."""
    candidates: list[Path] = []
    if sys.platform == "win32":
        # 标准安装位置 + 常见的盘符根目录安装（如 E:\KiCad）
        for ver in ("9.0", "8.0", "7.0"):
            candidates.append(Path(f"C:\\Program Files\\KiCad\\{ver}\\bin"))
            candidates.append(Path(f"C:\\Program Files (x86)\\KiCad\\{ver}\\bin"))
        for drive in ("E", "D", "C"):
            candidates.append(Path(f"{drive}:\\KiCad\\bin"))
    elif sys.platform == "darwin":
        candidates.append(Path("/Applications/KiCad/KiCad.app/Contents/Applications/bin"))
    else:
        candidates.extend([Path("/usr/bin"), Path("/usr/local/bin")])

    for c in candidates:
        kicad_cli = c / ("kicad-cli.exe" if sys.platform == "win32" else "kicad-cli")
        if kicad_cli.exists():
            return c
    return Path(".")  # fallback — user will get a clear error on first CLI call


_kicad_bin_env = os.getenv("KICAD_BIN", "").strip()
_KICAD_BIN = Path(_kicad_bin_env) if _kicad_bin_env else _resolve_kicad_bin()
if not _KICAD_BIN.exists():
    _KICAD_BIN = _resolve_kicad_bin()

# 注意：Path("") 会被规范化为 Path(".")（当前目录，恒存在），不能用真值判断
# 环境变量是否设置，必须显式判空串。
_kicad_cli_env = os.getenv("KICAD_CLI", "").strip()
if _kicad_cli_env and Path(_kicad_cli_env).exists():
    _KICAD_CLI_DEFAULT = Path(_kicad_cli_env)
else:
    _KICAD_CLI_DEFAULT = _KICAD_BIN / ("kicad-cli.exe" if sys.platform == "win32" else "kicad-cli")

# KiCad 自带 Python（含 pcbnew），用于运行模板的 .py 几何适配器
_KICAD_PYTHON = _KICAD_BIN / ("python.exe" if sys.platform == "win32" else "python3")

if sys.platform == "win32" and _KICAD_BIN.exists():
    os.environ["PATH"] = str(_KICAD_BIN) + os.pathsep + os.environ.get("PATH", "")
    try:
        os.add_dll_directory(str(_KICAD_BIN))
    except (AttributeError, OSError):
        pass


try:
    import cairosvg  # noqa: F401
except OSError:
    cairosvg = None  # type: ignore[assignment]


class KicadPipeline:
    def __init__(self, kicad_cli: Path | None = None):
        configured = os.getenv("KICAD_CLI")
        self.kicad_cli = Path(configured) if configured else kicad_cli or _KICAD_CLI_DEFAULT

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
        if not template.is_absolute():
            # 相对路径相对项目根目录解析（kicad.py 位于 battery_designer/ 下）
            template = Path(__file__).resolve().parents[1] / template
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

        # 用户指定了 MOS 型号时，更新原理图中 Q1 的 Value 显示
        if spec.mos_mpn:
            self._patch_mos_value(schematic, spec.mos_mpn)
            # 同步更新画导线版（若存在）
            wired_sch = design_dir / "schematic_wired.kicad_sch"
            if wired_sch.exists():
                self._patch_mos_value(wired_sch, spec.mos_mpn)

        # 元器件识别结果：将识别到的被动元件值 patch 到原理图
        passive_patches: dict[str, dict] = {}
        if spec.detected_components:
            passive_patches = self._patch_passive_values(schematic, spec.detected_components)
            wired_sch = design_dir / "schematic_wired.kicad_sch"
            if wired_sch.exists():
                self._patch_passive_values(wired_sch, spec.detected_components)

        reports = output_dir / "reports"
        manufacturing = output_dir / "manufacturing"
        gerber = manufacturing / "gerber"
        reports.mkdir(parents=True, exist_ok=True)
        gerber.mkdir(parents=True, exist_ok=True)

        # 生成元器件来源报告（记录哪些值来自 VLM，哪些保留模板默认）
        if spec.detected_components or passive_patches:
            source_report = {
                "passive_patches": passive_patches,
                "detected_count": len(spec.detected_components),
                "mos_mpn": spec.mos_mpn,
                "mos_count": spec.mos_count,
            }
            # 标注未被 patch 的默认元件
            all_refs = ["R1", "R2", "R3", "R4", "C1", "C2", "C3"]
            source_report["template_defaults_kept"] = [
                r for r in all_refs if r not in passive_patches
            ]
            (reports / "component-source.json").write_text(
                json.dumps(source_report, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

        # 模板几何/布线适配（硬门控前必须执行）。adapter 由审核过的器件包提供；
        # .py 适配器用 KiCad 自带 python（含 pcbnew）运行，无通用不安全回退。
        adapter = design_dir / manifest.get("adapter", "")
        if not manifest.get("adapter") or not adapter.exists():
            raise DesignError("TEMPLATE_ADAPTER_MISSING", "The reviewed template has no geometry/routing adapter.")
        spec_path = design_dir / "design-input.json"
        spec_path.write_text(spec.model_dump_json(indent=2), encoding="utf-8")
        adapt_cmd = [str(_KICAD_PYTHON), str(adapter), str(spec_path), str(pcb)] if adapter.suffix == ".py" else [str(adapter), str(spec_path), str(pcb)]
        adapt_result = subprocess.run(adapt_cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
        # 始终保存适配器输出到日志（便于调试 Freerouting 等问题）
        adapt_log = reports / "adapt.log"
        adapt_log.write_text(
            f"=== returncode: {adapt_result.returncode} ===\n"
            f"=== stdout ===\n{adapt_result.stdout[-8000:]}\n\n=== stderr ===\n{adapt_result.stderr[-8000:]}",
            encoding="utf-8", errors="replace",
        )
        if adapt_result.returncode != 0:
            raise DesignError("TEMPLATE_ADAPT_FAILED", "External design command failed.",
                              {"command": adapt_cmd, "stdout": adapt_result.stdout[-2000:], "stderr": adapt_result.stderr[-2000:]})

        erc = reports / "erc.rpt"
        drc = reports / "drc.rpt"
        # ERC/DRC 带 --exit-code-violations 时连警告也返回非零，不能用 _run（会对非零抛错）。
        # 候选模板定位：仅 ERC error 级违规、DRC 未连接项作为硬失败，其余违规仅作警告。
        self._run_check([str(self.kicad_cli), "sch", "erc", "--exit-code-violations", "-o", str(erc), str(schematic)], erc, "ERC_FAILED")
        self._assert_no_erc_errors(erc)
        self._run_check([str(self.kicad_cli), "pcb", "drc", "--exit-code-violations", "--all-track-errors", "-o", str(drc), str(pcb)], drc, "DRC_FAILED")
        self._assert_no_unconnected(drc)

        preview = output_dir / "preview"
        preview.mkdir(exist_ok=True)
        # 原理图 SVG：排除图纸边框，导出干净原理图
        self._run([str(self.kicad_cli), "sch", "export", "svg", "--exclude-drawing-sheet", "-o", str(preview), str(schematic)], "SCHEMATIC_SVG_FAILED")
        # 画导线版原理图（若模板包含）
        wired_sch = design_dir / "schematic_wired.kicad_sch"
        if wired_sch.exists():
            self._run([str(self.kicad_cli), "sch", "export", "svg", "--exclude-drawing-sheet", "-o", str(preview), str(wired_sch)], "SCHEMATIC_WIRED_SVG_FAILED")
        # PCB 各面 SVG：page-size-mode=2 仅导出板框区域（不含页面框架），mode-single 输出单个文件
        for side, layers in (("front", "F.Cu,F.Mask,F.Silkscreen,Edge.Cuts"), ("back", "B.Cu,B.Mask,B.Silkscreen,Edge.Cuts")):
            self._run([str(self.kicad_cli), "pcb", "export", "svg", "--mode-single", "--page-size-mode", "2", "--layers", layers, "-o", str(preview / f"pcb_{side}.svg"), str(pcb)], "PCB_SVG_FAILED")
        # PNG 光栅化：PCB 板小，用更高分辨率；原理图保持 1800px
        for svg in preview.rglob("*.svg"):
            if cairosvg is not None:
                fn = svg.name
                w = 3600 if fn.startswith("pcb_") else 1800
                cairosvg.svg2png(url=str(svg), write_to=str(svg.with_suffix(".png")), output_width=w)
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
        content = report.read_text(encoding="utf-8", errors="replace")
        import re
        m = re.search(r"Found\s+(\d+)\s+unconnected\s+pads", content)
        if m and int(m.group(1)) > 0:
            raise DesignError("UNCONNECTED_ITEMS", "KiCad DRC reports unconnected items.",
                              {"report": str(report), "unconnected_pads": int(m.group(1))})

    @staticmethod
    def _patch_mos_value(schematic: Path, mos_mpn: str) -> None:
        """Replace the default MOSFET value (FS8205A) in Q1 symbol instance with user-specified MPN."""
        import re
        text = schematic.read_text(encoding="utf-8")
        # 只替换元件实例的 Value 属性（紧跟 Reference "Q" 的那个）
        # KiCad 原理图中元件实例格式: (property "Value" "FS8205A" ...)
        text = re.sub(
            r'(\(property "Value" ")FS8205A(")',
            rf'\g<1>{re.escape(mos_mpn)}\2',
            text,
        )
        schematic.write_text(text, encoding="utf-8")

    @staticmethod
    def _patch_passive_values(
        schematic: Path,
        detected_components: list,
    ) -> dict[str, dict]:
        """将 VLM 识别到的被动元件值 patch 到原理图模板中。

        策略：按类型（电阻/电容）分组，按 confidence 降序排列，
        然后按元件序号（R1→R2→...、C1→C2→...）逐一匹配。
        数量不匹配时仅 patch 能匹配的部分。

        Returns:
            dict，key 为 ref（如 R1），value 为 {old_value, new_value, confidence, source}。
        """
        import re as _re

        # 按类型提取识别值（仅处理有 silkscreen 的元器件）
        resistors: list[dict] = []
        capacitors: list[dict] = []
        for comp in detected_components:
            # DetectedComponent is a Pydantic BaseModel — use getattr, NOT .get()
            ct = getattr(comp, "type", "other")
            ss = getattr(comp, "silkscreen", "")
            conf = getattr(comp, "confidence", 0.5)
            if not ss:
                continue
            if ct == "resistor":
                resistors.append({"silkscreen": ss, "confidence": conf})
            elif ct == "capacitor":
                capacitors.append({"silkscreen": ss, "confidence": conf})

        # 按 confidence 降序
        resistors.sort(key=lambda x: -x["confidence"])
        capacitors.sort(key=lambda x: -x["confidence"])

        # 模板默认值顺序（与 build_kicad_template.py 中 COMPONENTS 顺序一致）
        resistor_refs = ["R1", "R2", "R3", "R4"]
        capacitor_refs = ["C1", "C2", "C3"]

        patches: dict[str, dict] = {}  # ref -> {old_value, new_value, confidence}

        text = schematic.read_text(encoding="utf-8")
        original_text = text

        def _do_patch(ref: str, new_val: str, conf: float) -> None:
            nonlocal text
            # 匹配: (property "Reference" "R1" ...) ... (property "Value" "旧值" ...)
            # 注意：Reference 属性含有嵌套括号 (at ...)、(effects ...)，不能用 [^)]*
            # 使用 [\s\S]*?（非贪婪）跳过嵌套内容直到下一个 Value 属性
            pattern = (
                r'\(property "Reference" "' + _re.escape(ref) + r'"[\s\S]*?'
                r'\(property "Value" "([^"]*)"'
            )
            m = _re.search(pattern, text)
            if m:
                old_val = m.group(1)
                text = _re.sub(
                    pattern,
                    lambda mm: mm.group(0).replace(
                        f'(property "Value" "{old_val}"',
                        f'(property "Value" "{new_val}"',
                    ),
                    text,
                    count=1,
                )
                patches[ref] = {
                    "old_value": old_val,
                    "new_value": new_val,
                    "confidence": conf,
                }

        # 逐一对应 patch 电阻
        for i, det in enumerate(resistors[:len(resistor_refs)]):
            _do_patch(resistor_refs[i], det["silkscreen"], det["confidence"])

        # 逐一对应 patch 电容
        for i, det in enumerate(capacitors[:len(capacitor_refs)]):
            _do_patch(capacitor_refs[i], det["silkscreen"], det["confidence"])

        if text != original_text:
            schematic.write_text(text, encoding="utf-8")

        return patches

    @staticmethod
    def _assert_no_erc_errors(report: Path) -> None:
        content = report.read_text(encoding="utf-8", errors="replace")
        # 每条违规带一行严重级 "    ; error" 或 "    ; warning"；仅 error 级硬失败
        errors = sum(1 for line in content.splitlines() if line.strip() == "; error")
        if errors:
            raise DesignError("ERC_ERRORS", "KiCad ERC reports error-severity violations.", {"report": str(report), "errors": errors})

    @staticmethod
    def _run_check(command: list[str], report: Path, code: str) -> subprocess.CompletedProcess:
        """运行 ERC/DRC：--exit-code-violations 下警告也返回非零，故不据返回码抛错；
        仅当报告文件未生成时才视为命令本身失败。"""
        result = subprocess.run(command, capture_output=True, text=True, encoding="utf-8", errors="replace")
        if not report.exists():
            raise DesignError(code, "External design check failed to produce a report.", {"command": command, "stdout": result.stdout[-2000:], "stderr": result.stderr[-2000:]})
        return result

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
