"""端到端测试：DW01-G 候选模板 → 真实 KiCad 工程 + Gerber。

用项目 python 运行（内部经 subprocess 调 KiCad python 跑 adapter）：
  .venv\\Scripts\\python.exe scripts\\_e2e_kicad.py
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from battery_designer.catalog import IcCatalog  # noqa: E402
from battery_designer.generator import DesignGenerator  # noqa: E402
from battery_designer.kicad import KicadPipeline  # noqa: E402
from battery_designer.models import (  # noqa: E402
    BatterySpec,
    BoardOutline,
    ConnectionMode,
    DesignSpec,
    Point,
    Polarity,
    Terminal,
    TerminalRole,
)


def build_spec() -> DesignSpec:
    outline = BoardOutline(
        points=[
            Point(x_mm=0, y_mm=0),
            Point(x_mm=40, y_mm=0),
            Point(x_mm=40, y_mm=25),
            Point(x_mm=0, y_mm=25),
        ],
        confirmed=True,
    )
    terminals = [
        Terminal(id="BPOS", position=Point(x_mm=4, y_mm=10), roles={TerminalRole.BATTERY, TerminalRole.CHARGE, TerminalRole.DISCHARGE}, polarity=Polarity.POSITIVE, width_mm=2, height_mm=2),
        Terminal(id="B-", position=Point(x_mm=4, y_mm=13), roles={TerminalRole.BATTERY}, polarity=Polarity.NEGATIVE, width_mm=2, height_mm=2),
        Terminal(id="P-", position=Point(x_mm=4, y_mm=16), roles={TerminalRole.CHARGE, TerminalRole.DISCHARGE}, polarity=Polarity.NEGATIVE, width_mm=2, height_mm=2),
        Terminal(id="T", position=Point(x_mm=4, y_mm=19), roles={TerminalRole.TEMPERATURE}, width_mm=1.5, height_mm=1.5),
    ]
    return DesignSpec(
        name="DW01-G E2E Test Board",
        protection_ic="DW01-G",
        battery=BatterySpec(count=1, connection=ConnectionMode.PARALLEL, battery_type="18650"),
        mos_count=1,
        outline=outline,
        terminals=terminals,
    )


def main() -> int:
    work = ROOT / "work" / "debug" / "e2e_kicad"
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True)

    catalog = IcCatalog(ROOT / "data" / "ic_catalog", work / "ic_cache")
    pipeline = KicadPipeline()
    generator = DesignGenerator(pipeline)

    spec = build_spec()
    device = catalog.resolve(spec.protection_ic)
    print(f"[e2e] resolved device: {device.full_mpn} status={device.status} template_dir={device.template_dir}")

    result = generator.generate_manufacturing(spec, device, work, approved=True)
    print("[e2e] generate_manufacturing OK")
    print(json.dumps({k: result[k] for k in result if k != "manifest"}, ensure_ascii=False, indent=2)[:1500])

    # 验证关键产出
    out = work / "output"
    checks = {
        "kicad_pcb": out / "kicad" / "pcb.kicad_pcb",
        "kicad_sch": out / "kicad" / "schematic.kicad_sch",
        "erc": out / "reports" / "erc.rpt",
        "drc": out / "reports" / "drc.rpt",
        "bom": out / "manufacturing" / "bom.csv",
        "positions": out / "manufacturing" / "positions.csv",
        "bom_jlc": out / "manufacturing" / "bom_jlc.csv",
        "cpl_jlc": out / "manufacturing" / "cpl_jlc.csv",
    }
    gerbers = list((out / "manufacturing" / "gerber").glob("*")) if (out / "manufacturing" / "gerber").exists() else []
    ok = True
    for name, path in checks.items():
        exists = path.exists()
        ok = ok and exists
        print(f"  [{'OK' if exists else 'MISSING'}] {name}: {path}")
    print(f"  [{'OK' if gerbers else 'MISSING'}] gerber files: {len(gerbers)}")
    for g in gerbers:
        print(f"      - {g.name}")
    ok = ok and len(gerbers) > 0

    # DRC 未连接门控确认
    drc_txt = (out / "reports" / "drc.rpt").read_text(encoding="utf-8", errors="replace")
    unconnected = "unconnected_items" in drc_txt.lower()
    print(f"  [{'FAIL' if unconnected else 'OK'}] DRC 无未连接")
    ok = ok and not unconnected

    print("\n[e2e] RESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
