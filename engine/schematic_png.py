"""原理图 PNG 预览生成 — SKiDL auto_stub 模式"""
import os, sys, yaml, subprocess
from pathlib import Path

KICAD_PATH = r"C:\Program Files\KiCad\9.0"
KICAD_SHARE = os.path.join(KICAD_PATH, "share", "kicad")
KICAD_SYMBOL_DIR = os.path.join(KICAD_SHARE, "symbols")
KICAD_CLI = os.path.join(KICAD_PATH, "bin", "kicad-cli.exe")

os.environ["KICAD_SYMBOL_DIR"] = KICAD_SYMBOL_DIR
os.environ["KICAD9_SYMBOL_DIR"] = KICAD_SYMBOL_DIR
os.environ["KICAD8_SYMBOL_DIR"] = KICAD_SYMBOL_DIR
os.environ["KICAD7_SYMBOL_DIR"] = KICAD_SYMBOL_DIR
os.environ["KICAD6_SYMBOL_DIR"] = KICAD_SYMBOL_DIR

import skidl
from skidl import Net, Part, KICAD9, SchLib

CUSTOM_SYM_LIB = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "circuits", "symbols", "battery_protection.kicad_sym")


def main(yaml_path, output_dir):
    with open(yaml_path, "r", encoding="utf-8") as f:
        circuit = yaml.safe_load(f)

    skidl.set_default_tool(KICAD9)
    custom_lib = SchLib(CUSTOM_SYM_LIB) if os.path.exists(CUSTOM_SYM_LIB) else None

    parts = {}
    for d in circuit["parts"]:
        pid, pn, cat = d["id"], d["part_name"], d["category"]
        if cat in ("IC", "MOSFET_DUAL") and custom_lib:
            p = Part(custom_lib, pn, footprint=d["footprint"])
        elif cat == "RES":
            try: p = Part("Device", "R_Small_US", footprint=d["footprint"])
            except: p = Part("Device", "R", footprint=d["footprint"])
        elif cat == "CAP":
            try: p = Part("Device", "C_Small", footprint=d["footprint"])
            except: p = Part("Device", "C", footprint=d["footprint"])
        elif cat == "CONN":
            try: p = Part("Connector_Generic", "Conn_01x04", footprint=d["footprint"])
            except: p = Part("Connector", "Conn_01x04", footprint=d["footprint"])
        else: continue
        p.ref = pid; p.value = d["value"]
        parts[pid] = p

    for nd in circuit["nets"]:
        net = Net(nd["name"])
        for c in nd["connections"]:
            if c["part"] in parts:
                try: parts[c["part"]][str(c["pin"])] += net
                except: pass

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    name = circuit.get("name", "sch").replace(" ", "_")
    sch_file = out / f"{name}.kicad_sch"
    png_file = out / "schematic.png"

    print(f"生成原理图: {sch_file}")
    # auto_stub 让无法布线的网络自动转成标签
    skidl.generate_schematic(
        filepath=str(sch_file),
        auto_stub=True,
        auto_stub_fallback="warn",   # 失败时警告但继续
    )
    print(f"OK: {sch_file}")

    print(f"导出 PNG...")
    r = subprocess.run(
        [KICAD_CLI, "sch", "export", "png", str(sch_file),
         "--output", str(png_file), "--background", "opaque"],
        capture_output=True, text=True)
    if r.returncode == 0:
        b = os.path.getsize(png_file)
        print(f"OK: {png_file} ({b} bytes)")
    else:
        print(f"FAIL: {r.stderr}")
        if "Cannot open" in r.stderr:
            # 可能 kicad-cli 语法不同
            r2 = subprocess.run(
                [KICAD_CLI, "sch", "export", "svg", str(sch_file),
                 "--output", str(out / "schematic.svg")],
                capture_output=True, text=True)
            print(f"SVG: {r2.returncode} {r2.stderr[:200]}")

    return str(sch_file) if sch_file.exists() else None


if __name__ == "__main__":
    yf = sys.argv[1] if len(sys.argv) > 1 else "engine/circuits/protections/dw01_1s.yaml"
    od = sys.argv[2] if len(sys.argv) > 2 else "output/dw01_sch"
    result = main(yf, od)
    print(f"\n{'DONE' if result else 'FAILED'}: {result}")
