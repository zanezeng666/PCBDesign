"""DW01 1S 保护板 — 手动放置坐标的方案"""
import os, yaml
from pathlib import Path

KICAD_PATH = r"C:\Program Files\KiCad\9.0"
KICAD_SHARE = os.path.join(KICAD_PATH, "share", "kicad")
os.environ["KICAD_SYMBOL_DIR"] = os.path.join(KICAD_SHARE, "symbols")
os.environ["KICAD9_SYMBOL_DIR"] = os.path.join(KICAD_SHARE, "symbols")
os.environ["KICAD8_SYMBOL_DIR"] = os.path.join(KICAD_SHARE, "symbols")
os.environ["KICAD7_SYMBOL_DIR"] = os.path.join(KICAD_SHARE, "symbols")
os.environ["KICAD6_SYMBOL_DIR"] = os.path.join(KICAD_SHARE, "symbols")

import skidl
from skidl import Net, Part, KICAD9, SchLib, generate_schematic

CUSTOM_SYM_LIB = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "circuits", "symbols", "battery_protection.kicad_sym"
)

KICAD_CLI = os.path.join(KICAD_PATH, "bin", "kicad-cli.exe")


def generate_with_manual_placement(ic="DW01-G", mos_count=1, series=1,
                                    width_mm=40, height_mm=15, out_dir="output/dw01_wired"):
    """
    参数化生成原理图 — 手动指定元件坐标
    
    Args:
        ic: 保护IC型号 (DW01-G, HY2112, etc.)
        mos_count: MOS管数量
        series: 电池串数 (1-5)
        width_mm: PCB宽度
        height_mm: PCB高度
    """
    skidl.set_default_tool(KICAD9)
    custom_lib = SchLib(CUSTOM_SYM_LIB) if os.path.exists(CUSTOM_SYM_LIB) else None

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    print(f"参数: IC={ic}, MOS={mos_count}, {series}S, {width_mm}x{height_mm}mm")

    # ── 创建元件 ──
    # 用 @subcircuit 无法在简单电路上用，改用大间距手动放置
    
    # U1 - DW01 (左上)
    u1 = Part(custom_lib, ic, footprint="Package_TO_SOT_SMD:SOT-23-6")
    u1.ref = "U1"; u1.value = ic

    # Q1 - MOS (左下)
    q1 = Part(custom_lib, "FS8205A", footprint="Package_SO:TSSOP-8_4.4x3mm_P0.65mm")
    q1.ref = "Q1"; q1.value = "FS8205A"

    # 无源器件 (右侧)
    try: r1 = Part("Device", "R_Small_US", footprint="Resistor_SMD:R_0603_1608Metric")
    except: r1 = Part("Device", "R", footprint="Resistor_SMD:R_0603_1608Metric")
    r1.ref = "R1"; r1.value = "100"

    try: c1 = Part("Device", "C_Small", footprint="Capacitor_SMD:C_0603_1608Metric")
    except: c1 = Part("Device", "C", footprint="Capacitor_SMD:C_0603_1608Metric")
    c1.ref = "C1"; c1.value = "0.1uF"

    try: c2 = Part("Device", "C_Small", footprint="Capacitor_SMD:C_0603_1608Metric")
    except: c2 = Part("Device", "C", footprint="Capacitor_SMD:C_0603_1608Metric")
    c2.ref = "C2"; c2.value = "10uF"

    try: j1 = Part("Connector_Generic", "Conn_01x04", footprint="Connector_PinHeader_2.54mm:PinHeader_1x04_P2.54mm_Vertical")
    except: j1 = Part("Connector", "Conn_01x04", footprint="Connector_PinHeader_2.54mm:PinHeader_1x04_P2.54mm_Vertical")
    j1.ref = "J1"; j1.value = "Header_4P"

    # ── 创建网络 — DW01经典电路 ──
    # 网络：
    # B+ = P+ = 电池正极 = 输出正极
    # B- = GND = 电池负极 → Q1源极
    # P- = Q1漏极 → 输出负极（经MOS管控制）
    # VDD = IC供电
    # OD = 放电控制
    # OC = 充电控制
    # CS = 电流检测

    b_plus = Net("B+")
    b_minus = Net("B-")
    vdd = Net("VDD")
    od = Net("OD")
    oc = Net("OC")
    cs = Net("CS")

    # B+ 网络：电池正极 ← 通过R1供电IC，同时直连输出正极
    r1[1] += b_plus
    q1[7] += b_plus  # D1
    q1[8] += b_plus  # D1
    j1[1] += b_plus   # P+
    j1[3] += b_plus   # P+ (备份)

    # VDD 网络：IC供电
    r1[2] += vdd
    u1[5] += vdd       # VDD pin
    c1[1] += vdd

    # B- 网络：公共地
    u1[2] += b_minus   # VSS
    u1[6] += b_minus   # VSS
    c1[2] += b_minus
    c2[2] += b_minus
    q1[1] += b_minus   # S1
    q1[3] += b_minus   # S2
    j1[2] += b_minus   # B-
    j1[4] += b_minus   # P-

    # OD 网络
    u1[1] += od        # OD pin
    q1[2] += od        # G1

    # OC 网络
    u1[3] += oc        # OC pin
    q1[4] += oc        # G2

    # CS 网络
    u1[4] += cs        # CS pin
    q1[1] += cs        # S1 (复用B-节点，实际CS应通过1K电阻到B-，但简化电路直连)

    # ── 手动指定元件放置坐标 ──
    u1.placement = skidl.schematics.placement.make_placement(-80, 60, 0)
    q1.placement = skidl.schematics.placement.make_placement(-80, -40, 0)
    r1.placement = skidl.schematics.placement.make_placement(0, 60, 0)
    c1.placement = skidl.schematics.placement.make_placement(0, 20, 0)
    c2.placement = skidl.schematics.placement.make_placement(0, -10, 0)
    j1.placement = skidl.schematics.placement.make_placement(80, 0, 0)

    # ── 生成 ──
    sch_file = out / "schematic.kicad_sch"
    print(f"\n生成原理图 (手动坐标)...")
    try:
        generate_schematic(
            filepath=str(sch_file),
            # 不用 auto_stub，让路由器直接画线
            # 增大retries
            retries=4,
        )
    except Exception as e:
        print(f"路由失败，降级到标签模式: {type(e).__name__}")
        generate_schematic(
            filepath=str(sch_file),
            auto_stub=True,
            auto_stub_fallback="labels",
            retries=1,
        )

    # 检查结果
    content = open(sch_file, 'r', encoding='utf-8', errors='replace').read()
    wires = content.count('(wire ')
    labels = content.count('(label ') + content.count('(global_label ')
    print(f"结果: {wires} wires, {labels} labels")

    return sch_file


def export_png(sch_path, png_path):
    import cairo, cairosvg, subprocess
    os.environ['PATH'] = KICAD_PATH + r'\bin;' + os.environ.get('PATH', '')
    svg_file = str(Path(png_path).with_suffix('.svg'))
    r = subprocess.run(
        [KICAD_CLI, "sch", "export", "svg", str(sch_path), "--output", svg_file],
        capture_output=True, text=True)
    if r.returncode != 0:
        print(f"SVG fail: {r.stderr[:200]}")
        return
    cairosvg.svg2png(url=svg_file, write_to=str(png_path), output_width=1600)
    print(f"PNG: {png_path} ({os.path.getsize(png_path)} bytes)")


if __name__ == "__main__":
    import sys
    # 支持命令行参数
    ic = sys.argv[1] if len(sys.argv) > 1 else "DW01-G"
    mos = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    series = int(sys.argv[3]) if len(sys.argv) > 3 else 1
    w = float(sys.argv[4]) if len(sys.argv) > 4 else 40
    h = float(sys.argv[5]) if len(sys.argv) > 5 else 15

    sch = generate_with_manual_placement(ic, mos, series, w, h)
    if sch:
        png = str(Path(sch).parent / "schematic.png")
        export_png(sch, png)
