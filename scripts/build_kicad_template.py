"""1S 锂电池保护板 KiCad 模板通用生成器（结合本地 KiCad）。

用 KiCad 自带 python 运行（含 pcbnew）：
  E:\\KiCad\\bin\\python.exe scripts/build_kicad_template.py [IC型号]

示例：
  E:\\KiCad\\bin\\python.exe scripts/build_kicad_template.py DW01-G
  E:\\KiCad\\bin\\python.exe scripts/build_kicad_template.py HY2113-MB1B

从 data/ic_catalog/*.yaml 读取 IC 引脚/封装信息，自动生成对应模板到
data/ic_templates/<IC型号>/ 目录。

支持任何 1S SOT-23-6 保护 IC（引脚编号拓扑一致）：
  Pin1=OD, Pin2=CS, Pin3=OC, Pin4=NC/TD(no-connect), Pin5=VDD/VCC, Pin6=VSS/GND

设计网表（1S 共口，含 TH/ID 端子，自洽、无悬空、无冲突）：
  B+(=P+) : J1.1, J1.3, R1.1
  VDD     : R1.2, U1.5, C1.1, C2.1
  B-      : U1.6(VSS), Q1.1(S1), C1.2, C2.2, R3.2, R4.2, C3.2
  MID     : U1.2(CS), Q1.5, Q1.6, Q1.7, Q1.8, R2.1   (电流检测中点)
  P-      : Q1.3(S2), J1.2, J1.4
  OD      : U1.1, Q1.2(G1)
  OC      : U1.3, Q1.4(G2)
  TH      : R2.2, R3.1, C3.1          (温度检测/NTC)
  ID      : R4.1                        (识别)
  NC/TD   : U1.4  -> no_connect
"""
from __future__ import annotations

import json
import sys
import uuid
from pathlib import Path

try:
    import yaml
except ImportError:
    yaml = None  # KiCad python 可能没有 PyYAML，用备用解析器


def uid() -> str:
    return str(uuid.uuid4())


# ── IC 配置加载（从 catalog YAML）────────────────────
ROOT = Path(__file__).resolve().parents[1]


def _mini_yaml_load(text: str) -> dict:
    """极简 YAML 解析（仅支持 catalog 文件的平坦 key: value + 一层嵌套 dict）。"""
    import re
    result: dict = {}
    current_key = None
    for line in text.splitlines():
        if not line.strip() or line.strip().startswith("#"):
            continue
        indent = len(line) - len(line.lstrip())
        stripped = line.strip()
        if indent == 0 and ":" in stripped:
            key, _, val = stripped.partition(":")
            key = key.strip()
            val = val.strip()
            if val == "" or val == "{}":
                current_key = key
                result[key] = {}
            elif val.startswith("[") and val.endswith("]"):
                items = [v.strip().strip('"').strip("'") for v in val[1:-1].split(",") if v.strip()]
                result[key] = items
                current_key = None
            elif val in ("true", "True"):
                result[key] = True; current_key = None
            elif val in ("false", "False"):
                result[key] = False; current_key = None
            elif val in ("null", "~", '""'):
                result[key] = None; current_key = None
            else:
                try:
                    result[key] = int(val)
                except ValueError:
                    try:
                        result[key] = float(val)
                    except ValueError:
                        result[key] = val.strip('"').strip("'")
                current_key = None
        elif indent > 0 and current_key and ":" in stripped:
            k, _, v = stripped.partition(":")
            result[current_key][k.strip().strip('"').strip("'")] = v.strip().strip('"').strip("'")
    return result


def load_ic_config(model: str) -> dict:
    """从 data/ic_catalog/*.yaml 中查找匹配的 IC，返回配置 dict。"""
    import re
    catalog_dir = ROOT / "data" / "ic_catalog"
    norm = re.sub(r"[^A-Z0-9]", "", model.upper())
    for path in sorted(catalog_dir.glob("*.yaml")):
        text = path.read_text(encoding="utf-8")
        data = yaml.safe_load(text) if yaml else _mini_yaml_load(text)
        mpn_norm = re.sub(r"[^A-Z0-9]", "", str(data.get("full_mpn", "")).upper())
        aliases = data.get("aliases", [])
        if isinstance(aliases, str):
            aliases = [aliases]
        aliases_norm = [re.sub(r"[^A-Z0-9]", "", str(a).upper()) for a in aliases]
        if norm == mpn_norm or norm in aliases_norm:
            return data
    raise SystemExit(f"[error] IC '{model}' not found in {catalog_dir}")


# 当前构建的 IC 配置（由 __main__ 设置）
IC_CONFIG: dict = {}
IC_MODEL: str = "DW01-G"  # 默认值，由命令行参数覆盖


# ── 网表定义 ──────────────────────────────────────────────
# 每个网络：名称 -> [(ref, pin_number), ...]
# 注意：J1 拆分为 4 个独立焊盘（B+/B-/P+/P-），不是单个连接器
NETS: dict[str, list[tuple[str, str]]] = {
    "B+": [("J_B+", "1"), ("J_P+", "1"), ("R1", "1")],
    "VDD": [("R1", "2"), ("U1", "5"), ("C1", "1"), ("C2", "1")],
    "B-": [("U1", "6"), ("Q1", "1"), ("C1", "2"), ("C2", "2"), ("R3", "2"), ("R4", "2"), ("C3", "2")],
    "MID": [("U1", "2"), ("Q1", "5"), ("Q1", "6"), ("Q1", "7"), ("Q1", "8"), ("R2", "1")],
    "P-": [("Q1", "3"), ("J_B-", "1"), ("J_P-", "1")],
    "OD": [("U1", "1"), ("Q1", "2")],
    "OC": [("U1", "3"), ("Q1", "4")],
    "TH": [("R2", "2"), ("R3", "1"), ("C3", "1"), ("TP_TH", "1")],
    "ID": [("R4", "1"), ("TP_ID", "1")],
}
# 反查: (ref, pin) -> net
PIN_NET: dict[tuple[str, str], str] = {}
for _net, _conns in NETS.items():
    for _ref, _pin in _conns:
        PIN_NET[(_ref, _pin)] = _net

NO_CONNECT_PINS = [("U1", "4")]  # TD


# ── 符号库（全 passive 引脚）──────────────────────────────
def _pin(name: str, number: str, x: float, y: float, angle: int) -> str:
    return (
        f'        (pin passive line (at {x} {y} {angle}) (length 2.54)\n'
        f'          (name "{name}" (effects (font (size 1.27 1.27))))\n'
        f'          (number "{number}" (effects (font (size 1.27 1.27)))))'
    )


def symbol_library() -> str:
    L: list[str] = []
    L.append("(kicad_symbol_lib")
    L.append("  (version 20231120)")
    L.append('  (generator "bms-template")')

    # 保护 IC（引脚名从 catalog YAML 读取）
    pins = IC_CONFIG.get("pins", {"1": "OD", "2": "CS", "3": "OC", "4": "NC", "5": "VDD", "6": "VSS"})
    L.append(f'  (symbol "{IC_MODEL}"')
    L.append("    (pin_names (offset 1.016)) (in_bom yes) (on_board yes)")
    L.append(f'    (property "Reference" "U" (at 0 6.35 0) (effects (font (size 1.27 1.27))))')
    L.append(f'    (property "Value" "{IC_MODEL}" (at 0 -6.35 0) (effects (font (size 1.27 1.27))))')
    L.append('    (property "Footprint" "Package_TO_SOT_SMD:SOT-23-6" (at 0 -8.89 0) (effects (font (size 1.27 1.27)) hide))')
    L.append('    (property "Datasheet" "" (at 0 0 0) (effects (font (size 1.27 1.27)) hide))')
    L.append(f'    (symbol "{IC_MODEL}_0_1"')
    L.append("      (rectangle (start -5.08 5.08) (end 5.08 -5.08)")
    L.append("        (stroke (width 0.254) (type default)) (fill (type background)))")
    L.append(_pin(pins.get("1", "OD"), "1", -7.62, 2.54, 0))
    L.append(_pin(pins.get("2", "CS"), "2", -7.62, 0, 0))
    L.append(_pin(pins.get("3", "OC"), "3", -7.62, -2.54, 0))
    L.append(_pin(pins.get("4", "NC"), "4", 7.62, -2.54, 180))
    L.append(_pin(pins.get("5", "VDD"), "5", 7.62, 0, 180))
    L.append(_pin(pins.get("6", "VSS"), "6", 7.62, 2.54, 180))
    L.append("    )")
    L.append("  )")

    # FS8205A
    L.append('  (symbol "FS8205A"')
    L.append("    (pin_names (offset 1.016)) (in_bom yes) (on_board yes)")
    L.append('    (property "Reference" "Q" (at 0 6.35 0) (effects (font (size 1.27 1.27))))')
    L.append('    (property "Value" "FS8205A" (at 0 -6.35 0) (effects (font (size 1.27 1.27))))')
    L.append('    (property "Footprint" "Package_SO:TSSOP-8_4.4x3mm_P0.65mm" (at 0 -8.89 0) (effects (font (size 1.27 1.27)) hide))')
    L.append('    (property "Datasheet" "" (at 0 0 0) (effects (font (size 1.27 1.27)) hide))')
    L.append('    (symbol "FS8205A_0_1"')
    L.append("      (rectangle (start -5.08 5.08) (end 5.08 -5.08)")
    L.append("        (stroke (width 0.254) (type default)) (fill (type background)))")
    L.append(_pin("S1", "1", -7.62, 3.81, 0))
    L.append(_pin("G1", "2", 7.62, 3.81, 180))
    L.append(_pin("S2", "3", -7.62, 1.27, 0))
    L.append(_pin("G2", "4", 7.62, 1.27, 180))
    L.append(_pin("D2", "5", -7.62, -1.27, 0))
    L.append(_pin("D2", "6", -7.62, -3.81, 0))
    L.append(_pin("D1", "7", 7.62, -1.27, 180))
    L.append(_pin("D1", "8", 7.62, -3.81, 180))
    L.append("    )")
    L.append("  )")

    # 电阻 R
    L.append('  (symbol "R"')
    L.append("    (pin_names (offset 0)) (in_bom yes) (on_board yes)")
    L.append('    (property "Reference" "R" (at 2.032 0 90) (effects (font (size 1.27 1.27))))')
    L.append('    (property "Value" "R" (at -2.032 0 90) (effects (font (size 1.27 1.27))))')
    L.append('    (property "Footprint" "" (at 0 0 0) (effects (font (size 1.27 1.27)) hide))')
    L.append('    (property "Datasheet" "~" (at 0 0 0) (effects (font (size 1.27 1.27)) hide))')
    L.append('    (symbol "R_0_1"')
    L.append("      (rectangle (start -1.016 2.54) (end 1.016 -2.54)")
    L.append("        (stroke (width 0.254) (type default)) (fill (type none)))")
    L.append(_pin("1", "1", 0, 5.08, 270))
    L.append(_pin("2", "2", 0, -5.08, 90))
    L.append("    )")
    L.append("  )")

    # 电容 C
    L.append('  (symbol "C"')
    L.append("    (pin_names (offset 0)) (in_bom yes) (on_board yes)")
    L.append('    (property "Reference" "C" (at 2.032 0 90) (effects (font (size 1.27 1.27))))')
    L.append('    (property "Value" "C" (at -2.032 0 90) (effects (font (size 1.27 1.27))))')
    L.append('    (property "Footprint" "" (at 0 0 0) (effects (font (size 1.27 1.27)) hide))')
    L.append('    (property "Datasheet" "~" (at 0 0 0) (effects (font (size 1.27 1.27)) hide))')
    L.append('    (symbol "C_0_1"')
    L.append("      (polyline (pts (xy -2.032 0.762) (xy 2.032 0.762))")
    L.append("        (stroke (width 0.508) (type default)) (fill (type none)))")
    L.append("      (polyline (pts (xy -2.032 -0.762) (xy 2.032 -0.762))")
    L.append("        (stroke (width 0.508) (type default)) (fill (type none)))")
    L.append(_pin("1", "1", 0, 5.08, 270))
    L.append(_pin("2", "2", 0, -5.08, 90))
    L.append("    )")
    L.append("  )")

    # 单引脚端子符号（用于 B+/B-/P+/P- 独立焊盘）
    L.append('  (symbol "Conn_01x01"')
    L.append("    (pin_names (offset 1.016) hide) (in_bom yes) (on_board yes)")
    L.append('    (property "Reference" "J" (at 0 3.81 0) (effects (font (size 1.27 1.27))))')
    L.append('    (property "Value" "Conn_01x01" (at 0 -3.81 0) (effects (font (size 1.27 1.27))))')
    L.append('    (property "Footprint" "" (at 0 0 0) (effects (font (size 1.27 1.27)) hide))')
    L.append('    (property "Datasheet" "~" (at 0 0 0) (effects (font (size 1.27 1.27)) hide))')
    L.append('    (symbol "Conn_01x01_1_1"')
    L.append("      (circle (center 0 0) (radius 1.27)")
    L.append("        (stroke (width 0.254) (type default)) (fill (type background)))")
    L.append(_pin("1", "1", -3.81, 0, 0))
    L.append("    )")
    L.append("  )")

    # 测试点符号（用于 TH/ID 端子的物理焊盘表示）
    L.append('  (symbol "TestPoint"')
    L.append("    (pin_names (offset 1.016)) (in_bom yes) (on_board yes)")
    L.append('    (property "Reference" "TP" (at 0 3.81 0) (effects (font (size 1.27 1.27))))')
    L.append('    (property "Value" "TestPoint" (at 0 -3.81 0) (effects (font (size 1.27 1.27))))')
    L.append('    (property "Footprint" "" (at 0 0 0) (effects (font (size 1.27 1.27)) hide))')
    L.append('    (property "Datasheet" "~" (at 0 0 0) (effects (font (size 1.27 1.27)) hide))')
    L.append('    (symbol "TestPoint_0_1"')
    L.append("      (circle (center 0 0) (radius 1.27)")
    L.append("        (stroke (width 0.254) (type default)) (fill (type none)))")
    L.append(_pin("1", "1", 0, 2.54, 270))
    L.append("    )")
    L.append("  )")

    L.append(")")
    return "\n".join(L)


# ── 元件布局与引脚几何 ──────────────────────────────
# 引脚几何: pin_number -> (px, py, angle)【符号局部坐标，y 向上】
PIN_GEOM: dict[str, dict[str, tuple[float, float, int]]] = {
    "J_B+": {"1": (-3.81, 0, 0)},
    "J_P+": {"1": (-3.81, 0, 0)},
    "J_B-": {"1": (-3.81, 0, 0)},
    "J_P-": {"1": (-3.81, 0, 0)},
    "U1": {"1": (-7.62, 2.54, 0), "2": (-7.62, 0, 0), "3": (-7.62, -2.54, 0),
           "4": (7.62, -2.54, 180), "5": (7.62, 0, 180), "6": (7.62, 2.54, 180)},
    "Q1": {"1": (-7.62, 3.81, 0), "2": (7.62, 3.81, 180), "3": (-7.62, 1.27, 0),
           "4": (7.62, 1.27, 180), "5": (-7.62, -1.27, 0), "6": (-7.62, -3.81, 0),
           "7": (7.62, -1.27, 180), "8": (7.62, -3.81, 180)},
    "R1": {"1": (0, 5.08, 270), "2": (0, -5.08, 90)},
    "R2": {"1": (0, 5.08, 270), "2": (0, -5.08, 90)},
    "R3": {"1": (0, 5.08, 270), "2": (0, -5.08, 90)},
    "R4": {"1": (0, 5.08, 270), "2": (0, -5.08, 90)},
    "C1": {"1": (0, 5.08, 270), "2": (0, -5.08, 90)},
    "C2": {"1": (0, 5.08, 270), "2": (0, -5.08, 90)},
    "C3": {"1": (0, 5.08, 270), "2": (0, -5.08, 90)},
    "TP_TH": {"1": (0, 2.54, 270)},
    "TP_ID": {"1": (0, 2.54, 270)},
}

# 元件实例：ref -> (lib_symbol, value, footprint, sx, sy)
def _components() -> dict[str, tuple[str, str, str, float, float]]:
    return {
        "J_B+": ("Conn_01x01", "B+", "Connector_PinHeader_2.54mm:PinHeader_1x01_P2.54mm_Vertical", 25.4, 50.0),
        "J_P+": ("Conn_01x01", "P+", "Connector_PinHeader_2.54mm:PinHeader_1x01_P2.54mm_Vertical", 25.4, 55.0),
        "J_B-": ("Conn_01x01", "B-", "Connector_PinHeader_2.54mm:PinHeader_1x01_P2.54mm_Vertical", 25.4, 65.0),
        "J_P-": ("Conn_01x01", "P-", "Connector_PinHeader_2.54mm:PinHeader_1x01_P2.54mm_Vertical", 25.4, 70.0),
        "U1": (IC_MODEL, IC_MODEL, "Package_TO_SOT_SMD:SOT-23-6", 76.2, 60.96),
        "Q1": ("FS8205A", "FS8205A", "Package_SO:TSSOP-8_4.4x3mm_P0.65mm", 76.2, 111.76),
        "R1": ("R", "100", "Resistor_SMD:R_0603_1608Metric", 127.0, 35.56),
        "R2": ("R", "2K", "Resistor_SMD:R_0603_1608Metric", 101.6, 60.96),
        "R3": ("R", "10K", "Resistor_SMD:R_0603_1608Metric", 25.4, 111.76),
        "R4": ("R", "10K", "Resistor_SMD:R_0603_1608Metric", 25.4, 137.16),
        "C1": ("C", "0.1uF", "Capacitor_SMD:C_0603_1608Metric", 127.0, 60.96),
        "C2": ("C", "10uF", "Capacitor_SMD:C_0603_1608Metric", 127.0, 86.36),
        "C3": ("C", "0.01uF", "Capacitor_SMD:C_0603_1608Metric", 25.4, 86.36),
        "TP_TH": ("TestPoint", "TH", "TestPoint:TestPoint_Pad_D1.5mm", 10.0, 111.76),
        "TP_ID": ("TestPoint", "ID", "TestPoint:TestPoint_Pad_D1.5mm", 10.0, 137.16),
    }

COMPONENTS: dict[str, tuple[str, str, str, float, float]] = {}

# 画导线版专用布局（紧凑信号流向优化）：
#   布局策略：
#     - 4 个端子焊盘分散在左侧（B+/P+ 在上，B-/P- 在下）
#     - U1/Q1 紧密排列在中间
#     - 被动元件按功能分组，紧凑排列在右侧
#     - 最小化长距离走线
WIRED_POSITIONS: dict[str, tuple[float, float]] = {
    "J_B+": (50.0, 50.0),    # B+ 焊盘（左上）
    "J_P+": (50.0, 60.0),    # P+ 焊盘（B+ 下方）
    "J_B-": (50.0, 100.0),   # B- 焊盘（左下）
    "J_P-": (50.0, 110.0),   # P- 焊盘（B- 下方）
    "U1": (100.0, 75.0),     # 保护 IC（中间偏上）
    "Q1": (100.0, 100.0),    # MOSFET（U1 正下方，引脚对齐）
    "R1": (127.0, 35.0),     # B+ 限流电阻（右上）
    "R2": (120.0, 50.0),     # TH 分压电阻（U1/Q1 之间右侧）
    "C1": (170.0, 60.0),     # VDD 滤波电容（右侧）
    "C2": (170.0, 95.0),     # VDD 大电容（C1 下方）
    "C3": (30.0, 85.0),      # B- 滤波电容（左侧）
    "R3": (30.0, 115.0),     # B- 下拉电阻（C3 下方）
    "R4": (30.0, 135.0),     # ID 上拉电阻（R3 下方）
    "TP_TH": (15.0, 115.0),  # TH 测试点（R3 左侧）
    "TP_ID": (15.0, 135.0),  # ID 测试点（R4 左侧）
}

# 引脚角度 ->  outward 单位向量（原理图坐标，y 向下）
_OUTWARD = {0: (-1, 0), 180: (1, 0), 270: (0, -1), 90: (0, 1)}


def _pin_endpoint(ref: str, pin: str, positions: dict[str, tuple[float, float]] | None = None) -> tuple[float, float, int]:
    """返回引脚连接端点的原理图坐标 (x, y, angle)。positions 用于覆盖默认布局。"""
    if positions and ref in positions:
        sx, sy = positions[ref]
    else:
        _, _, _, sx, sy = COMPONENTS[ref]
    px, py, angle = PIN_GEOM[ref][pin]
    return (sx + px, sy - py, angle)


def _place_symbol(ref: str, positions: dict[str, tuple[float, float]] | None = None) -> str:
    sym, value, fp, _, _ = COMPONENTS[ref]
    if positions and ref in positions:
        sx, sy = positions[ref]
    else:
        _, _, _, sx, sy = COMPONENTS[ref]
    lib_id = f"battery_protection:{sym}"
    L = []
    L.append(f'  (symbol (lib_id "{lib_id}") (at {sx} {sy} 0) (unit 1)')
    L.append("    (exclude_from_sim no) (in_bom yes) (on_board yes) (dnp no)")
    L.append(f"    (uuid {uid()})")
    L.append(f'    (property "Reference" "{ref}" (at {sx} {sy - 8.89} 0)')
    L.append("      (effects (font (size 1.27 1.27))))")
    L.append(f'    (property "Value" "{value}" (at {sx} {sy + 8.89} 0)')
    L.append("      (effects (font (size 1.27 1.27))))")
    L.append(f'    (property "Footprint" "{fp}" (at 0 0 0)')
    L.append("      (effects (font (size 1.27 1.27)) hide))")
    L.append(f'    (property "Datasheet" "" (at 0 0 0)')
    L.append("      (effects (font (size 1.27 1.27)) hide))")
    L.append("    (instances")
    L.append(f'      (project "template" (path "/{uid()}" (reference "{ref}") (unit 1))))')
    for pin in PIN_GEOM[ref]:
        L.append(f'    (pin "{pin}" (uuid {uid()}))')
    L.append("  )")
    return "\n".join(L)


def _net_labels() -> str:
    """为每个引脚生成“短线 stub + 网络标签”（或 no_connect 标记）。"""
    L: list[str] = []
    for ref in COMPONENTS:
        for pin in PIN_GEOM[ref]:
            ex, ey, angle = _pin_endpoint(ref, pin)
            ox, oy = _OUTWARD[angle]
            sx2, sy2 = ex + ox * 2.54, ey + oy * 2.54
            if (ref, pin) in NO_CONNECT_PINS:
                L.append(f"  (no_connect (at {ex} {ey}) (uuid {uid()}))")
                continue
            net = PIN_NET.get((ref, pin))
            if not net:
                continue
            # 短线从引脚端点向外
            L.append(f"  (wire (pts (xy {ex} {ey}) (xy {sx2} {sy2}))")
            L.append("    (stroke (width 0) (type default))")
            L.append(f"    (uuid {uid()}))")
            L.append(f'  (label "{net}" (at {sx2} {sy2} 0) (fields_autoplaced yes)')
            L.append("    (effects (font (size 1.27 1.27)) (justify left))")
            L.append(f"    (uuid {uid()}))")
    return "\n".join(L)


def _net_wires(positions: dict[str, tuple[float, float]] | None = None) -> str:
    """为每个网络生成可见导线连接（所有走线严格连接引脚端点，不穿过元件本体）。

    布局策略：
    - U1/Q1 垂直对齐在中间（x=100），引脚面对面
    - B+ 水平总线在顶部 (y=25)
    - VDD/B- 垂直总线在右侧 (x=185)，C1/C2 水平连线不穿越本体
    - P- 水平总线在底部 (y=125)
    - OD/OC 用折线绕过 U1/Q1 本体
    - MID 用左侧垂直总线 (x=85) 连接 U1.CS 和 Q1.D 引脚
    positions 用于覆盖默认元件布局。
    """
    import math
    L: list[str] = []
    segments: list[tuple[float, float, float, float, str]] = []

    def add_seg(x1, y1, x2, y2, net):
        if abs(x1 - x2) < 0.001 and abs(y1 - y2) < 0.001:
            return
        segments.append((x1, y1, x2, y2, net))

    def emit_wire(x1, y1, x2, y2):
        L.append(f"  (wire (pts (xy {x1} {y1}) (xy {x2} {y2}))")
        L.append("    (stroke (width 0) (type default))")
        L.append(f"    (uuid {uid()}))")

    def emit_junction(x, y):
        L.append(f"  (junction (at {x} {y}) (diameter 0) (color 0 0 0 0)")
        L.append(f"    (uuid {uid()}))")

    def emit_hop(cx, cy, horizontal=True):
        r = 0.76
        pts = []
        for i in range(5):
            angle = math.pi * i / 4
            if horizontal:
                px = cx - r * math.cos(angle)
                py = cy - r * math.sin(angle)
            else:
                px = cx + r * math.sin(angle)
                py = cy - r * math.cos(angle)
            pts.append((round(px, 4), round(py, 4)))
        L.append(f"  (polyline (pts")
        for px, py in pts:
            L.append(f"    (xy {px} {py})")
        L.append("    )")
        L.append("    (stroke (width 0) (type default))")
        L.append(f"    (uuid {uid()}))")

    def pin_ep(ref, pin):
        """获取引脚端点坐标"""
        ex, ey, angle = _pin_endpoint(ref, pin, positions)
        ox, oy = _OUTWARD[angle]
        stub_x, stub_y = ex + ox * 2.54, ey + oy * 2.54
        return ex, ey, stub_x, stub_y, angle

    # no_connect 标记
    for (ref, pin) in NO_CONNECT_PINS:
        ex, ey, _ = _pin_endpoint(ref, pin, positions)
        L.append(f"  (no_connect (at {ex} {ey}) (uuid {uid()}))")

    # ── 为每个网络生成显式布线路径 ──
    # 所有走线严格连接引脚端点，不穿过任何元件本体。

    # === B+ 网络：J_B+.1, J_P+.1, R1.1 ===
    bplus_bus_y = 25.0  # B+ 水平总线（所有元件上方）

    for ref, pin in [("J_B+", "1"), ("J_P+", "1")]:
        ex, ey, sx, sy, angle = pin_ep(ref, pin)
        add_seg(ex, ey, sx, sy, "B+")       # stub 从引脚端点
        add_seg(sx, sy, sx, bplus_bus_y, "B+")  # 垂直到 B+ 总线
        emit_junction(sx, bplus_bus_y)

    # B+ 水平总线
    r1_1_ex, r1_1_ey, r1_1_sx, r1_1_sy, _ = pin_ep("R1", "1")
    j_pins_x = [pin_ep(ref, p)[2] for ref, p in [("J_B+", "1"), ("J_P+", "1")]]
    add_seg(min(j_pins_x), bplus_bus_y, r1_1_sx, bplus_bus_y, "B+")
    # R1.1 垂直 stub 从总线到引脚
    add_seg(r1_1_sx, bplus_bus_y, r1_1_sx, r1_1_sy, "B+")
    emit_junction(r1_1_sx, bplus_bus_y)

    # === VDD 网络：R1.2, U1.5, C1.1, C2.1 ===
    vdd_bus_x = 185.0  # VDD 垂直总线（C1/C2 右侧，不穿过元件）

    # R1.2 → 向右 → VDD 总线
    r1_2_ex, r1_2_ey, r1_2_sx, r1_2_sy, _ = pin_ep("R1", "2")
    add_seg(r1_2_ex, r1_2_ey, r1_2_sx, r1_2_sy, "VDD")
    add_seg(r1_2_sx, r1_2_sy, vdd_bus_x, r1_2_sy, "VDD")
    emit_junction(vdd_bus_x, r1_2_sy)

    # U1.5 → 向右 → VDD 总线
    u1_5_ex, u1_5_ey, u1_5_sx, u1_5_sy, _ = pin_ep("U1", "5")
    add_seg(u1_5_ex, u1_5_ey, u1_5_sx, u1_5_sy, "VDD")
    add_seg(u1_5_sx, u1_5_sy, vdd_bus_x, u1_5_sy, "VDD")
    emit_junction(vdd_bus_x, u1_5_sy)

    # C1.1/C2.1 → 向右 → VDD 总线
    for ref, pin in [("C1", "1"), ("C2", "1")]:
        ex, ey, sx, sy, angle = pin_ep(ref, pin)
        add_seg(ex, ey, sx, sy, "VDD")
        add_seg(sx, sy, vdd_bus_x, sy, "VDD")
        emit_junction(vdd_bus_x, sy)

    # VDD 垂直总线
    vdd_ys = [r1_2_sy, u1_5_sy]
    for ref, pin in [("C1", "1"), ("C2", "1")]:
        _, _, _, sy, _ = pin_ep(ref, pin)
        vdd_ys.append(sy)
    add_seg(vdd_bus_x, min(vdd_ys), vdd_bus_x, max(vdd_ys), "VDD")

    # === B- 网络：U1.6, Q1.1, C1.2, C2.2, R3.2, R4.2, C3.2 ===
    bminus_bus_x = 185.0  # B- 垂直总线（C1/C2 右侧）

    # U1.6 → 向右 → B- 总线
    u1_6_ex, u1_6_ey, u1_6_sx, u1_6_sy, _ = pin_ep("U1", "6")
    add_seg(u1_6_ex, u1_6_ey, u1_6_sx, u1_6_sy, "B-")
    add_seg(u1_6_sx, u1_6_sy, bminus_bus_x, u1_6_sy, "B-")
    emit_junction(bminus_bus_x, u1_6_sy)

    # Q1.1 → 向右 → B- 总线
    q1_1_ex, q1_1_ey, q1_1_sx, q1_1_sy, _ = pin_ep("Q1", "1")
    add_seg(q1_1_ex, q1_1_ey, q1_1_sx, q1_1_sy, "B-")
    add_seg(q1_1_sx, q1_1_sy, bminus_bus_x, q1_1_sy, "B-")
    emit_junction(bminus_bus_x, q1_1_sy)

    # C1.2/C2.2 → 向下/上 → B- 总线
    for ref, pin in [("C1", "2"), ("C2", "2")]:
        ex, ey, sx, sy, angle = pin_ep(ref, pin)
        add_seg(ex, ey, sx, sy, "B-")
        add_seg(sx, sy, bminus_bus_x, sy, "B-")
        emit_junction(bminus_bus_x, sy)

    # C3.2 → 向右 → B- 总线
    c3_2_ex, c3_2_ey, c3_2_sx, c3_2_sy, _ = pin_ep("C3", "2")
    add_seg(c3_2_ex, c3_2_ey, c3_2_sx, c3_2_sy, "B-")
    add_seg(c3_2_sx, c3_2_sy, bminus_bus_x, c3_2_sy, "B-")
    emit_junction(bminus_bus_x, c3_2_sy)

    # R3.2/R4.2 → 向右 → B- 总线
    for ref, pin in [("R3", "2"), ("R4", "2")]:
        ex, ey, sx, sy, angle = pin_ep(ref, pin)
        add_seg(ex, ey, sx, sy, "B-")
        add_seg(sx, sy, bminus_bus_x, sy, "B-")
        emit_junction(bminus_bus_x, sy)

    # B- 垂直总线
    bminus_ys = [u1_6_sy, q1_1_sy, c3_2_sy]
    for ref, pin in [("C1", "2"), ("C2", "2"), ("R3", "2"), ("R4", "2")]:
        _, _, _, sy, _ = pin_ep(ref, pin)
        bminus_ys.append(sy)
    add_seg(bminus_bus_x, min(bminus_ys), bminus_bus_x, max(bminus_ys), "B-")

    # === MID 网络: U1.2, Q1.5-8, R2.1 ===
    # 使用左侧垂直总线 x=85（在 U1/Q1 本体左侧），避免穿越本体
    mid_bus_x = 85.0

    # U1.2(CS) → 向左 → 向下 → 向右到 R2.1（绕过 MID 总线上方）
    u1_2_ex, u1_2_ey, u1_2_sx, u1_2_sy, _ = pin_ep("U1", "2")
    r2_1_ex, r2_1_ey, r2_1_sx, r2_1_sy, _ = pin_ep("R2", "1")
    add_seg(u1_2_ex, u1_2_ey, u1_2_sx, u1_2_sy, "MID")
    # 向左到 MID 总线左侧
    add_seg(u1_2_sx, u1_2_sy, 90.0, u1_2_sy, "MID")
    # 向上到 R2 高度上方
    add_seg(90.0, u1_2_sy, 90.0, 48.0, "MID")
    # 向右到 R2.1
    add_seg(90.0, 48.0, r2_1_sx, 48.0, "MID")
    add_seg(r2_1_sx, 48.0, r2_1_sx, r2_1_sy, "MID")
    add_seg(r2_1_ex, r2_1_ey, r2_1_sx, r2_1_sy, "MID")
    emit_junction(90.0, 48.0)

    # MID 垂直总线 x=85: 从 y=48 到 Q1 最低引脚
    q1_pins_y = [pin_ep("Q1", p)[3] for p in ["5", "6", "7", "8"]]
    mid_bus_y_top = 48.0
    mid_bus_y_bot = max(q1_pins_y)
    add_seg(mid_bus_x, mid_bus_y_top, mid_bus_x, mid_bus_y_bot, "MID")

    # Q1.5/Q1.6（左侧引脚）→ 向左到 MID 总线
    for pin in ["5", "6"]:
        ex, ey, sx, sy, angle = pin_ep("Q1", pin)
        add_seg(ex, ey, sx, sy, "MID")
        add_seg(sx, sy, mid_bus_x, sy, "MID")
        emit_junction(mid_bus_x, sy)

    # Q1.7/Q1.8（右侧引脚）→ 向下 → 向左 → 到 MID 总线
    r2_2_ex, r2_2_ey, r2_2_sx, r2_2_sy, _ = pin_ep("R2", "2")
    route_y = max(q1_pins_y) + 2.0  # Q1 本体下方
    for pin in ["7", "8"]:
        ex, ey, sx, sy, angle = pin_ep("Q1", pin)
        add_seg(ex, ey, sx, sy, "MID")
        add_seg(sx, sy, sx, route_y, "MID")
        add_seg(sx, route_y, mid_bus_x, route_y, "MID")
        emit_junction(mid_bus_x, route_y)

    # MID 总线与 R2.1 的连接点
    emit_junction(mid_bus_x, 48.0)

    # === OD 网络: U1.1 → Q1.2 (折线绕过本体) ===
    u1_od = pin_ep("U1", "1")
    q1_g1 = pin_ep("Q1", "2")
    add_seg(u1_od[0], u1_od[1], u1_od[2], u1_od[3], "OD")
    # 从 U1.OD stub 向上到 U1 上方
    od_route_y = u1_od[3] - 5.0  # U1 本体上方
    add_seg(u1_od[2], u1_od[3], u1_od[2], od_route_y, "OD")
    # 水平到 Q1.G2 x 坐标
    add_seg(u1_od[2], od_route_y, q1_g1[2], od_route_y, "OD")
    # 向下到 Q1.G2
    add_seg(q1_g1[2], od_route_y, q1_g1[2], q1_g1[3], "OD")
    add_seg(q1_g1[0], q1_g1[1], q1_g1[2], q1_g1[3], "OD")

    # === OC 网络: U1.3 → Q1.4 (折线绕过本体) ===
    u1_oc = pin_ep("U1", "3")
    q1_g2 = pin_ep("Q1", "4")
    add_seg(u1_oc[0], u1_oc[1], u1_oc[2], u1_oc[3], "OC")
    # 从 U1.OC stub 向左
    oc_left_x = u1_oc[2] - 2.54
    add_seg(u1_oc[2], u1_oc[3], oc_left_x, u1_oc[3], "OC")
    # 向下到 Q1 上方
    oc_route_y = u1_oc[3] + 5.0
    add_seg(oc_left_x, u1_oc[3], oc_left_x, oc_route_y, "OC")
    # 水平到 Q1.G4
    add_seg(oc_left_x, oc_route_y, q1_g2[2], oc_route_y, "OC")
    # 向下到 Q1.G4
    add_seg(q1_g2[2], oc_route_y, q1_g2[2], q1_g2[3], "OC")
    add_seg(q1_g2[0], q1_g2[1], q1_g2[2], q1_g2[3], "OC")

    # === P- 网络：Q1.3, J_B-.1, J_P-.1 ===
    pminus_bus_y = 125.0  # P- 水平总线（J_B-/J_P- 下方）

    # Q1.3 → 向下 → P- 总线
    q1_3_ex, q1_3_ey, q1_3_sx, q1_3_sy, _ = pin_ep("Q1", "3")
    add_seg(q1_3_ex, q1_3_ey, q1_3_sx, q1_3_sy, "P-")
    add_seg(q1_3_sx, q1_3_sy, q1_3_sx, pminus_bus_y, "P-")
    emit_junction(q1_3_sx, pminus_bus_y)

    # J_B-/J_P- → 向下 → P- 总线
    for ref, pin in [("J_B-", "1"), ("J_P-", "1")]:
        ex, ey, sx, sy, angle = pin_ep(ref, pin)
        add_seg(ex, ey, sx, sy, "P-")
        add_seg(sx, sy, sx, pminus_bus_y, "P-")
        emit_junction(sx, pminus_bus_y)

    # P- 水平总线
    j_p_pins_x = [pin_ep(ref, p)[2] for ref, p in [("J_B-", "1"), ("J_P-", "1")]]
    add_seg(min(j_p_pins_x), pminus_bus_y, q1_3_sx, pminus_bus_y, "P-")

    # === TH 网络: R2.2, R3.1, C3.1, TP_TH.1 ===
    th_bus_y = 135.0

    # R2.2 → 向下 → TH 总线
    r2_2_ex2, r2_2_ey2, r2_2_sx2, r2_2_sy2, _ = pin_ep("R2", "2")
    add_seg(r2_2_sx2, r2_2_sy2, r2_2_sx2, th_bus_y, "TH")
    emit_junction(r2_2_sx2, th_bus_y)

    # R3.1 → TH 总线
    r3_1_ex, r3_1_ey, r3_1_sx, r3_1_sy, _ = pin_ep("R3", "1")
    add_seg(r3_1_ex, r3_1_ey, r3_1_sx, r3_1_sy, "TH")
    add_seg(r3_1_sx, r3_1_sy, r3_1_sx, th_bus_y, "TH")
    emit_junction(r3_1_sx, th_bus_y)

    # C3.1 → TH 总线
    c3_1_ex, c3_1_ey, c3_1_sx, c3_1_sy, _ = pin_ep("C3", "1")
    add_seg(c3_1_ex, c3_1_ey, c3_1_sx, c3_1_sy, "TH")
    add_seg(c3_1_sx, c3_1_sy, c3_1_sx, th_bus_y, "TH")
    emit_junction(c3_1_sx, th_bus_y)

    # TP_TH.1 → TH 总线
    tp_th_ex, tp_th_ey, tp_th_sx, tp_th_sy, _ = pin_ep("TP_TH", "1")
    add_seg(tp_th_ex, tp_th_ey, tp_th_sx, tp_th_sy, "TH")
    add_seg(tp_th_sx, tp_th_sy, tp_th_sx, th_bus_y, "TH")
    emit_junction(tp_th_sx, th_bus_y)

    # TH 水平总线
    th_xs = [r2_2_sx2, r3_1_sx, c3_1_sx, tp_th_sx]
    add_seg(min(th_xs), th_bus_y, max(th_xs), th_bus_y, "TH")

    # === ID 网络: R4.1, TP_ID.1 ===
    id_bus_y = 140.0

    # R4.1 → ID 总线
    r4_1_ex, r4_1_ey, r4_1_sx, r4_1_sy, _ = pin_ep("R4", "1")
    add_seg(r4_1_ex, r4_1_ey, r4_1_sx, r4_1_sy, "ID")
    add_seg(r4_1_sx, r4_1_sy, r4_1_sx, id_bus_y, "ID")
    emit_junction(r4_1_sx, id_bus_y)

    # TP_ID.1 → ID 总线
    tp_id_ex, tp_id_ey, tp_id_sx, tp_id_sy, _ = pin_ep("TP_ID", "1")
    add_seg(tp_id_ex, tp_id_ey, tp_id_sx, tp_id_sy, "ID")
    add_seg(tp_id_sx, tp_id_sy, tp_id_sx, id_bus_y, "ID")
    emit_junction(tp_id_sx, id_bus_y)

    # ID 水平总线
    id_xs = [r4_1_sx, tp_id_sx]
    add_seg(min(id_xs), id_bus_y, max(id_xs), id_bus_y, "ID")

    # ── 输出所有导线 ──
    for (x1, y1, x2, y2, net) in segments:
        emit_wire(x1, y1, x2, y2)

    # ── 交叉检测 + 跳线弧 ──
    crossings = []
    for i, (ax1, ay1, ax2, ay2, net_a) in enumerate(segments):
        a_horiz = abs(ay1 - ay2) < 0.001
        for j, (bx1, by1, bx2, by2, net_b) in enumerate(segments):
            if j <= i or net_a == net_b:
                continue
            b_horiz = abs(by1 - by2) < 0.001
            if a_horiz == b_horiz:
                continue
            if a_horiz:
                hx1, hx2 = min(ax1, ax2), max(ax1, ax2)
                vy1, vy2 = min(by1, by2), max(by1, by2)
                cx, cy = bx1, ay1
                if hx1 < cx < hx2 and vy1 < cy < vy2:
                    crossings.append((cx, cy, True))
            else:
                hx1, hx2 = min(bx1, bx2), max(bx1, bx2)
                vy1, vy2 = min(ay1, ay2), max(ay1, ay2)
                cx, cy = ax1, by1
                if hx1 < cx < hx2 and vy1 < cy < vy2:
                    crossings.append((cx, cy, False))
    seen = set()
    for cx, cy, h in crossings:
        key = (round(cx, 2), round(cy, 2))
        if key not in seen:
            seen.add(key)
            emit_hop(cx, cy, horizontal=h)

    return "\n".join(L)


def schematic() -> str:
    L: list[str] = []
    L.append("(kicad_sch")
    L.append("  (version 20250114)")
    L.append('  (generator "bms-template")')
    L.append('  (generator_version "10.0")')
    L.append(f"  (uuid {uid()})")
    L.append('  (paper "A4")')
    L.append("  (title_block")
    L.append(f'    (title "{IC_MODEL} 1S Battery Protection Board")')
    L.append('    (comment 1 "Common-port, FS8205A companion")')
    L.append("  )")
    L.append("  (lib_symbols")
    # 嵌入符号定义：外层 symbol 需加库名前缀以匹配 lib_id
    # （battery_protection:XXX），内层单元符号（XXX_0_1）保持原名。
    import re
    for line in symbol_library().splitlines()[3:-1]:
        m = re.match(r'^  \(symbol "([^"]+)"\s*$', line)
        if m:
            line = f'  (symbol "battery_protection:{m.group(1)}"'
        L.append("  " + line)
    L.append("  )")
    for ref in COMPONENTS:
        L.append(_place_symbol(ref))
    L.append(_net_labels())
    L.append(")")
    return "\n".join(L)


def schematic_wired() -> str:
    """生成画导线版原理图（元件间用可见导线连接，不用网络标签）。"""
    L: list[str] = []
    L.append("(kicad_sch")
    L.append("  (version 20250114)")
    L.append('  (generator "bms-template")')
    L.append('  (generator_version "10.0")')
    L.append(f"  (uuid {uid()})")
    L.append('  (paper "A4")')
    L.append("  (title_block")
    L.append(f'    (title "{IC_MODEL} 1S Battery Protection (Wired)")')
    L.append('    (comment 1 "Common-port, explicit wire connections")')
    L.append("  )")
    L.append("  (lib_symbols")
    import re
    for line in symbol_library().splitlines()[3:-1]:
        m = re.match(r'^  \(symbol "([^"]+)"\s*$', line)
        if m:
            line = f'  (symbol "battery_protection:{m.group(1)}"'
        L.append("  " + line)
    L.append("  )")
    for ref in COMPONENTS:
        L.append(_place_symbol(ref, WIRED_POSITIONS))
    L.append(_net_wires(WIRED_POSITIONS))
    L.append(")")
    return "\n".join(L)


# ── PCB 生成（需 KiCad 自带 python 的 pcbnew）────────────────────
# 元件布局（mm）与封装
FP_LAYOUT: dict[str, tuple[str, str, float, float, float]] = {
    # ref -> (pretty 库，封装名，x_mm, y_mm, 旋转 deg)
    # 4 个独立焊盘（B+/P+ 在上，B-/P- 在下）
    "J_B+": ("Connector_PinHeader_2.54mm.pretty", "PinHeader_1x01_P2.54mm_Vertical", 4.0, 5.0, 0),
    "J_P+": ("Connector_PinHeader_2.54mm.pretty", "PinHeader_1x01_P2.54mm_Vertical", 4.0, 7.5, 0),
    "J_B-": ("Connector_PinHeader_2.54mm.pretty", "PinHeader_1x01_P2.54mm_Vertical", 4.0, 12.5, 0),
    "J_P-": ("Connector_PinHeader_2.54mm.pretty", "PinHeader_1x01_P2.54mm_Vertical", 4.0, 15.0, 0),
    "U1": ("Package_TO_SOT_SMD.pretty", "SOT-23-6", 16.0, 6.5, 0),
    "Q1": ("Package_SO.pretty", "TSSOP-8_4.4x3mm_P0.65mm", 16.0, 14.0, 0),
    "R1": ("Resistor_SMD.pretty", "R_0603_1608Metric", 28.0, 5.0, 0),
    "R2": ("Resistor_SMD.pretty", "R_0603_1608Metric", 22.0, 10.0, 0),
    "R3": ("Resistor_SMD.pretty", "R_0603_1608Metric", 4.0, 17.0, 0),
    "R4": ("Resistor_SMD.pretty", "R_0603_1608Metric", 8.0, 17.0, 0),
    "C1": ("Capacitor_SMD.pretty", "C_0603_1608Metric", 28.0, 10.0, 0),
    "C2": ("Capacitor_SMD.pretty", "C_0603_1608Metric", 28.0, 15.0, 0),
    "C3": ("Capacitor_SMD.pretty", "C_0603_1608Metric", 12.0, 17.0, 0),
}


def build_pcb(out_dir: Path, kicad_fp_dir: Path, width_mm: float = 34.0, height_mm: float = 20.0) -> Path:
    import pcbnew

    board = pcbnew.BOARD()

    def mm(v: float) -> int:
        return pcbnew.FromMM(v)

    # 板框 Edge.Cuts（矩形；adapter 会按 spec 轮廓重塑）
    def edge_line(x1, y1, x2, y2):
        seg = pcbnew.PCB_SHAPE(board)
        seg.SetShape(pcbnew.SHAPE_T_SEGMENT)
        seg.SetStart(pcbnew.VECTOR2I(mm(x1), mm(y1)))
        seg.SetEnd(pcbnew.VECTOR2I(mm(x2), mm(y2)))
        seg.SetLayer(pcbnew.Edge_Cuts)
        seg.SetWidth(mm(0.1))
        board.Add(seg)

    edge_line(0, 0, width_mm, 0)
    edge_line(width_mm, 0, width_mm, height_mm)
    edge_line(width_mm, height_mm, 0, height_mm)
    edge_line(0, height_mm, 0, 0)

    # 网络
    net_map: dict[str, object] = {}
    for name in NETS:
        ni = pcbnew.NETINFO_ITEM(board, name)
        board.Add(ni)
        net_map[name] = ni

    # 放置封装
    fp_objs: dict[str, object] = {}
    for ref, (lib, fpn, x, y, rot) in FP_LAYOUT.items():
        libpath = str(kicad_fp_dir / lib)
        fp = pcbnew.FootprintLoad(libpath, fpn)
        if fp is None:
            raise RuntimeError(f"无法加载封装 {fpn} ← {libpath}")
        board.Add(fp)
        fp.SetPosition(pcbnew.VECTOR2I(mm(x), mm(y)))
        fp.SetOrientation(pcbnew.EDA_ANGLE(rot, pcbnew.DEGREES_T))
        fp.SetReference(ref)
        fp.SetValue(fpn.split(":")[-1])
        fp_objs[ref] = fp

    # 焊盘分配网络（跳过没有封装的元件，如 TP_TH/TP_ID 测试点符号）
    for (ref, pin), netname in PIN_NET.items():
        if ref not in fp_objs:
            continue
        fp = fp_objs[ref]
        pad = fp.FindPadByNumber(pin)
        if pad is not None:
            pad.SetNetCode(net_map[netname].GetNetCode())

    def pad_pos(ref, pin):
        if ref not in fp_objs:
            return None
        pad = fp_objs[ref].FindPadByNumber(pin)
        if pad is None:
            return None
        p = pad.GetPosition()
        return (pcbnew.ToMM(p.x), pcbnew.ToMM(p.y))

    fcu = pcbnew.F_Cu
    bcu = pcbnew.B_Cu

    def seg(net, ax, ay, bx, by, layer, width):
        t = pcbnew.PCB_TRACK(board)
        t.SetStart(pcbnew.VECTOR2I(mm(ax), mm(ay)))
        t.SetEnd(pcbnew.VECTOR2I(mm(bx), mm(by)))
        t.SetWidth(mm(width))
        t.SetLayer(layer)
        t.SetNetCode(net_map[net].GetNetCode())
        board.Add(t)

    def route_path(net, pts, layer, width):
        """沿途经点列表逐段直线走线。"""
        for i in range(len(pts) - 1):
            seg(net, pts[i][0], pts[i][1], pts[i + 1][0], pts[i + 1][1], layer, width)

    def route_chain(net, conns, layer, width=0.25):
        """将同一网络的焊盘按顺序两两 L 型相连（跳过无封装元件）。"""
        pts = [pad_pos(r, p) for (r, p) in conns if pad_pos(r, p) is not None]
        if len(pts) < 2:
            return
        route_path(net, pts, layer, width)

    def is_smd(ref, pin):
        if ref not in fp_objs:
            return False
        pad = fp_objs[ref].FindPadByNumber(pin)
        if pad is None:
            return False
        return pad.GetAttribute() == pcbnew.PAD_ATTRIB_SMD

    def add_via(net, x, y):
        via = pcbnew.PCB_VIA(board)
        via.SetPosition(pcbnew.VECTOR2I(mm(x), mm(y)))
        via.SetWidth(mm(0.8))
        via.SetDrill(mm(0.4))
        via.SetViaType(pcbnew.VIATYPE_THROUGH)
        board.Add(via)
        via.SetNetCode(net_map[net].GetNetCode())

    # 信号网络走在 B.Cu（背面）；SMD 焊盘用过孔引到背面，PTH 焊盘直通背面。
    # 这样 F.Cu 无信号走线，B- 顶铜皮 zone 不会被割裂，能连通所有 B- 焊盘。
    for net, conns in NETS.items():
        if net == "B-":
            continue
        width = 0.3 if net in ("B+", "VDD", "P-") else 0.2
        for (ref, pin) in conns:
            if is_smd(ref, pin):
                x, y = pad_pos(ref, pin)
                add_via(net, x, y)  # 过孔打在 SMD 焊盘中心（via-in-pad，仅警告）
        route_chain(net, conns, bcu, width)

    # Q1.1(B-) 被 Q1.2 过孔等异物包围，zone 填充不会生成连接辐条；
    # 加一段顶面 B- 走线短桩连到左侧 zone 铜皮，确保连通。
    q11 = pad_pos("Q1", "1")
    seg("B-", q11[0], q11[1], 11.0, q11[1], fcu, 0.3)

    # B- 顶铜皮 zone（F.Cu 无信号走线割裂，zone 完整覆盖并连通所有 B- 焊盘）
    zone = pcbnew.ZONE(board)
    zone.SetLayer(fcu)
    zone.SetMinThickness(mm(0.2))
    # 热风间隙/辐条调小（默认 0.5mm 比窄焊盘还大），确保 zone 熔接到所有 B- 焊盘
    zone.SetThermalReliefGap(mm(0.2))
    zone.SetThermalReliefSpokeWidth(mm(0.25))
    chain = pcbnew.SHAPE_LINE_CHAIN()
    chain.Append(mm(0), mm(0))
    chain.Append(mm(width_mm), mm(0))
    chain.Append(mm(width_mm), mm(height_mm))
    chain.Append(mm(0), mm(height_mm))
    chain.SetClosed(True)
    zone.AddPolygon(chain)
    board.Add(zone)
    # 注意：SetNet 必须在 board.Add(zone) 之后调用，否则网络码会被重置为 0
    zone.SetNet(net_map["B-"])

    # 填充 zone 并建立连通性
    filler = pcbnew.ZONE_FILLER(board)
    filler.Fill([zone])
    board.BuildConnectivity()

    pcb_path = out_dir / "pcb.kicad_pcb"
    board.Save(str(pcb_path))
    return pcb_path


def sym_lib_table() -> str:
    """生成 sym-lib-table（元素间需空格，否则 KiCad 10 报 UNINITIALIZED）。"""
    return (
        "(sym_lib_table\n"
        "  (version 7)\n"
        '  (lib (name "battery_protection") (type "KiCad") '
        '(uri "${KIPRJMOD}/battery_protection.kicad_sym") (options "") (descr ""))\n'
        ")\n"
    )


if __name__ == "__main__":
    import os

    # 命令行参数：IC 型号（默认 DW01-G）
    IC_MODEL = sys.argv[1] if len(sys.argv) > 1 else "DW01-G"
    IC_CONFIG = load_ic_config(IC_MODEL)
    COMPONENTS.update(_components())
    print(f"[info] building template for: {IC_MODEL} ({IC_CONFIG.get('full_mpn', '')})")
    print(f"[info] pins: {IC_CONFIG.get('pins', {})}")

    out = ROOT / "data" / "ic_templates" / IC_MODEL
    out.mkdir(parents=True, exist_ok=True)
    (out / "battery_protection.kicad_sym").write_text(symbol_library(), encoding="utf-8")
    (out / "schematic.kicad_sch").write_text(schematic(), encoding="utf-8")
    (out / "schematic_wired.kicad_sch").write_text(schematic_wired(), encoding="utf-8")
    (out / "sym-lib-table").write_text(sym_lib_table(), encoding="utf-8")
    print("[ok] symbol library + schematic (labels) + schematic_wired (wires) + sym-lib-table")

    # PCB 需 pcbnew（仅 KiCad 自带 python 有）
    try:
        fp_dir = Path(os.environ.get("KICAD_FOOTPRINT_DIR", r"E:\KiCad\share\kicad\footprints"))
        pcb = build_pcb(out, fp_dir)
        print(f"[ok] pcb -> {pcb}")
    except Exception as exc:  # noqa: BLE001
        print(f"[skip] pcb generation: {exc}")

    # template.json manifest
    manifest = {
        "device": IC_MODEL,
        "description": f"1S 锂电池保护板（{IC_MODEL} + FS8205A 共口），候选模板",
        "schematic": "schematic.kicad_sch",
        "pcb": "pcb.kicad_pcb",
        "adapter": "adapt.py",
        "adapter_interpreter": "kicad-python",
        "status": "candidate",
    }
    (out / "template.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("[ok] template.json")

    # adapt.py 几何适配器（通用，复制自 DW01-G 模板）
    # ★ 始终覆盖以确保最新版本
    adapt_src = ROOT / "data" / "ic_templates" / "DW01-G" / "adapt.py"
    if adapt_src.exists() and str(adapt_src) != str(out / "adapt.py"):
        import shutil
        shutil.copy2(adapt_src, out / "adapt.py")
        print("[ok] adapt.py (updated from DW01-G)")
    elif not (out / "adapt.py").exists():
        print("[warn] adapt.py source not found")
    else:
        print("[ok] adapt.py (already up-to-date)")

    # adapt_common.py 公共模块（端子 side 分配、zone 层选择等）
    # ★ 始终覆盖以确保最新版本（含 compute_pad_rotation / create_custom_pad 等）
    common_src = ROOT / "data" / "ic_templates" / "adapt_common.py"
    if common_src.exists() and str(common_src) != str(out / "adapt_common.py"):
        import shutil
        shutil.copy2(common_src, out / "adapt_common.py")
        print("[ok] adapt_common.py (updated)")
    elif not (out / "adapt_common.py").exists():
        print("[warn] adapt_common.py source not found")
    else:
        print("[ok] adapt_common.py (already up-to-date)")

    print(f"\n[done] template output: {out}")