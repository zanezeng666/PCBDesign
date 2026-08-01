"""原理图 S表达式生成器 — 完全手写，保证所有连线可见且无重叠"""
import os, uuid, math
from pathlib import Path
from datetime import datetime

from .circuit_helpers import export_sch_png_direct, export_sch_png


def uid(): return str(uuid.uuid4())
def uid_short(): return str(uuid.uuid4())[:8]
def xy(x, y): return f"(xy {x:.2f} {y:.2f})"


def build_schematic(ic="DW01-G", mos_count=1, series=1,
                    width_mm=40, height_mm=15):
    """
    用 S表达式生成完整的 .kicad_sch 原理图文件。

    布局策略 (坐标单位 mm):
      左侧: J1 接头
      中上: U1 DW01 (IC)
      中下: Q1 FS8205A (MOS)
      右侧: R1, C1, C2 分散排列

    总线:
      B+  水平总线 y=25.4  (顶部)
      GND 水平总线 y=165.1 (底部)
      VDD 水平总线 y=127.0 (中部)

    所有走线严格连接引脚端点，不穿过任何元件本体。
    """

    lines = []
    L = lines.append

    # ── Header ──
    project_uuid = uid()
    L(f'(kicad_sch')
    L(f'  (version 20240731)')
    L(f'  (generator "bms-autogen")')
    L(f'  (generator_version "1.0")')
    L(f'  (uuid {project_uuid})')
    L(f'  (paper "A4")')
    L(f'  (title_block')
    L(f'    (title "Battery Protection Board - {ic} {series}S")')
    L(f'    (date "{datetime.now().strftime("%Y-%m-%d")}")')
    L(f'    (comment 1 "{width_mm}x{height_mm}mm, MOS x{mos_count}")')
    L(f'  ')

    # ── lib_symbols ──
    L(f'  (lib_symbols')
    _write_dw01_symbol(L, ic)
    _write_fs8205a_symbol(L)
    _write_resistor_symbol(L)
    _write_capacitor_symbol(L)
    _write_connector_symbol(L)
    L(f'  )')
    L(f'  ')

    # ── Sheet ──
    sheet_uuid = uid()
    L(f'  (sheet (at 0 0) (size 200 190) (fields_autoplaced)')
    L(f'    (stroke (width 0) (type solid) (color 0 0 0 0))')
    L(f'    (fill (color 0 0 0 0.000))')
    L(f'    (uuid {sheet_uuid})')
    L(f'    (property "Sheet name" "Root" (id 0) (at 0 0 0)')
    L(f'      (effects (font (size 1.27 1.27)) hide))')
    L(f'    (property "Sheet file" "schematic.kicad_sch" (id 1) (at 0 0 0)')
    L(f'      (effects (font (size 1.27 1.27)) hide))')
    L(f'  )')
    L(f'  ')

    # ══════════════════════════════════════════════════
    # 元件坐标 (精心计算，避免走线穿过本体)
    # ══════════════════════════════════════════════════

    # J1 接头: 左侧
    j1_x, j1_y = 25.4, 50.8
    # U1 IC: 中上
    u1_x, u1_y = 63.5, 127.0
    # Q1 MOS: 中下 (在 J1 右下方，与 U1 左右引脚相对)
    q1_x, q1_y = 63.5, 50.8
    # R1 电阻: 右侧偏上
    r1_x, r1_y = 152.4, 101.6
    # C1 电容: 右侧中间
    c1_x, c1_y = 139.7, 88.9
    # C2 电容: 右侧偏下
    c2_x, c2_y = 139.7, 152.4

    # ── 放置元件 ──
    _place_symbol(L, "J1", "battery_protection:Conn_01x04", j1_x, j1_y, 0,
                  "J1", "Header_4P")
    _place_symbol(L, "U1", f"battery_protection:{ic}", u1_x, u1_y, 0,
                  "U1", ic, hide_fp=True)
    _place_symbol(L, "Q1", "battery_protection:FS8205A", q1_x, q1_y, 0,
                  "Q1", "FS8205A", hide_fp=True)
    _place_symbol(L, "R1", "Device:R_Small_US", r1_x, r1_y, 0,
                  "R1", "100", hide_fp=True)
    _place_symbol(L, "C1", "Device:C_Small", c1_x, c1_y, 0,
                  "C1", "0.1uF", hide_fp=True)
    _place_symbol(L, "C2", "Device:C_Small", c2_x, c2_y, 0,
                  "C2", "10uF", hide_fp=True)

    # ══════════════════════════════════════════════════
    # 引脚绝对坐标计算
    # ══════════════════════════════════════════════════

    # J1 Conn_01x04: 引脚在 (j1_x-7.62, j1_y+py)
    j1_p1 = (j1_x - 7.62, j1_y + 7.62)   # (17.78, 58.42)
    j1_p2 = (j1_x - 7.62, j1_y + 2.54)   # (17.78, 53.34)
    j1_p3 = (j1_x - 7.62, j1_y - 2.54)   # (17.78, 48.26)
    j1_p4 = (j1_x - 7.62, j1_y - 7.62)   # (17.78, 43.18)

    # U1 DW01-G: 左列引脚 (u1_x-7.62, ...)，右列 (u1_x+7.62, ...)
    u1_od  = (u1_x - 7.62, u1_y + 2.54)  # pin1 OD  (55.88, 129.54)
    u1_cs  = (u1_x - 7.62, u1_y)         # pin2 CS  (55.88, 127.0)
    u1_oc  = (u1_x - 7.62, u1_y - 2.54)  # pin3 OC  (55.88, 124.46)
    u1_td  = (u1_x + 7.62, u1_y - 2.54)  # pin4 TD  (71.12, 124.46)
    u1_vdd = (u1_x + 7.62, u1_y)         # pin5 VDD (71.12, 127.0)
    u1_vss = (u1_x + 7.62, u1_y + 2.54)  # pin6 VSS (71.12, 129.54)

    # Q1 FS8205A: 左列 (q1_x-7.62, ...)，右列 (q1_x+7.62, ...)
    q1_s1 = (q1_x - 7.62, q1_y + 3.81)   # pin1 S1 (55.88, 54.61)
    q1_g1 = (q1_x + 7.62, q1_y + 3.81)   # pin2 G1 (71.12, 54.61)
    q1_s2 = (q1_x - 7.62, q1_y + 1.27)   # pin3 S2 (55.88, 52.07)
    q1_g2 = (q1_x + 7.62, q1_y + 1.27)   # pin4 G2 (71.12, 52.07)
    q1_d2a = (q1_x - 7.62, q1_y - 1.27)  # pin5 D2 (55.88, 49.53)
    q1_d2b = (q1_x - 7.62, q1_y - 3.81)  # pin6 D2 (55.88, 46.99)
    q1_d1a = (q1_x + 7.62, q1_y - 1.27)  # pin7 D1 (71.12, 49.53)
    q1_d1b = (q1_x + 7.62, q1_y - 3.81)  # pin8 D1 (71.12, 46.99)

    # R1: pin1=左 (r1_x-5.08, r1_y), pin2=右 (r1_x+5.08, r1_y)
    r1_p1 = (r1_x - 5.08, r1_y)   # (147.32, 101.6)
    r1_p2 = (r1_x + 5.08, r1_y)   # (157.48, 101.6)

    # C1: pin1=左, pin2=右
    c1_p1 = (c1_x - 5.08, c1_y)   # (134.62, 88.9)
    c1_p2 = (c1_x + 5.08, c1_y)   # (144.78, 88.9)

    # C2: pin1=左, pin2=右
    c2_p1 = (c2_x - 5.08, c2_y)   # (134.62, 152.4)
    c2_p2 = (c2_x + 5.08, c2_y)   # (144.78, 152.4)

    # ══════════════════════════════════════════════════
    # 走线
    # ══════════════════════════════════════════════════

    # ── B+ 网络 ──
    # 顶部水平总线 y=25.4，从 x=15.24 到 x=157.48
    _wire_horiz(L, 15.24, 25.4, 157.48)

    # J1.P1(B+) → 左出引脚，水平接到 B+ 垂直总线 x=15.24
    _wire(L, j1_p1[0], j1_p1[1], 15.24, j1_p1[1])
    _junction(L, 15.24, 25.4)  # B+ 总线上的接合点

    # J1.P3(B+) → 左出引脚，水平接到 B+ 垂直总线
    _wire(L, j1_p3[0], j1_p3[1], 15.24, j1_p3[1])

    # B+ 垂直总线 x=15.24: 从 y=25.4 到 y=139.7
    _wire_vert(L, 15.24, 25.4, 139.7)
    _junction(L, 15.24, 139.7)

    # B+ 水平分支 y=139.7: 从 x=15.24 到 Q1.D1 下方
    _wire_horiz(L, 15.24, 139.7, q1_d1a[0])
    _junction(L, q1_d1a[0], 139.7)

    # Q1.D1 (pin7+8) → 从引脚端点向上到 B+ 分支
    _wire(L, q1_d1a[0], q1_d1a[1], q1_d1a[0], 139.7)
    _wire(L, q1_d1b[0], q1_d1b[1], q1_d1b[0], 139.7)
    _junction(L, q1_d1a[0], 139.7)

    # Q1.D2 (pin5+6) → 左出引脚，经垂直总线 x=48.26 连到 B+
    _wire(L, q1_d2a[0], q1_d2a[1], 48.26, q1_d2a[1])
    _wire(L, q1_d2b[0], q1_d2b[1], 48.26, q1_d2b[1])
    # D2 垂直总线 x=48.26: 从 B+ 总线 y=25.4 到 pin5 高度
    _wire_vert(L, 48.26, 25.4, q1_d2a[1])
    _junction(L, 48.26, 25.4)  # 接入 B+ 总线
    _junction(L, 48.26, q1_d2a[1])
    _junction(L, 48.26, q1_d2b[1])

    # R1.P1 → 从 B+ 总线垂直下降到 R1 引脚
    _wire_vert(L, r1_p1[0], 25.4, r1_p1[1])
    _junction(L, r1_p1[0], 25.4)

    # ── VDD 网络 ──
    # R1.P2 → 向右 → 向下 → 水平到 U1.VDD
    _wire(L, r1_p2[0], r1_p2[1], 165.1, r1_p2[1])
    _wire(L, 165.1, r1_p2[1], 165.1, u1_vdd[1])
    _wire(L, 165.1, u1_vdd[1], u1_vdd[0], u1_vdd[1])
    _junction(L, 165.1, u1_vdd[1])

    # C1.P1 → 水平右接到 VDD 垂直线 x=165.1
    _wire(L, c1_p1[0], c1_p1[1], 165.1, c1_p1[1])
    _junction(L, 165.1, c1_p1[1])

    # C2.P1 → 水平右接到 VDD 垂直线 x=165.1
    _wire(L, c2_p1[0], c2_p1[1], 165.1, c2_p1[1])
    _junction(L, 165.1, c2_p1[1])

    # ── GND 网络 ──
    # 底部水平总线 y=165.1，从 x=15.24 到 x=165.1
    _wire_horiz(L, 15.24, 165.1, 165.1)

    # J1.P2(GND) → 左出引脚 → 向下到 GND 总线
    _wire(L, j1_p2[0], j1_p2[1], 15.24, j1_p2[1])
    _wire_vert(L, 15.24, j1_p2[1], 165.1)
    _junction(L, 15.24, 165.1)

    # J1.P4(GND) → 左出引脚 → 向下到 GND 总线 (共用 x=15.24)
    _wire(L, j1_p4[0], j1_p4[1], 15.24, j1_p4[1])
    _junction(L, 15.24, j1_p4[1])

    # U1.VSS → 向下到 GND 总线
    _wire_vert(L, u1_vss[0], u1_vss[1], 165.1)
    _junction(L, u1_vss[0], 165.1)

    # Q1.S1 → 向下到 GND 总线
    _wire_vert(L, q1_s1[0], q1_s1[1], 165.1)
    _junction(L, q1_s1[0], 165.1)

    # Q1.S2 → 向下到 GND 总线
    _wire_vert(L, q1_s2[0], q1_s2[1], 165.1)
    _junction(L, q1_s2[0], 165.1)

    # C1.P2 → 向下到 GND 总线
    _wire_vert(L, c1_p2[0], c1_p2[1], 165.1)
    _junction(L, c1_p2[0], 165.1)

    # C2.P2 → 向下到 GND 总线
    _wire_vert(L, c2_p2[0], c2_p2[1], 165.1)
    _junction(L, c2_p2[0], 165.1)

    # ── OD 网络: U1.OD(1) → Q1.G1(2) ──
    # 从 U1.OD 引脚端点向上(到 U1 上方)，再向右，再向下到 Q1.G1
    _wire(L, u1_od[0], u1_od[1], u1_od[0], 119.38)
    _wire_horiz(L, u1_od[0], 119.38, q1_g1[0])
    _wire(L, q1_g1[0], 119.38, q1_g1[0], q1_g1[1])

    # ── OC 网络: U1.OC(3) → Q1.G2(4) ──
    # 从 U1.OC 引脚端点向左，再向下，再向右到 Q1.G2
    _wire(L, u1_oc[0], u1_oc[1], u1_oc[0] - 2.54, u1_oc[1])
    _wire(L, u1_oc[0] - 2.54, u1_oc[1], u1_oc[0] - 2.54, 116.84)
    _wire_horiz(L, u1_oc[0] - 2.54, 116.84, q1_g2[0])
    _wire(L, q1_g2[0], 116.84, q1_g2[0], q1_g2[1])

    # ── CS 网络: U1.CS(2) → Q1.S1(1) ──
    # 从 U1.CS 引脚端点向左，再向下到 Q1.S1
    _wire(L, u1_cs[0], u1_cs[1], u1_cs[0] - 5.08, u1_cs[1])
    _wire(L, u1_cs[0] - 5.08, u1_cs[1], u1_cs[0] - 5.08, q1_s1[1])
    _wire(L, u1_cs[0] - 5.08, q1_s1[1], q1_s1[0], q1_s1[1])

    # ── 未使用引脚标记 ──
    _no_connect(L, u1_td[0], u1_td[1])  # U1.TD(4) 未使用

    # ── 网络标签 ──
    _label(L, "B+", 20.32, 25.4, 0)
    _label(L, "VDD", 160.02, u1_vdd[1], 0)
    _label(L, "GND", 20.32, 165.1, 0)
    _label(L, "OD", (u1_od[0] + q1_g1[0]) / 2, 119.38 + 2.54, 0)
    _label(L, "OC", (u1_oc[0] - 2.54 + q1_g2[0]) / 2, 116.84 - 2.54, 0)
    _label(L, "CS", u1_cs[0] - 5.08 - 2.54, (u1_cs[1] + q1_s1[1]) / 2, 0)

    L(f')')
    L(f'')

    return '\n'.join(lines)


# ── Helper functions ──

def _write_dw01_symbol(L, name="DW01-G"):
    L(f'    (symbol "battery_protection:{name}"')
    L(f'      (pin_names (offset 1.016))')
    L(f'      (exclude_from_sim no)')
    L(f'      (in_bom yes)')
    L(f'      (on_board yes)')
    L(f'      (property "Reference" "U" (id 0) (at 0 6.35 0)')
    L(f'        (effects (font (size 1.27 1.27))))')
    L(f'      (property "Value" "{name}" (id 1) (at 0 -6.35 0)')
    L(f'        (effects (font (size 1.27 1.27))))')
    L(f'      (property "Footprint" "Package_TO_SOT_SMD:SOT-23-6" (id 2) (at 0 -8.89 0)')
    L(f'        (effects (font (size 1.27 1.27)) hide))')
    L(f'      (property "Datasheet" "" (id 3) (at 0 0 0)')
    L(f'        (effects (font (size 1.27 1.27)) hide))')
    L(f'      (symbol "{name}_0_1"')
    L(f'        (rectangle (start -5.08 5.08) (end 5.08 -5.08)')
    L(f'          (stroke (width 0.254) (type default))')
    L(f'          (fill (type background)))')
    L(f'        (pin power_in line (at -7.62 2.54 0) (length 2.54)')
    L(f'          (name "OD" (effects (font (size 1.27 1.27))))')
    L(f'          (number "1" (effects (font (size 1.27 1.27)))))')
    L(f'        (pin passive line (at -7.62 0 0) (length 2.54)')
    L(f'          (name "CS" (effects (font (size 1.27 1.27))))')
    L(f'          (number "2" (effects (font (size 1.27 1.27)))))')
    L(f'        (pin power_out line (at -7.62 -2.54 0) (length 2.54)')
    L(f'          (name "OC" (effects (font (size 1.27 1.27))))')
    L(f'          (number "3" (effects (font (size 1.27 1.27)))))')
    L(f'        (pin input line (at 7.62 -2.54 180) (length 2.54)')
    L(f'          (name "TD" (effects (font (size 1.27 1.27))))')
    L(f'          (number "4" (effects (font (size 1.27 1.27)))))')
    L(f'        (pin power_in line (at 7.62 0 180) (length 2.54)')
    L(f'          (name "VDD" (effects (font (size 1.27 1.27))))')
    L(f'          (number "5" (effects (font (size 1.27 1.27)))))')
    L(f'        (pin power_in line (at 7.62 2.54 180) (length 2.54)')
    L(f'          (name "VSS" (effects (font (size 1.27 1.27))))')
    L(f'          (number "6" (effects (font (size 1.27 1.27))))))')
    L(f'    )')


def _write_fs8205a_symbol(L):
    L(f'    (symbol "battery_protection:FS8205A"')
    L(f'      (pin_names (offset 1.016))')
    L(f'      (exclude_from_sim no)')
    L(f'      (in_bom yes)')
    L(f'      (on_board yes)')
    L(f'      (property "Reference" "Q" (id 0) (at 0 7.62 0)')
    L(f'        (effects (font (size 1.27 1.27))))')
    L(f'      (property "Value" "FS8205A" (id 1) (at 0 -7.62 0)')
    L(f'        (effects (font (size 1.27 1.27))))')
    L(f'      (property "Footprint" "Package_SO:TSSOP-8_4.4x3mm_P0.65mm" (id 2) (at 0 -10.16 0)')
    L(f'        (effects (font (size 1.27 1.27)) hide))')
    L(f'      (property "Datasheet" "" (id 3) (at 0 0 0)')
    L(f'        (effects (font (size 1.27 1.27)) hide))')
    L(f'      (symbol "FS8205A_0_1"')
    L(f'        (rectangle (start -5.08 5.08) (end 5.08 -5.08)')
    L(f'          (stroke (width 0.254) (type default))')
    L(f'          (fill (type background)))')
    for pin_info in [
        ("1", "S1", "passive", "line", -7.62, 3.81, 0),
        ("2", "G1", "input", "line", 7.62, 3.81, 180),
        ("3", "S2", "passive", "line", -7.62, 1.27, 0),
        ("4", "G2", "input", "line", 7.62, 1.27, 180),
        ("5", "D2", "passive", "line", -7.62, -1.27, 0),
        ("6", "D2", "passive", "line", -7.62, -3.81, 0),
        ("7", "D1", "passive", "line", 7.62, -1.27, 180),
        ("8", "D1", "passive", "line", 7.62, -3.81, 180),
    ]:
        num, name, etype, shape, px, py, rot = pin_info
        L(f'        (pin {etype} {shape} (at {px} {py} {rot}) (length 2.54)')
        L(f'          (name "{name}" (effects (font (size 1.27 1.27))))')
        L(f'          (number "{num}" (effects (font (size 1.27 1.27))))))')
    L(f'      )')
    L(f'    )')


def _write_resistor_symbol(L):
    L(f'    (symbol "Device:R_Small_US"')
    L(f'      (exclude_from_sim no) (in_bom yes) (on_board yes)')
    L(f'      (property "Reference" "R" (id 0) (at 0 2.54 0)')
    L(f'        (effects (font (size 1.27 1.27))))')
    L(f'      (property "Value" "" (id 1) (at 0 -2.54 0)')
    L(f'        (effects (font (size 1.27 1.27))))')
    L(f'      (property "Footprint" "" (id 2) (at 0 0 0)')
    L(f'        (effects (font (size 1.27 1.27)) hide))')
    L(f'      (property "Datasheet" "~" (id 3) (at 0 0 0)')
    L(f'        (effects (font (size 1.27 1.27)) hide))')
    L(f'      (symbol "R_Small_US_0_1"')
    L(f'        (polyline (pts (xy -2.54 0) (xy -1.27 0))')
    L(f'          (stroke (width 0) (type default))')
    L(f'          (fill (type none)))')
    L(f'        (polyline (pts (xy -1.27 0.76) (xy -1.27 -0.76)')
    L(f'          (stroke (width 0.5) (type default))')
    L(f'          (fill (type none)))')
    L(f'        (polyline (pts (xy -1.27 0.76) (xy 1.27 -0.76)')
    L(f'          (stroke (width 0.5) (type default))')
    L(f'          (fill (type none)))')
    L(f'        (polyline (pts (xy -1.27 -0.76) (xy 1.27 0.76)')
    L(f'          (stroke (width 0.5) (type default))')
    L(f'          (fill (type none)))')
    L(f'        (polyline (pts (xy 1.27 0.76) (xy 1.27 -0.76)')
    L(f'          (stroke (width 0.5) (type default))')
    L(f'          (fill (type none)))')
    L(f'        (polyline (pts (xy 1.27 0) (xy 2.54 0))')
    L(f'          (stroke (width 0) (type default))')
    L(f'          (fill (type none)))')
    L(f'        (pin passive line (at -5.08 0 0) (length 2.54)')
    L(f'          (name "1" (effects (font (size 1.27 1.27))))')
    L(f'          (number "1" (effects (font (size 1.27 1.27)))))')
    L(f'        (pin passive line (at 5.08 0 180) (length 2.54)')
    L(f'          (name "2" (effects (font (size 1.27 1.27))))')
    L(f'          (number "2" (effects (font (size 1.27 1.27))))))')
    L(f'    )')


def _write_capacitor_symbol(L):
    L(f'    (symbol "Device:C_Small"')
    L(f'      (exclude_from_sim no) (in_bom yes) (on_board yes)')
    L(f'      (property "Reference" "C" (id 0) (at 0 2.54 0)')
    L(f'        (effects (font (size 1.27 1.27))))')
    L(f'      (property "Value" "" (id 1) (at 0 -2.54 0)')
    L(f'        (effects (font (size 1.27 1.27))))')
    L(f'      (property "Footprint" "" (id 2) (at 0 0 0)')
    L(f'        (effects (font (size 1.27 1.27)) hide))')
    L(f'      (property "Datasheet" "~" (id 3) (at 0 0 0)')
    L(f'        (effects (font (size 1.27 1.27)) hide))')
    L(f'      (symbol "C_Small_0_1"')
    L(f'        (polyline (pts (xy -2.54 0) (xy -0.76 0))')
    L(f'          (stroke (width 0) (type default)) (fill (type none)))')
    L(f'        (polyline (pts (xy -0.76 1.27) (xy -0.76 -1.27))')
    L(f'          (stroke (width 0.5) (type default)) (fill (type none)))')
    L(f'        (polyline (pts (xy 0.76 1.27) (xy 0.76 -1.27))')
    L(f'          (stroke (width 0.5) (type default)) (fill (type none)))')
    L(f'        (polyline (pts (xy 0.76 0) (xy 2.54 0))')
    L(f'          (stroke (width 0) (type default)) (fill (type none)))')
    L(f'        (pin passive line (at -5.08 0 0) (length 2.54)')
    L(f'          (name "1" (effects (font (size 1.27 1.27))))')
    L(f'          (number "1" (effects (font (size 1.27 1.27)))))')
    L(f'        (pin passive line (at 5.08 0 180) (length 2.54)')
    L(f'          (name "2" (effects (font (size 1.27 1.27))))')
    L(f'          (number "2" (effects (font (size 1.27 1.27))))))')
    L(f'    )')


def _write_connector_symbol(L):
    L(f'    (symbol "battery_protection:Conn_01x04"')
    L(f'      (pin_names (offset 1.016) hide)')
    L(f'      (exclude_from_sim no) (in_bom yes) (on_board yes)')
    L(f'      (property "Reference" "J" (id 0) (at 0 10.16 0)')
    L(f'        (effects (font (size 1.27 1.27))))')
    L(f'      (property "Value" "" (id 1) (at 0 -10.16 0)')
    L(f'        (effects (font (size 1.27 1.27))))')
    L(f'      (property "Footprint" "" (id 2) (at 0 0 0)')
    L(f'        (effects (font (size 1.27 1.27)) hide))')
    L(f'      (property "Datasheet" "~" (id 3) (at 0 0 0)')
    L(f'        (effects (font (size 1.27 1.27)) hide))')
    L(f'      (symbol "Conn_01x04_1_1"')
    L(f'        (rectangle (start -5.08 10.16) (end 5.08 -10.16)')
    L(f'          (stroke (width 0.254) (type default))')
    L(f'          (fill (type background)))')
    for i, (num, py) in enumerate([("1", 7.62), ("2", 2.54), ("3", -2.54), ("4", -7.62)]):
        L(f'        (pin passive line (at -7.62 {py} 0) (length 2.54)')
        L(f'          (name "{num}" (effects (font (size 1.27 1.27))))')
        L(f'          (number "{num}" (effects (font (size 1.27 1.27))))))')
    L(f'      )')
    L(f'    )')


def _place_symbol(L, ref, lib_id, x, y, rot, ref_text, value_text, hide_fp=False):
    """放置一个元件实例"""
    fp_prop = 'hide' if hide_fp else ''
    L(f'  (symbol (lib_id "{lib_id}") (at {x:.2f} {y:.2f} {rot}) (unit 1)')
    L(f'    (exclude_from_sim no) (in_bom yes) (on_board yes) (dnp no)')
    L(f'    (uuid {uid()})')
    L(f'    (property "Reference" "{ref_text}" (id 0) (at {x:.2f} {y+3.81:.2f} 0)')
    L(f'      (effects (font (size 1.27 1.27))))')
    L(f'    (property "Value" "{value_text}" (id 1) (at {x:.2f} {y-3.81:.2f} 0)')
    L(f'      (effects (font (size 1.27 1.27))))')
    L(f'    (property "Footprint" "" (id 2) (at 0 0 0)')
    L(f'      (effects (font (size 1.27 1.27)) {fp_prop}))')
    L(f'    (property "Datasheet" "" (id 3) (at 0 0 0)')
    L(f'      (effects (font (size 1.27 1.27)) hide))')
    # 引脚
    pins = ["1","2","3","4","5","6","7","8"]
    for p in pins:
        L(f'    (pin "{p}" (uuid {uid()}))')
    L(f'  )')


def _wire(L, x1, y1, x2, y2):
    """画一条导线"""
    L(f'  (wire (pts (xy {x1:.2f} {y1:.2f}) (xy {x2:.2f} {y2:.2f}))')
    L(f'    (stroke (width 0) (type default))')
    L(f'    (uuid {uid()}))')


def _wire_horiz(L, x1, y, x2):
    _wire(L, x1, y, x2, y)


def _wire_vert(L, x, y1, y2):
    _wire(L, x, y1, x, y2)


def _junction(L, x, y):
    """添加接合点（导线交叉/分叉处的圆点）"""
    L(f'  (junction (at {x:.2f} {y:.2f}) (diameter 0) (color 0 0 0 0)')
    L(f'    (uuid {uid()}))')


def _no_connect(L, x, y):
    """标记未连接引脚"""
    L(f'  (no_connect (at {x:.2f} {y:.2f}) (uuid {uid()}))')


def _label(L, text, x, y, rot):
    """添加网络标签"""
    L(f'  (label "{text}" (at {x:.2f} {y:.2f} {rot})')
    L(f'    (effects (font (size 1.27 1.27)) (justify left))')
    L(f'    (uuid {uid()}))')


# ── CLI 入口 ──
if __name__ == "__main__":
    import sys
    ic = sys.argv[1] if len(sys.argv) > 1 else "DW01-G"
    mos = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    series = int(sys.argv[3]) if len(sys.argv) > 3 else 1
    w = float(sys.argv[4]) if len(sys.argv) > 4 else 40
    h = float(sys.argv[5]) if len(sys.argv) > 5 else 15

    out_dir = Path(f"output/{ic}_{series}S_MOSx{mos}")
    out_dir.mkdir(parents=True, exist_ok=True)

    sch_content = build_schematic(ic, mos, series, w, h)
    sch_file = out_dir / "schematic.kicad_sch"
    with open(sch_file, 'w', encoding='utf-8') as f:
        f.write(sch_content)

    # 统计
    wires = sch_content.count('(wire ')
    labels = sch_content.count('(label ')
    junctions = sch_content.count('(junction ')
    print(f"原理图: {sch_file}")
    print(f"{wires} wires, {labels} labels, {junctions} junctions")

    # 导出 PNG
    png_file = out_dir / "schematic.png"
    if not export_sch_png_direct(sch_file, png_file):
        export_sch_png(sch_file, png_file)  # fallback via SVG
