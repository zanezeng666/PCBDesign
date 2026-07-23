"""原理图 S表达式生成器 — 完全手写，保证所有连线可见"""
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
    
    布局 (坐标单位: 50 = 1 inch grid):
      [J1] 接头        [R1] 电阻
      [U1] DW01        [C1] 电容
      [Q1] MOS         [C2] 电容
    """
    
    lines = []
    L = lines.append  # shortcut
    
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
    
    # ── lib_symbols (嵌入式符号定义) ──
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
    L(f'  (sheet (at 0 0) (size 200 150) (fields_autoplaced)')
    L(f'    (stroke (width 0) (type solid) (color 0 0 0 0))')
    L(f'    (fill (color 0 0 0 0.000))')
    L(f'    (uuid {sheet_uuid})')
    L(f'    (property "Sheet name" "Root" (id 0) (at 0 0 0)') 
    L(f'      (effects (font (size 1.27 1.27)) hide))')
    L(f'    (property "Sheet file" "schematic.kicad_sch" (id 1) (at 0 0 0)')
    L(f'      (effects (font (size 1.27 1.27)) hide))')
    L(f'  )')
    L(f'  ')
    
    # ── 元件坐标 ──
    # J1 接头: 左下 (column 2, row 1)
    # U1 DW01: 中上 (column 4, row 9)
    # Q1 MOS:  中下 (column 4, row 3)
    # R1:      右上 (column 8, row 9)
    # C1:      右中 (column 8, row 6)
    # C2:      右下 (column 8, row 3)
    
    j1_x, j1_y = 25.4, 25.4     # 1" x 1"
    u1_x, u1_y = 63.5, 127.0    # 2.5" x 5" 
    q1_x, q1_y = 63.5, 50.8     # 2.5" x 2"
    r1_x, r1_y = 127.0, 127.0   # 5" x 5"
    c1_x, c1_y = 127.0, 88.9    # 5" x 3.5"
    c2_x, c2_y = 127.0, 50.8    # 5" x 2"
    
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
    
    # ── 连线 ──
    # 获取各元件的引脚位置 (符号坐标 + 引脚相对偏移)
    # 这些需要根据实际符号定义来算。先按符号的引脚偏移估算。
    
    # DW01-G 引脚 (引脚号->相对位置, 方向):
    # 1(OD): left,  (u1_x-7.62, u1_y+2.54)  -- input side
    # 2(CS): left,  (u1_x-7.62, u1_y+0)
    # 3(OC): left,  (u1_x-7.62, u1_y-2.54)
    # 4(TD): right, (u1_x+7.62, u1_y-2.54)
    # 5(VDD):right, (u1_x+7.62, u1_y+0)
    # 6(VSS):right, (u1_x+7.62, u1_y+2.54)
    
    # 简化：估算关键引脚坐标
    u1_od_x, u1_od_y = u1_x - 7.62, u1_y + 2.54
    u1_cs_x, u1_cs_y = u1_x - 7.62, u1_y
    u1_oc_x, u1_oc_y = u1_x - 7.62, u1_y - 2.54
    u1_vdd_x, u1_vdd_y = u1_x + 7.62, u1_y
    u1_vss_x, u1_vss_y = u1_x + 7.62, u1_y + 2.54
    
    # Q1 FS8205A 引脚 (左侧: S1,S2,D2,D2; 右侧: G1,G2,D1,D1)
    q1_s1_x, q1_s1_y = q1_x - 7.62, q1_y + 3.81
    q1_g1_x, q1_g1_y = q1_x + 7.62, q1_y + 3.81
    q1_s2_x, q1_s2_y = q1_x - 7.62, q1_y + 1.27
    q1_g2_x, q1_g2_y = q1_x + 7.62, q1_y + 1.27
    q1_d2_x, q1_d2_y = q1_x - 7.62, q1_y - 1.27
    q1_d1_x, q1_d1_y = q1_x + 7.62, q1_y - 3.81
    
    # R1 电阻 (水平: 1=左, 2=右, 引脚X偏移~5.08)
    r1_1_x, r1_1_y = r1_x - 5.08, r1_y
    r1_2_x, r1_2_y = r1_x + 5.08, r1_y
    
    # C1 电容 (水平: 1=左, 2=右)
    c1_1_x, c1_1_y = c1_x - 5.08, c1_y
    c1_2_x, c1_2_y = c1_x + 5.08, c1_y
    
    # C2 电容
    c2_1_x, c2_1_y = c2_x - 5.08, c2_y
    c2_2_x, c2_2_y = c2_x + 5.08, c2_y
    
    # J1 接头 (4pin, pin 1-4 从上到下，左侧引脚)
    j1_p1_x, j1_p1_y = j1_x - 5.08, j1_y + 7.62
    j1_p2_x, j1_p2_y = j1_x - 5.08, j1_y + 2.54
    j1_p3_x, j1_p3_y = j1_x - 5.08, j1_y - 2.54
    j1_p4_x, j1_p4_y = j1_x - 5.08, j1_y - 7.62
    
    # ── 画线 ──
    wires = []
    
    # 1. B+ = P+ 网络: R1.1 - Q1.D1(7,8) - J1.1(P+) - J1.3(P+)
    _wire(L, q1_d1_x, q1_d1_y, q1_d1_x, j1_1_y)
    _wire(L, q1_d1_x, j1_1_y, r1_1_x, j1_1_y)
    _wire(L, r1_1_x, j1_1_y, r1_1_x, r1_1_y)  # 接到R1
    
    # J1.1 = J1.3 (P+ = P+ 内部短接)
    _wire(L, j1_p1_x, j1_p1_y, j1_p1_x - 2.54, j1_p1_y)
    _wire_vert(L, j1_p1_x - 2.54, j1_p1_y, j1_p3_y)
    _wire(L, j1_p1_x - 2.54, j1_p3_y, j1_p3_x, j1_p3_y)
    # 也连到主干
    _wire_horiz(L, j1_p1_x - 2.54, j1_1_y, r1_1_x)
    
    # 2. VDD 网络: R1.2 - U1.VDD(5) - C1.1
    _wire(L, r1_2_x, r1_2_y, r1_2_x + 5.08, r1_2_y)
    _wire(L, r1_2_x + 5.08, r1_2_y, r1_2_x + 5.08, u1_vdd_y)
    _wire(L, r1_2_x + 5.08, u1_vdd_y, u1_vdd_x, u1_vdd_y)
    # C1.1 也接 VDD
    _wire_horiz(L, r1_2_x + 5.08, c1_1_y, c1_1_x)
    
    # 3. B- = GND 网络: U1.VSS(6) - C1.2 - C2.2 - Q1.S1(1) - Q1.S2(3) - J1.2 - J1.4
    # 主干横线
    gnd_y = q1_y - 15.24  # 公共地线位置
    _wire_horiz(L, u1_vss_x, gnd_y, j1_p2_x)
    _wire(L, u1_vss_x, u1_vss_y, u1_vss_x, gnd_y)   # U1.VSS 向下
    _wire(L, c1_2_x, c1_2_y, c1_2_x, gnd_y)          # C1.2 向下
    _wire(L, c2_2_x, c2_2_y, c2_2_x, gnd_y)          # C2.2 向下
    _wire(L, q1_s1_x, q1_s1_y, q1_s1_x, gnd_y)       # Q1.S1 向下
    _wire(L, q1_s2_x, q1_s2_y, q1_s2_x, gnd_y)       # Q1.S2 向下
    _wire(L, j1_p2_x, j1_p2_y, j1_p2_x, gnd_y)        # J1.2 向上
    _wire(L, j1_p4_x, j1_p4_y, j1_p4_x, gnd_y)        # J1.4 向上
    # U1.VSS(6)
    _wire(L, u1_vss_x, u1_vss_y, u1_vss_x, gnd_y)
    
    # 4. OD 网络: U1.OD(1) - Q1.G1(2) (水平连线)
    od_y = u1_y + 2.54
    _wire_horiz(L, u1_od_x, od_y, q1_g1_x)
    
    # 5. OC 网络: U1.OC(3) - Q1.G2(4) (水平连线)
    oc_y = u1_y - 2.54
    _wire_horiz(L, u1_oc_x, oc_y, q1_g2_x)
    
    # 6. CS 网络: U1.CS(2) - Q1.S1(1) (检测电流)
    # 从 U1.CS 向下弯到 Q1.S1
    cs_y = u1_y
    _wire(L, u1_cs_x, cs_y, q1_s1_x, cs_y)
    _wire(L, q1_s1_x, cs_y, q1_s1_x, q1_s1_y)
    
    # 7. Q1.D2(5,6) 内部连接 (D2连D2) — 不需要额外连线，同一pin号
    
    # ── 网络标签 (方便阅读) ──
    _label(L, "B+", r1_1_x - 2.54, r1_1_y, 0)
    _label(L, "VDD", r1_2_x + 2.54, r1_2_y, 0)
    _label(L, "GND", c2_2_x, gnd_y, 0)
    _label(L, "OD", (u1_od_x + q1_g1_x)/2, od_y + 2.54, 0)
    _label(L, "OC", (u1_oc_x + q1_g2_x)/2, oc_y - 2.54, 0)
    _label(L, "CS", (u1_cs_x + q1_s1_x)/2, cs_y + 2.54 + 2.54, 0)
    
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
    # 简化的 R 符号 — 内嵌基础符号避免依赖外部库
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
    """画一条线"""
    L(f'  (wire (pts (xy {x1:.2f} {y1:.2f}) (xy {x2:.2f} {y2:.2f}))')
    L(f'    (stroke (width 0) (type default))')
    L(f'    (uuid {uid()}))')


def _wire_horiz(L, x1, y, x2):
    _wire(L, x1, y, x2, y)


def _wire_vert(L, x, y1, y2):
    _wire(L, x, y1, x, y2)


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
    print(f"原理图: {sch_file}")
    print(f"{wires} wires, {labels} labels")
    
    # 导出 PNG
    png_file = out_dir / "schematic.png"
    if not export_sch_png_direct(sch_file, png_file):
        export_sch_png(sch_file, png_file)  # fallback via SVG
