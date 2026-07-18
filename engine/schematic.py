"""
电池保护板原理图生成器
基于 SKiDL + YAML电路定义 -> 网表 + 原理图
"""
import os
import yaml
from pathlib import Path

KICAD_PATH = r"C:\Program Files\KiCad\9.0"
KICAD_SHARE = os.path.join(KICAD_PATH, "share", "kicad")
KICAD_SYMBOL_DIR = os.path.join(KICAD_SHARE, "symbols")
KICAD_FOOTPRINT_DIR = os.path.join(KICAD_SHARE, "footprints")

os.environ["KICAD_SYMBOL_DIR"] = KICAD_SYMBOL_DIR
os.environ["KICAD9_SYMBOL_DIR"] = KICAD_SYMBOL_DIR
os.environ["KICAD8_SYMBOL_DIR"] = KICAD_SYMBOL_DIR
os.environ["KICAD7_SYMBOL_DIR"] = KICAD_SYMBOL_DIR
os.environ["KICAD6_SYMBOL_DIR"] = KICAD_SYMBOL_DIR
os.environ["KICAD_FOOTPRINT_DIR"] = KICAD_FOOTPRINT_DIR

import skidl
from skidl import Net, Part, KICAD9, SchLib, generate_netlist

CUSTOM_SYM_LIB = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "circuits", "symbols", "battery_protection.kicad_sym"
)


def generate_from_yaml(yaml_path: str, output_dir: str) -> dict:
    with open(yaml_path, "r", encoding="utf-8") as f:
        circuit = yaml.safe_load(f)

    skidl.set_default_tool(KICAD9)

    print(f"[1/3] 解析电路: {circuit['name']}")
    print(f"       类型: {circuit['type']} | 板尺寸: {circuit.get('board', {}).get('width_mm', '?')}x{circuit.get('board', {}).get('height_mm', '?')}mm")
    print(f"       元件数: {len(circuit['parts'])}, 网络数: {len(circuit['nets'])}")

    custom_lib = SchLib(CUSTOM_SYM_LIB) if os.path.exists(CUSTOM_SYM_LIB) else None

    # 创建元件
    parts = {}
    print("\n[2/3] 创建元件...")
    for part_def in circuit["parts"]:
        pid = part_def["id"]
        pname = part_def["part_name"]
        cat = part_def["category"]

        if cat in ("IC", "MOSFET_DUAL") and custom_lib:
            p = Part(custom_lib, pname, footprint=part_def["footprint"])
        elif cat == "RES":
            try: p = Part("Device", "R_Small_US", footprint=part_def["footprint"])
            except: p = Part("Device", "R", footprint=part_def["footprint"])
        elif cat == "CAP":
            try: p = Part("Device", "C_Small", footprint=part_def["footprint"])
            except: p = Part("Device", "C", footprint=part_def["footprint"])
        elif cat == "CONN":
            try: p = Part("Connector_Generic", "Conn_01x04", footprint=part_def["footprint"])
            except: p = Part("Connector", "Conn_01x04", footprint=part_def["footprint"])
        else:
            print(f"       [跳过] {pid}: 未知类型")
            continue

        p.ref = pid
        p.value = part_def["value"]
        parts[pid] = p
        print(f"       {pid}: {pname} ({part_def['footprint']})")

    # 创建网络
    print("\n[3/3] 创建网络连接...")
    nets = {}
    for net_def in circuit["nets"]:
        net = Net(net_def["name"])
        nets[net_def["name"]] = net
        for conn in net_def["connections"]:
            part_id = conn["part"]
            pin_name = str(conn["pin"])
            if part_id in parts:
                try:
                    parts[part_id][pin_name] += net
                except Exception as e:
                    print(f"       ! 连接失败 {part_id}.{pin_name}: {e}")

    # 直接生成网表（最可靠的输出）
    sch_name = circuit.get("name", "schematic").replace(" ", "_")
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    netlist_file = output_path / f"{sch_name}.net"
    print(f"\n生成网表: {netlist_file}")
    generate_netlist(filepath=str(netlist_file))

    # 也试图生成原理图（可选）
    sch_file = output_path / f"{sch_name}.kicad_sch"
    print(f"生成原理图: {sch_file}")
    generated_sch = False
    try:
        skidl.generate_schematic(filepath=str(sch_file))
        generated_sch = True
    except Exception as e:
        print(f"       (原理图自动布局跳过: {type(e).__name__})")

    result = {
        "netlist_path": str(netlist_file),
        "sch_path": str(sch_file) if generated_sch else None,
        "name": circuit["name"],
        "type": circuit["type"],
        "board": circuit.get("board", {}),
        "parts": {pid: {"ref": p.ref, "value": p.value, "footprint": p.footprint}
                  for pid, p in parts.items()},
        "net_count": len(nets),
    }

    print(f"\n[完成] 网表: {result['netlist_path']}")
    if result["sch_path"]:
        print(f"[完成] 原理图: {result['sch_path']}")
    
    return result


if __name__ == "__main__":
    import sys
    yaml_file = sys.argv[1] if len(sys.argv) > 1 else "engine/circuits/protections/dw01_1s.yaml"
    output = sys.argv[2] if len(sys.argv) > 2 else "output/dw01_test"
    generate_from_yaml(yaml_file, output)
