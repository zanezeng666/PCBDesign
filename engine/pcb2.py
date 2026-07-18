"""
生成 PCB 文件 — 使用 SKiDL 的 generate_pcb
"""
import os
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

import yaml
import skidl
from skidl import Net, Part, KICAD9, SchLib, generate_pcb

CUSTOM_SYM_LIB = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "circuits", "symbols", "battery_protection.kicad_sym"
)

# 收集所有封装库路径
fp_lib_dirs = [KICAD_FOOTPRINT_DIR]
for root, dirs, files in os.walk(KICAD_FOOTPRINT_DIR):
    for d in dirs:
        if d.endswith('.pretty'):
            fp_lib_dirs.append(os.path.join(root, d))

print(f"封装库目录数: {len(fp_lib_dirs)}")


def generate_pcb_from_yaml(yaml_path: str, output_dir: str, 
                            width_mm: float = 40, height_mm: float = 15):
    """从 YAML 电路定义生成 PCB"""
    
    with open(yaml_path, "r", encoding="utf-8") as f:
        circuit = yaml.safe_load(f)

    skidl.set_default_tool(KICAD9)
    
    custom_lib = SchLib(CUSTOM_SYM_LIB) if os.path.exists(CUSTOM_SYM_LIB) else None
    
    # 创建元件
    parts = {}
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
            continue
        
        p.ref = pid
        p.value = part_def["value"]
        parts[pid] = p

    # 创建网络
    for net_def in circuit["nets"]:
        net = Net(net_def["name"])
        for conn in net_def["connections"]:
            part_id = conn["part"]
            pin_name = str(conn["pin"])
            if part_id in parts:
                try:
                    parts[part_id][pin_name] += net
                except:
                    pass

    # 生成 PCB
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    sch_name = circuit.get("name", "schematic").replace(" ", "_")
    pcb_file = output_path / f"{sch_name}.kicad_pcb"
    
    print(f"生成 PCB: {pcb_file}")
    print(f"板尺寸: {width_mm}x{height_mm}mm")
    
    board_dim = (int(width_mm), int(height_mm))
    
    try:
        generate_pcb(
            file_=str(pcb_file),
            tool=KICAD9,
            fp_libs=fp_lib_dirs,
            board_dim=board_dim,
        )
        print(f"PCB 生成成功: {pcb_file}")
        return {"pcb_path": str(pcb_file), "status": "ok"}
    except Exception as e:
        print(f"SKiDL PCB 生成失败: {e}")
        return {"pcb_path": None, "status": "failed", "error": str(e)}


if __name__ == "__main__":
    result = generate_pcb_from_yaml(
        "engine/circuits/protections/dw01_1s.yaml",
        "output/dw01_pcb"
    )
    print(result)
