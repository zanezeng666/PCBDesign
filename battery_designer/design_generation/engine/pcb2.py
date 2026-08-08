"""
生成 PCB 文件 — 使用 SKiDL 的 generate_pcb
"""
import os
from pathlib import Path

from .config import KICAD_FOOTPRINT_DIR
from .circuit_helpers import init_skidl, build_circuit_from_yaml, create_nets_from_yaml

import yaml
import skidl
from skidl import KICAD9, generate_pcb

# 收集所有封装库路径
fp_lib_dirs = [KICAD_FOOTPRINT_DIR]
if KICAD_FOOTPRINT_DIR.exists():
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

    custom_lib = init_skidl()
    parts = build_circuit_from_yaml(circuit, custom_lib)
    create_nets_from_yaml(circuit, parts)

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
