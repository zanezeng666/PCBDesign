"""
电池保护板原理图生成器
基于 SKiDL + YAML电路定义 -> 网表 + 原理图
"""
import os
import yaml
from pathlib import Path

from .circuit_helpers import init_skidl, build_circuit_from_yaml, create_nets_from_yaml


def generate_from_yaml(yaml_path: str, output_dir: str) -> dict:
    with open(yaml_path, "r", encoding="utf-8") as f:
        circuit = yaml.safe_load(f)

    custom_lib = init_skidl()

    print(f"[1/3] 解析电路: {circuit['name']}")
    print(f"       类型: {circuit['type']} | "
          f"板尺寸: {circuit.get('board', {}).get('width_mm', '?')}x"
          f"{circuit.get('board', {}).get('height_mm', '?')}mm")
    print(f"       元件数: {len(circuit['parts'])}, 网络数: {len(circuit['nets'])}")

    # 创建元件
    print("\n[2/3] 创建元件...")
    parts = build_circuit_from_yaml(circuit, custom_lib)
    for pid, p in parts.items():
        print(f"       {pid}: {p.value} ({p.footprint})")

    # 创建网络
    print("\n[3/3] 创建网络连接...")
    nets = create_nets_from_yaml(circuit, parts)

    # 直接生成网表（最可靠的输出）
    sch_name = circuit.get("name", "schematic").replace(" ", "_")
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    import skidl
    netlist_file = output_path / f"{sch_name}.net"
    print(f"\n生成网表: {netlist_file}")
    skidl.generate_netlist(filepath=str(netlist_file))

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
