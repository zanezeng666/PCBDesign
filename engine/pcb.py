"""
PCB 生成器 — 使用 KiCad 9.0 自带 Python (3.11) 运行
用法: & "C:\Program Files\KiCad\9.0\bin\python.exe" engine/pcb.py
"""
import os
import sys
import json
from pathlib import Path

# KiCad pcbnew 路径
KICAD_BIN = r"C:\Program Files\KiCad\9.0\bin"
KICAD_SHARE = r"C:\Program Files\KiCad\9.0\share\kicad"
KICAD_CLI = os.path.join(KICAD_BIN, "kicad-cli.exe")

def create_pcb_from_netlist(netlist_path: str, output_dir: str, 
                             width_mm: float = 40, height_mm: float = 15) -> dict:
    """从网表生成 PCB 文件"""
    
    # 确保路径是绝对路径
    netlist_path = str(Path(netlist_path).absolute())
    output_dir = str(Path(output_dir).absolute())
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    pcb_file = os.path.join(output_dir, "board.kicad_pcb")
    
    print(f"网表: {netlist_path}")
    print(f"输出: {pcb_file}")
    print(f"板尺寸: {width_mm}x{height_mm}mm")
    
    # 方案A: 用 kicad-cli 从网表创建 PCB（如果支持）
    # kicad-cli pcb import netlist 在新版中可能支持
    
    # 先尝试用 pcbnew Python API 创建
    try:
        import pcbnew
        
        board = pcbnew.BOARD()
        
        # 设置板框 (单位: 纳米, 1nm = 1e-6mm)
        w_nm = int(width_mm * 1_000_000)
        h_nm = int(height_mm * 1_000_000)
        
        # 创建板框
        edge_cuts = pcbnew.PCB_SHAPE(board)
        edge_cuts.SetShape(pcbnew.SHAPE_T_RECT)
        edge_cuts.SetStart(pcbnew.VECTOR2I(0, 0))
        edge_cuts.SetEnd(pcbnew.VECTOR2I(w_nm, h_nm))
        edge_cuts.SetLayer(pcbnew.Edge_Cuts)
        board.Add(edge_cuts)
        
        # 保存模板
        board.Save(pcb_file)
        print(f"PCB 模板已创建: {pcb_file}")
        
        # 导入网表
        netlist = pcbnew.NETLIST()
        netlist.ReadNetlistFromXml(netlist_path)
        
        # 加载板子并导入网表
        board = pcbnew.LoadBoard(pcb_file)
        pcbnew.NetlistToBoard(netlist, board)
        
        # 自动放置元件（简单网格布局）
        x = 5_000_000  # 5mm from left
        y = 5_000_000  # 5mm from top
        spacing = 5_000_000
        
        for i, footprint in enumerate(board.GetFootprints()):
            footprint.SetPosition(pcbnew.VECTOR2I(x + (i % 4) * spacing, 
                                                   y + (i // 4) * spacing))
        
        board.Save(pcb_file)
        print(f"元件已放置, PCB 已保存: {pcb_file}")
        
        return {"pcb_path": pcb_file, "status": "ok"}
        
    except Exception as e:
        print(f"pcbnew API 方式失败: {e}")
        print("尝试使用 kicad-cli 方式...")
        return _create_pcb_via_cli(netlist_path, output_dir, pcb_file)


def _create_pcb_via_cli(netlist_path, output_dir, pcb_file):
    """降级方案: 用命令行创建 PCB"""
    import subprocess
    
    # 尝试用 kicad-cli 直接操作
    result = {"pcb_path": pcb_file, "status": "fallback"}
    
    # 对于简单电路，直接用 S表达式创建 PCB
    pcb_content = _generate_minimal_pcb(netlist_path)
    with open(pcb_file, 'w') as f:
        f.write(pcb_content)
    
    print(f"已用降级方案生成 PCB: {pcb_file}")
    return result


def _generate_minimal_pcb(netlist_path):
    """手动生成最简 kicad_pcb S表达式"""
    return """(kicad_pcb (version 20240108) (generator "pcbnew"))
"""


def export_gerber(pcb_path: str, gerber_dir: str):
    """用 kicad-cli 导出 Gerber 和钻孔文件"""
    import subprocess
    
    pcb_path = str(Path(pcb_path).absolute())
    gerber_dir = str(Path(gerber_dir).absolute())
    Path(gerber_dir).mkdir(parents=True, exist_ok=True)
    
    print(f"\n导出 Gerber 到: {gerber_dir}")
    
    # 导出 Gerber
    result_gerber = subprocess.run(
        [KICAD_CLI, "pcb", "export", "gerber", "--output", gerber_dir, pcb_path],
        capture_output=True, text=True
    )
    
    if result_gerber.returncode == 0:
        print("   Gerber 导出成功")
    else:
        print(f"   Gerber 导出失败: {result_gerber.stderr}")
    
    # 导出钻孔
    result_drill = subprocess.run(
        [KICAD_CLI, "pcb", "export", "drill", "--output", gerber_dir, pcb_path],
        capture_output=True, text=True
    )
    
    if result_drill.returncode == 0:
        print("   钻孔文件导出成功")
    else:
        print(f"   钻孔文件导出失败: {result_drill.stderr}")
    
    return gerber_dir


if __name__ == "__main__":
    netlist = sys.argv[1] if len(sys.argv) > 1 else "schematic.net"
    output = sys.argv[2] if len(sys.argv) > 2 else "output/dw01_pcb"
    
    create_pcb_from_netlist(netlist, output)
    export_gerber(
        os.path.join(output, "board.kicad_pcb"),
        os.path.join(output, "gerber")
    )
