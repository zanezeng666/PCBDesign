"""保护板模板几何适配器（用 KiCad 自带 python 运行，含 pcbnew）。

用法:
  E:\\KiCad\\bin\\python.exe adapt.py <design-input.json> <pcb.kicad_pcb>

核心逻辑：
  1. 用 spec.outline 重塑板框（Edge.Cuts）
  2. 用 spec.terminals 的检测焊盘位置放置外部端子（J1 引脚映射到实物焊盘坐标）
     ★ 根据 side 字段自动分配到 F.Cu / B.Cu
  3. 内部元件（U1/Q1/R1/C1/C2）按模板相对关系定位在板内
  4. 重新布线 + B- zone 保证 DRC 0 未连接
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pcbnew

# ── 导入公共模块（端子 side 分配、zone 层选择等）──
_common_dir = str(Path(__file__).resolve().parent.parent)
if _common_dir not in sys.path:
    sys.path.insert(0, _common_dir)
from adapt_common import (  # noqa: E402
    _mm, classify_terminal, place_terminal_footprints,
    determine_zone_layer, rebuild_b_zone, fill_zone_and_fix_stubs,
    avoid_terminal_overlap, space_internal_references,
    compute_pad_rotation, create_custom_pad,
    _get_terminal_polygon_mm,
    place_holes, place_internal_components, route_nets_manhattan, center_board,
    run_freerouting_autorouter, connect_bminus_pads_to_zone,
    optimize_component_placement,
)


def _point_in_polygon(px: float, py: float, polygon: list[tuple[float, float]]) -> bool:
    """射线法判断点是否在多边形内。"""
    n = len(polygon)
    inside = False
    j = n - 1
    for i in range(n):
        xi, yi = polygon[i]
        xj, yj = polygon[j]
        if ((yi > py) != (yj > py)) and (px < (xj - xi) * (py - yi) / (yj - yi) + xi):
            inside = not inside
        j = i
    return inside


def _clamp_footprints_to_outline(board, outline_pts: list[tuple[float, float]], margin: float = 1.0) -> int:
    """确保所有内部元件（非端子）在 PCB 轮廓内。

    如果元件中心在轮廓外，将其推到轮廓内最近的位置。
    使用向内收缩 margin 后的轮廓作为安全边界。

    Args:
        board: pcbnew.BOARD 实例
        outline_pts: PCB 轮廓顶点 [(x_mm, y_mm), ...]
        margin: 安全边距 mm

    Returns:
        被移动的元件数量
    """
    if len(outline_pts) < 3:
        return 0

    # 计算轮廓中心
    xs = [p[0] for p in outline_pts]
    ys = [p[1] for p in outline_pts]
    cx = sum(xs) / len(xs)
    cy = sum(ys) / len(ys)

    # 向内收缩轮廓（用于碰撞检测）
    shrunk = []
    for px, py in outline_pts:
        dx = px - cx
        dy = py - cy
        dist = (dx * dx + dy * dy) ** 0.5
        if dist > margin:
            factor = (dist - margin) / dist
            shrunk.append((cx + dx * factor, cy + dy * factor))
        else:
            shrunk.append((px, py))

    moved = 0
    for fp in board.GetFootprints():
        if fp.GetReference().startswith("TP"):
            continue  # 端子焊盘不处理
        p = fp.GetPosition()
        px, py = pcbnew.ToMM(p.x), pcbnew.ToMM(p.y)

        if _point_in_polygon(px, py, shrunk):
            continue  # 已在轮廓内

        # 找到轮廓上最近的点，将元件移到该位置
        best_dist = float('inf')
        best_x, best_y = cx, cy  # fallback to center
        for sx, sy in shrunk:
            d = (px - sx) ** 2 + (py - sy) ** 2
            if d < best_dist:
                best_dist = d
                best_x, best_y = sx, sy

        fp.SetPosition(pcbnew.VECTOR2I(_mm(best_x), _mm(best_y)))
        moved += 1

    if moved:
        print(f"[adapt] 将 {moved} 个超出轮廓的元件拉回板内")
    return moved


def main(spec_path: str, pcb_path: str) -> int:
    spec = json.loads(Path(spec_path).read_text(encoding="utf-8-sig"))
    board = pcbnew.LoadBoard(pcb_path)

    outline = spec.get("outline", {}).get("points", [])
    if len(outline) < 3:
        print("[adapt] spec 无有效板框轮廓；保持原板不变")
        return 0

    pts = [(float(p["x_mm"]), float(p["y_mm"])) for p in outline]
    spec_xs = [p[0] for p in pts]
    spec_ys = [p[1] for p in pts]
    spec_minx, spec_maxx = min(spec_xs), max(spec_xs)
    spec_miny, spec_maxy = min(spec_ys), max(spec_ys)
    spec_w = spec_maxx - spec_minx
    spec_h = spec_maxy - spec_miny
    spec_cx = (spec_minx + spec_maxx) / 2
    spec_cy = (spec_miny + spec_maxy) / 2

    # ── 解析检测到的焊盘位置 ──
    terminals = spec.get("terminals", [])
    print(f"[adapt] 检测到 {len(terminals)} 个端子")

    # ── 1) 重塑板框 Edge.Cuts ──
    old_edges = [g for g in board.GetDrawings() if g.GetLayer() == pcbnew.Edge_Cuts]
    for g in old_edges:
        board.Remove(g)
    for i in range(len(pts)):
        x1, y1 = pts[i]
        x2, y2 = pts[(i + 1) % len(pts)]
        seg = pcbnew.PCB_SHAPE(board)
        seg.SetShape(pcbnew.SHAPE_T_SEGMENT)
        seg.SetStart(pcbnew.VECTOR2I(_mm(x1), _mm(y1)))
        seg.SetEnd(pcbnew.VECTOR2I(_mm(x2), _mm(y2)))
        seg.SetLayer(pcbnew.Edge_Cuts)
        seg.SetWidth(_mm(0.1))
        board.Add(seg)

    # ── 2) 删除旧 J1（连接器），用检测位置放置独立端子焊盘 ──
    j1_fp = None
    for fp in list(board.GetFootprints()):
        if fp.GetReference() == "J1":
            j1_fp = fp
            board.Remove(fp)
            break

    # 加载测试点封装（KiCad 标准库）
    fp_dir = Path(os.environ.get("KICAD_FOOTPRINT_DIR", r"E:\KiCad\share\kicad\footprints"))
    tp_lib = str(fp_dir / "TestPoint.pretty")
    tp_name = "TestPoint_Pad_D1.5mm"
    tp_alt_lib = str(fp_dir / "Connector_PinHeader_2.54mm.pretty")
    tp_alt_name = "PinHeader_1x01_P2.54mm_Vertical"

    def load_tp_footprint():
        fp = pcbnew.FootprintLoad(tp_lib, tp_name)
        if fp is None:
            fp = pcbnew.FootprintLoad(tp_alt_lib, tp_alt_name)
        if fp is None:
            fp = pcbnew.FootprintLoad(tp_lib, "TestPoint_THTPad_D1.0mm_Drill0.5mm")
        return fp

    # ★ 使用公共模块放置端子（自动处理 side → 铜层分配）
    term_info = place_terminal_footprints(board, terminals, load_tp_footprint)

    # ── 3) 放置孔槽（预留，当前测试板无孔槽）──
    holes = spec.get("holes", [])
    place_holes(board, holes)

    # ── 4) 内部元件布局：行排列+碰撞检测（替代原缩放逻辑）──
    place_internal_components(board, pts, terminals, holes, min_gap=0.5)

    # ★ 连接感知布局优化（模拟退火，最小化走线长度）
    optimize_component_placement(board, pts)

    # ★ 确保所有内部元件在 PCB 轮廓内
    _clamp_footprints_to_outline(board, pts, margin=1.0)
    # ★ 缩小丝印文本并放到元件周边空闲位置（约束板内，用最终元件位置）
    space_internal_references(board)

    # ── 4) 删除旧走线/过孔/zone ──
    old_tracks = list(board.GetTracks())
    for t in old_tracks:
        board.Remove(t)
    # ★ 删除所有旧 zone（Freerouting 需要干净的双面布线环境）
    for z in list(board.Zones()):
        board.Remove(z)

    # ── 5) 自动布线：优先 Freerouting 双面布线，失败时回退 zone 方案 ──
    zone_layer = determine_zone_layer(board, terminals)
    outline_dicts = [{"x_mm": p[0], "y_mm": p[1]} for p in pts]

    if run_freerouting_autorouter(board):
        # ★ Freerouting 成功：所有网络（含 B-）已由云端双面布线
        # 正反面通过过孔连接，无需 B- zone 后处理
        print("[adapt] Freerouting 双面布线完成")
        routed = -1
    else:
        # ★ Freerouting 不可用/失败：回退到 zone 方案
        print("[adapt] Freerouting 不可用，回退 zone 方案（B.Cu 地平面 + F.Cu 信号）")
        rebuild_b_zone(board, (spec_minx, spec_miny, spec_maxx, spec_maxy), zone_layer,
                       outline_points=outline_dicts)
        routed = route_nets_manhattan(board, zone_layer)
        connect_bminus_pads_to_zone(board, zone_layer)
        fill_zone_and_fix_stubs(board, zone_layer, spec_cx, spec_cy)

    # ── 8) 板框居中（使 PCB 在 KiCad 画布中央显示）──
    center_board(board)

    board.Save(pcb_path)
    print(f"[adapt] 完成: 板框{len(pts)}点, {len(terminals)}端子@检测位, {routed}网络布线")
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("usage: adapt.py <design-input.json> <pcb.kicad_pcb>", file=sys.stderr)
        sys.exit(2)
    sys.exit(main(sys.argv[1], sys.argv[2]))
