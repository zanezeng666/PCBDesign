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
)


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

    # ── 3) 内部元件定位：缩放到板框内（排除 J1） ──
    fps = [fp for fp in board.GetFootprints() if not fp.GetReference().startswith("TP")]
    if fps:
        fp_positions = [(pcbnew.ToMM(fp.GetPosition().x), pcbnew.ToMM(fp.GetPosition().y)) for fp in fps]
        fp_xs = [p[0] for p in fp_positions]
        fp_ys = [p[1] for p in fp_positions]
        fp_minx, fp_maxx = min(fp_xs), max(fp_xs)
        fp_miny, fp_maxy = min(fp_ys), max(fp_ys)
        fp_w = fp_maxx - fp_minx
        fp_h = fp_maxy - fp_miny

        inset = 2.0
        target_w = max(spec_w - 2 * inset, fp_w * 0.3)
        target_h = max(spec_h - 2 * inset, fp_h * 0.3)
        sx = target_w / fp_w if fp_w > 0.01 else 1.0
        sy = target_h / fp_h if fp_h > 0.01 else 1.0
        scale = min(sx, sy, 2.0)

        fp_cx = (fp_minx + fp_maxx) / 2
        fp_cy = (fp_miny + fp_maxy) / 2

        for fp in fps:
            p = fp.GetPosition()
            ox, oy = pcbnew.ToMM(p.x), pcbnew.ToMM(p.y)
            nx = spec_cx + (ox - fp_cx) * scale
            ny = spec_cy + (oy - fp_cy) * scale
            fp.SetPosition(pcbnew.VECTOR2I(_mm(nx), _mm(ny)))

        print(f"[adapt] 内部元件变换: scale={scale:.3f}")

    # ── 4) 删除旧走线/过孔 ──
    old_tracks = list(board.GetTracks())
    for t in old_tracks:
        board.Remove(t)

    # ── 5) 重建 B- zone（★ 使用公共模块根据 side 选择铜层）──
    zone_layer = determine_zone_layer(board, terminals)
    rebuild_b_zone(board, (spec_minx, spec_miny, spec_maxx, spec_maxy), zone_layer)

    # ── 6) 重新布线 ──
    bcu = pcbnew.B_Cu

    def pad_pos(pad):
        p = pad.GetPosition()
        return (pcbnew.ToMM(p.x), pcbnew.ToMM(p.y))

    def add_track(net_code, ax, ay, bx, by, layer, w):
        t = pcbnew.PCB_TRACK(board)
        t.SetStart(pcbnew.VECTOR2I(_mm(ax), _mm(ay)))
        t.SetEnd(pcbnew.VECTOR2I(_mm(bx), _mm(by)))
        t.SetWidth(_mm(w))
        t.SetLayer(layer)
        t.SetNetCode(net_code)
        board.Add(t)

    def add_via(net_code, x, y):
        via = pcbnew.PCB_VIA(board)
        via.SetPosition(pcbnew.VECTOR2I(_mm(x), _mm(y)))
        via.SetWidth(_mm(0.8))
        via.SetDrill(_mm(0.4))
        via.SetViaType(pcbnew.VIATYPE_THROUGH)
        board.Add(via)
        via.SetNetCode(net_code)

    # 收集每个网络的所有焊盘
    net_pads: dict[int, list] = {}
    for fp in board.GetFootprints():
        for pad in fp.Pads():
            nc = pad.GetNetCode()
            if nc == 0:
                continue
            x, y = pad_pos(pad)
            is_smd = pad.GetAttribute() == pcbnew.PAD_ATTRIB_SMD
            net_pads.setdefault(nc, []).append((x, y, is_smd))

    routed = 0
    for nc, pads in net_pads.items():
        net_obj = board.FindNet(nc)
        net_name = net_obj.GetNetname() if net_obj else ""
        width = 0.4 if net_name in ("B+", "B-", "P-") else 0.25
        for (x, y, is_smd) in pads:
            if is_smd:
                add_via(nc, x, y)
        for i in range(len(pads) - 1):
            add_track(nc, pads[i][0], pads[i][1], pads[i + 1][0], pads[i + 1][1], bcu, width)
        routed += 1
    print(f"[adapt] 重新布线 {routed} 个网络")

    # ── 7) 填充 zone + 修复未覆盖焊盘（★ 使用公共模块）──
    fill_zone_and_fix_stubs(board, zone_layer, spec_cx, spec_cy)

    board.Save(pcb_path)
    print(f"[adapt] 完成: 板框{len(pts)}点, {len(terminals)}端子@检测位, {routed}网络布线")
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("usage: adapt.py <design-input.json> <pcb.kicad_pcb>", file=sys.stderr)
        sys.exit(2)
    sys.exit(main(sys.argv[1], sys.argv[2]))
