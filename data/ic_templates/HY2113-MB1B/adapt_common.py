"""KiCad 模板适配器公共模块（用 KiCad 自带 python 运行，含 pcbnew）。

所有 adapt.py 都应 import 此模块，确保：
  1. 端子焊盘根据 spec 的 side 字段分配到正确铜层（F.Cu / B.Cu）
  2. B- zone 根据 B- 端子所在面选择铜层
  3. zone filler 和 stub track 使用一致的层

新模板通过 build_kicad_template.py 自动复制此文件。
"""
from __future__ import annotations

import pcbnew


def _mm(v: float) -> int:
    return pcbnew.FromMM(v)


def classify_terminal(t: dict) -> str | None:
    """将 spec terminal 映射到网络名（1S 共口拓扑）。"""
    roles = set(t.get("roles", []))
    pol = t.get("polarity", "")
    if "battery" in roles:
        return "B+" if pol == "positive" else "B-"
    if roles & {"charge", "discharge"}:
        return "B+" if pol == "positive" else "P-"
    if "temperature" in roles:
        return "TH"
    if "identification" in roles:
        return "ID"
    return None


def place_terminal_footprints(
    board,
    terminals: list[dict],
    load_fp_fn,
    tp_name_prefix: str = "TP",
) -> dict[str, list[dict]]:
    """按 spec terminals 放置端子焊盘封装，根据 side 自动 Flip 到背面。

    Args:
        board: pcbnew.BOARD 实例
        terminals: spec["terminals"] 列表
        load_fp_fn: 无参 callable，返回一个 pcbnew.FOOTPRINT 或 None
        tp_name_prefix: 端子 reference 前缀

    Returns:
        term_info: net_name -> [terminal_dict, ...] 映射（供后续布线使用）
    """
    term_info: dict[str, list[dict]] = {}
    for t in terminals:
        net = classify_terminal(t)
        if net is None:
            continue
        term_info.setdefault(net, []).append(t)

    tp_count = 0
    for net_name, t_list in term_info.items():
        net_info = board.FindNet(net_name)
        if net_info is None:
            continue
        for idx, t in enumerate(t_list):
            pos = t.get("position", {})
            tx, ty = float(pos.get("x_mm", 0)), float(pos.get("y_mm", 0))
            w_mm = max(float(t.get("width_mm", 2.0)), 1.5)
            h_mm = max(float(t.get("height_mm", 2.0)), 1.5)

            fp = load_fp_fn()
            if fp is None:
                print(f"[adapt_common] WARNING: 无法加载测试点封装，跳过 {net_name}[{idx}]")
                continue
            board.Add(fp)
            fp.SetPosition(pcbnew.VECTOR2I(_mm(tx), _mm(ty)))
            fp.SetReference(f"{tp_name_prefix}{tp_count + 1}")
            fp.SetValue(net_name)

            # ★ 核心：根据焊盘识别的 side 分配到对应铜层
            if t.get("side") == "back":
                fp.Flip(fp.GetPosition(), False)

            for pad in fp.Pads():
                pad.SetNetCode(net_info.GetNetCode())
                pad.SetSize(pcbnew.VECTOR2I(_mm(w_mm), _mm(h_mm)))
            tp_count += 1

    print(f"[adapt_common] 放置 {tp_count} 个端子焊盘（含 side 层分配）")
    return term_info


def determine_zone_layer(board, terminals: list[dict]) -> int:
    """根据 B- 端子所在面决定 B- zone 应放的铜层。

    如果 B- 端子在背面 → B.Cu；否则 → F.Cu。
    """
    b_terminals_back = any(
        t.get("side") == "back"
        for t in terminals
        if classify_terminal(t) == "B-"
    )
    return pcbnew.B_Cu if b_terminals_back else pcbnew.F_Cu


def rebuild_b_zone(board, spec_bounds: tuple[float, float, float, float],
                   zone_layer: int, margin: float = 0.5) -> None:
    """删除旧 B- zone 并重建覆盖整个板框的新 zone。

    Args:
        spec_bounds: (min_x, min_y, max_x, max_y) mm
        zone_layer: pcbnew.F_Cu 或 pcbnew.B_Cu
        margin: zone 超出板框的余量 mm
    """
    spec_minx, spec_miny, spec_maxx, spec_maxy = spec_bounds
    b_net = board.FindNet("B-")
    if b_net is None:
        return

    # 删除旧 zone
    for z in list(board.Zones()):
        if z.GetNetCode() == b_net.GetNetCode():
            board.Remove(z)

    zone = pcbnew.ZONE(board)
    zone.SetLayer(zone_layer)
    zone.SetMinThickness(_mm(0.2))
    zone.SetThermalReliefGap(_mm(0.2))
    zone.SetThermalReliefSpokeWidth(_mm(0.25))
    chain = pcbnew.SHAPE_LINE_CHAIN()
    chain.Append(_mm(spec_minx - margin), _mm(spec_miny - margin))
    chain.Append(_mm(spec_maxx + margin), _mm(spec_miny - margin))
    chain.Append(_mm(spec_maxx + margin), _mm(spec_maxy + margin))
    chain.Append(_mm(spec_minx - margin), _mm(spec_maxy + margin))
    chain.SetClosed(True)
    zone.AddPolygon(chain)
    board.Add(zone)
    zone.SetNet(b_net)


def fill_zone_and_fix_stubs(board, zone_layer: int,
                            spec_cx: float, spec_cy: float) -> None:
    """填充 zone 并为未被 zone 覆盖的 B- 焊盘添加连接短桩。"""
    b_net = board.FindNet("B-")
    if b_net is None:
        return

    filler = pcbnew.ZONE_FILLER(board)
    filler.Fill(list(board.Zones()))
    board.BuildConnectivity()

    # 查找 B- zone 对象
    zone_obj = None
    for z in board.Zones():
        if z.GetNetCode() == b_net.GetNetCode():
            zone_obj = z
            break
    if zone_obj is None:
        return

    stubs = 0
    for fp in board.GetFootprints():
        for pad in fp.Pads():
            if pad.GetNetCode() != b_net.GetNetCode():
                continue
            if zone_obj.HitTestFilledArea(zone_layer, pad.GetPosition()):
                continue
            p = pad.GetPosition()
            px, py = pcbnew.ToMM(p.x), pcbnew.ToMM(p.y)
            dx, dy = spec_cx - px, spec_cy - py
            length = (dx * dx + dy * dy) ** 0.5
            if length < 0.01:
                dx, dy, length = -1.0, 0.0, 1.0
            ext = 3.0
            t = pcbnew.PCB_TRACK(board)
            t.SetStart(pcbnew.VECTOR2I(_mm(px), _mm(py)))
            t.SetEnd(pcbnew.VECTOR2I(_mm(px + dx / length * ext),
                                      _mm(py + dy / length * ext)))
            t.SetWidth(_mm(0.4))
            t.SetLayer(zone_layer)
            t.SetNetCode(b_net.GetNetCode())
            board.Add(t)
            stubs += 1

    if stubs:
        filler.Fill(list(board.Zones()))
        board.BuildConnectivity()
        print(f"[adapt_common] 补充 {stubs} 个 B- 短桩")
