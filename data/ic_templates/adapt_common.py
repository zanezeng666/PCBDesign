"""KiCad 模板适配器公共模块（用 KiCad 自带 python 运行，含 pcbnew）。

所有 adapt.py 都应 import 此模块，确保：
  1. 端子焊盘根据 spec 的 side 字段分配到正确铜层（F.Cu / B.Cu）
  2. B- zone 根据 B- 端子所在面选择铜层
  3. zone filler 和 stub track 使用一致的层
  4. ★ 焊盘形状/旋转/自定义形状 根据 VLM 检测的 polygon 精确还原
  5. ★ B- zone 使用实际 PCB 轮廓多边形而非简单 bbox

新模板通过 build_kicad_template.py 自动复制此文件。
"""
from __future__ import annotations

import math
import os
import subprocess
import tempfile
from pathlib import Path

import pcbnew

# ── 模块级缓存：B- netcode（SWIG 对象在 zone 操作后可能失效）──
_cached_b_netcode: int = -1

# ── Freerouting Cloud API 配置（可用环境变量覆盖）──
# ★ 已迁移到 Freerouting Cloud API（本地 JAR 方式废弃）
_FREEROUTING_API_KEY = os.environ.get(
    "FREEROUTING_API_KEY", "201be9f3-e8eb-4395-84b9-bdc36531690f")
_FREEROUTING_PROFILE_ID = os.environ.get(
    "FREEROUTING_PROFILE_ID", "4e75b344-64f7-48ee-89ce-a0e085df80dd")
_FREEROUTING_HOST = os.environ.get(
    "FREEROUTING_HOST", "KiCad/9.0")
_FREEROUTING_BASE_URL = os.environ.get(
    "FREEROUTING_BASE_URL", "https://api.freerouting.app/v1")


def _mm(v: float) -> int:
    return pcbnew.FromMM(v)


def _apply_pad_shape(pad, shape: str, w_mm: float, h_mm: float):
    """根据 VLM 检测结果设置焊盘真实形状。"""
    if shape == "circle":
        pad.SetShape(pcbnew.PAD_SHAPE_CIRCLE)
    elif shape == "oval":
        pad.SetShape(pcbnew.PAD_SHAPE_OVAL)
    elif shape == "rounded_rect":
        pad.SetShape(pcbnew.PAD_SHAPE_ROUNDRECT)
        pad.SetRoundRectRadiusRatio(0.25)
    elif shape == "rect":
        pad.SetShape(pcbnew.PAD_SHAPE_RECT)
    else:
        # custom / 未知 → 按宽高比推断
        aspect = max(w_mm, h_mm) / max(min(w_mm, h_mm), 0.01)
        if aspect < 1.2 and abs(w_mm - h_mm) < 0.3:
            pad.SetShape(pcbnew.PAD_SHAPE_CIRCLE)
        elif aspect < 3.0:
            pad.SetShape(pcbnew.PAD_SHAPE_ROUNDRECT)
            pad.SetRoundRectRadiusRatio(0.25)
        else:
            pad.SetShape(pcbnew.PAD_SHAPE_OVAL)


def compute_pad_rotation(polygon: list[dict]) -> float:
    """从 VLM 检测的多边形顶点计算焊盘旋转角度（度）。

    使用 PCA（主成分分析）思路：取多边形最长边方向作为焊盘朝向。
    对于 4 顶点矩形，取长边方向；对于更多顶点，用协方差矩阵主方向。
    返回角度范围 -90° ~ +90°。
    """
    if not polygon or len(polygon) < 3:
        return 0.0

    xs = [float(p.get("x_mm", 0)) for p in polygon]
    ys = [float(p.get("y_mm", 0)) for p in polygon]
    n = len(xs)

    # 计算协方差矩阵
    cx = sum(xs) / n
    cy = sum(ys) / n
    cov_xx = sum((x - cx) ** 2 for x in xs) / n
    cov_yy = sum((y - cy) ** 2 for y in ys) / n
    cov_xy = sum((x - cx) * (y - cy) for x, y in zip(xs, ys)) / n

    # 主方向角度 = 0.5 * atan2(2*cov_xy, cov_xx - cov_yy)
    angle_rad = 0.5 * math.atan2(2 * cov_xy, cov_xx - cov_yy)
    angle_deg = math.degrees(angle_rad)

    # 规范化到 -90° ~ +90°
    while angle_deg > 90:
        angle_deg -= 180
    while angle_deg < -90:
        angle_deg += 180

    # 小角度（<1°）视为无旋转，避免噪声
    if abs(angle_deg) < 1.0:
        return 0.0

    return round(angle_deg, 2)


def _get_terminal_polygon_mm(terminal: dict) -> list[dict] | None:
    """从 terminal 的 source_region 提取多边形顶点（mm）。"""
    sr = terminal.get("source_region") or {}
    poly = sr.get("polygon") or []
    if len(poly) >= 3:
        return poly
    # fallback: matched_regions
    matched = terminal.get("matched_regions") or []
    if matched and len(matched[0].get("polygon", [])) >= 3:
        return matched[0]["polygon"]
    return None


def create_custom_pad(
    board,
    pad,
    polygon_mm: list[dict],
    rotation_deg: float,
) -> None:
    """将焊盘设置为自定义形状（Custom shape），用检测到的多边形顶点定义。

    KiCad 10 的 PAD_SHAPE_CUSTOM 通过 AddPrimitivePoly 添加多边形图元定义形状。
    多边形坐标相对于焊盘中心，已考虑旋转。
    """
    if not polygon_mm or len(polygon_mm) < 3:
        return

    # 计算多边形中心（用于归一化坐标）
    xs = [float(p.get("x_mm", 0)) for p in polygon_mm]
    ys = [float(p.get("y_mm", 0)) for p in polygon_mm]
    cx = sum(xs) / len(xs)
    cy = sum(ys) / len(ys)

    # 将多边形顶点转为相对于中心的坐标（纳米），并反向旋转以对齐焊盘朝向
    angle_rad = math.radians(-rotation_deg)
    cos_a = math.cos(angle_rad)
    sin_a = math.sin(angle_rad)

    # 构建顶点列表 (VECTOR_VECTOR2I)
    poly_pts = pcbnew.VECTOR_VECTOR2I()
    for p in polygon_mm:
        dx = float(p.get("x_mm", 0)) - cx
        dy = float(p.get("y_mm", 0)) - cy
        # 反向旋转，使焊盘本体对齐到 0°（KiCad 自定义形状是相对于 0° 焊盘的）
        rx = dx * cos_a - dy * sin_a
        ry = dx * sin_a + dy * cos_a
        poly_pts.append(pcbnew.VECTOR2I(_mm(rx), _mm(ry)))

    # 设置焊盘尺寸为多边形的包围盒（用于 DRC 间距计算）
    w_mm = max(xs) - min(xs)
    h_mm = max(ys) - min(ys)
    pad.SetSize(pcbnew.VECTOR2I(_mm(max(w_mm, 0.5)), _mm(max(h_mm, 0.5))))

    # 设置为自定义形状并添加多边形图元
    pad.SetShape(pcbnew.PAD_SHAPE_CUSTOM)
    # 获取焊盘所在层（F.Cu 或 B.Cu）
    pad_layer = pad.GetLayer()
    # 添加填充多边形（thickness=0 表示填充）
    pad.AddPrimitivePoly(pad_layer, poly_pts, 0, True)


def estimate_corner_radius_ratio(polygon_mm: list[dict], w_mm: float, h_mm: float) -> float:
    """从 VLM 检测多边形估算圆角矩形的圆角半径比例（相对短边，0.1~0.5）。

    - 4 顶点（外接四边形，圆角不可见）：返回默认比例 0.25
    - >4 顶点（含圆角过渡点）：从包围盒角点到最近过渡点的距离反推半径
      （弧上离角点最近的点距离 = r*(sqrt2-1)）
    同时限制绝对半径不超过 0.8mm，避免大焊盘过度圆角化。
    """
    default_ratio = 0.25
    max_radius_mm = 0.8
    short_side = max(min(w_mm, h_mm), 0.01)

    if not polygon_mm or len(polygon_mm) <= 4:
        ratio = default_ratio
    else:
        xs = [float(p.get("x_mm", 0)) for p in polygon_mm]
        ys = [float(p.get("y_mm", 0)) for p in polygon_mm]
        minx, maxx, miny, maxy = min(xs), max(xs), min(ys), max(ys)
        corners = [(minx, miny), (maxx, miny), (maxx, maxy), (minx, maxy)]
        offsets = []
        for cx, cy in corners:
            best = min(
                math.hypot(float(p.get("x_mm", 0)) - cx, float(p.get("y_mm", 0)) - cy)
                for p in polygon_mm
            )
            offsets.append(best)
        avg_offset = sum(offsets) / len(offsets) if offsets else 0.0
        r_mm = avg_offset / (math.sqrt(2) - 1) if avg_offset > 0 else default_ratio * short_side
        ratio = r_mm / short_side

    # 限制绝对半径，避免大焊盘圆角过大
    ratio = min(ratio, max_radius_mm / short_side)
    return max(0.1, min(ratio, 0.5))


def _create_rounded_rect_pad(pad, polygon_mm: list[dict], rotation_deg: float) -> None:
    """将焊盘设置为圆角矩形（PAD_SHAPE_ROUNDRECT），还原 VLM 识别的圆角矩形参数。

    - 尺寸：多边形在主方向（PCA 旋转角）上的投影范围，保证旋转后与检测包围盒一致
    - 圆角半径：从多边形顶点估算（estimate_corner_radius_ratio）
    - 朝向：由封装旋转（rotation_deg）控制，无需反向旋转坐标
    """
    xs = [float(p.get("x_mm", 0)) for p in polygon_mm]
    ys = [float(p.get("y_mm", 0)) for p in polygon_mm]
    cx = sum(xs) / len(xs)
    cy = sum(ys) / len(ys)

    # 主方向单位向量（与 compute_pad_rotation 一致）
    a = math.radians(rotation_deg)
    ux, uy = math.cos(a), math.sin(a)
    # 投影到主方向及垂直方向，得到焊盘局部尺寸
    proj_main = [(x - cx) * ux + (y - cy) * uy for x, y in zip(xs, ys)]
    proj_perp = [-(x - cx) * uy + (y - cy) * ux for x, y in zip(xs, ys)]
    w_local = max(max(proj_main) - min(proj_main), 0.5)
    h_local = max(max(proj_perp) - min(proj_perp), 0.5)

    pad.SetSize(pcbnew.VECTOR2I(_mm(w_local), _mm(h_local)))
    ratio = estimate_corner_radius_ratio(polygon_mm, w_local, h_local)
    pad.SetShape(pcbnew.PAD_SHAPE_ROUNDRECT)
    pad.SetRoundRectRadiusRatio(ratio)


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


def terminal_label(t: dict) -> str | None:
    """将 spec terminal 映射到视觉标签（可能与网络名不同）。"""
    roles = set(t.get("roles", []))
    pol = t.get("polarity", "")
    if "battery" in roles:
        return "B+" if pol == "positive" else "B-"
    if roles & {"charge", "discharge"}:
        return "P+" if pol == "positive" else "P-"
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

    ★ 增强：
      - 从 source_region.polygon 提取旋转角，旋转封装以匹配检测朝向
      - 从 source_region.polygon 创建自定义焊盘形状（PAD_SHAPE_CUSTOM）
      - 保留原始 bbox 尺寸用于 DRC

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
        # 同名网络多个焊盘时用后缀区分（如 B+_1, B+_2）
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

            # ★ Reference 统一用网络名 + 序号后缀（如 B+_1/B+_2、P-_1、TH_1）。
            #   始终加后缀保证所有端子命名风格一致；用网络名避免 P+/B+ 混用。
            ref = f"{net_name}_{idx + 1}"
            fp.SetReference(ref)
            fp.Reference().SetVisible(True)
            fp.SetValue(net_name)
            fp.Value().SetVisible(True)

            # ★ 核心：根据焊盘识别的 side 分配到对应铜层
            if t.get("side") == "back":
                fp.Flip(fp.GetPosition(), False)

            # ★ 从 VLM polygon 计算旋转角并旋转封装
            polygon_mm = _get_terminal_polygon_mm(t)
            rotation_deg = compute_pad_rotation(polygon_mm) if polygon_mm else 0.0
            if rotation_deg != 0.0:
                # KiCad 10: Rotate 需要 EDA_ANGLE 对象
                fp.Rotate(fp.GetPosition(), pcbnew.EDA_ANGLE(rotation_deg))
                print(f"[adapt_common] 旋转 {ref} {rotation_deg:.1f}° (来自 VLM polygon)")

            for pad in fp.Pads():
                pad.SetNetCode(net_info.GetNetCode())

                pad_shape = t.get("shape", "rect")
                if polygon_mm and len(polygon_mm) >= 3 and pad_shape in ("rect", "rounded_rect"):
                    # ★ 圆角矩形焊盘：用原生 ROUNDRECT 形状 + 从 polygon 估算的圆角半径
                    #   （还原 VLM 识别的圆角矩形参数，而非尖角自定义多边形）
                    _create_rounded_rect_pad(pad, polygon_mm, rotation_deg)
                elif polygon_mm and len(polygon_mm) >= 3:
                    # 其他不规则形状：自定义多边形精确还原
                    create_custom_pad(board, pad, polygon_mm, rotation_deg)
                else:
                    pad.SetSize(pcbnew.VECTOR2I(_mm(w_mm), _mm(h_mm)))
                    _apply_pad_shape(pad, pad_shape, w_mm, h_mm)
            tp_count += 1

    print(f"[adapt_common] 放置 {tp_count} 个端子焊盘（含 side 层分配 + 形状/旋转还原）")
    return term_info


def determine_zone_layer(board, terminals: list[dict]) -> int:
    """决定 B- 地平面 zone 所在铜层。

    ★ 策略：B- zone 作为完整地平面放在 B.Cu（背面），信号走线放在 F.Cu（元件面）。
    这样信号走线与地平面分层隔离，避免同层拥挤导致的短路/间距违规。
    B- 焊盘多为背面 SMD，可直接连接 B.Cu 地平面；正面 B- 焊盘通过过孔连接。
    """
    return pcbnew.B_Cu


def rebuild_b_zone(board, spec_bounds: tuple[float, float, float, float],
                   zone_layer: int, margin: float = 0.5,
                   outline_points: list[dict] | None = None) -> None:
    """删除旧 B- zone 并重建覆盖整个板框的新 zone。

    ★ 增强：当提供 outline_points 时，使用实际 PCB 轮廓多边形而非 bbox 矩形，
    使 zone 精确贴合板框形状（包括凹槽/凸起）。

    Args:
        spec_bounds: (min_x, min_y, max_x, max_y) mm
        zone_layer: pcbnew.F_Cu 或 pcbnew.B_Cu
        margin: zone 超出板框的余量 mm
        outline_points: PCB 轮廓顶点列表 [{"x_mm": ..., "y_mm": ...}, ...]
    """
    global _cached_b_netcode

    # ★ 优先使用缓存的 B- netcode（SES 导入后 FindNet 可能返回无效 SWIG 对象）
    b_nc = _cached_b_netcode
    b_net = None
    if b_nc < 0:
        b_net = board.FindNet("B-")
        if b_net is None:
            return
        try:
            b_nc = b_net.GetNetCode()
            _cached_b_netcode = b_nc
        except AttributeError:
            # ★ 从 tracks 中反查 B- netcode
            for t in board.GetTracks():
                try:
                    if t.GetNetCode() > 0:
                        # 检查网络名称
                        net_name = t.GetNetname() if hasattr(t, 'GetNetname') else ""
                        if net_name == "B-":
                            b_nc = t.GetNetCode()
                            _cached_b_netcode = b_nc
                            break
                except Exception:
                    continue
            if b_nc < 0:
                print("[adapt_common] WARNING: 无法找到 B- netcode，跳过 zone 重建")
                return
            print(f"[adapt_common] 从 tracks 反查到 B- netcode={b_nc}")

    # 尝试获取 net 对象（用于 zone.SetNet），失败时用 SetNetCode
    if b_net is None:
        try:
            b_net = board.FindNet("B-")
            _ = b_net.GetNetCode()  # 测试是否有效
        except (AttributeError, Exception):
            b_net = None

    # 删除旧 zone
    for z in list(board.Zones()):
        try:
            znc = z.GetNetCode()
        except AttributeError:
            continue
        if znc == b_nc:
            board.Remove(z)

    zone = pcbnew.ZONE(board)
    zone.SetLayer(zone_layer)
    zone.SetMinThickness(_mm(0.2))
    zone.SetThermalReliefGap(_mm(0.2))
    zone.SetThermalReliefSpokeWidth(_mm(0.25))
    # ★ 地平面与其他网络的安全间距：局部间距 0.5mm
    #   （避免大端子焊盘处出现 shorting_items）
    zone.SetLocalClearance(_mm(0.5))

    # ★ 提高板级默认网络间距，保证信号与地平面/电源间足够间距
    try:
        board.GetDesignSettings().GetDefault().SetClearance(_mm(0.25))
    except Exception:
        pass

    chain = pcbnew.SHAPE_LINE_CHAIN()

    if outline_points and len(outline_points) >= 3:
        # ★ 使用实际 PCB 轮廓多边形（带 margin 外扩）
        # 计算轮廓中心用于外扩
        xs = [float(p.get("x_mm", 0)) for p in outline_points]
        ys = [float(p.get("y_mm", 0)) for p in outline_points]
        cx = sum(xs) / len(xs)
        cy = sum(ys) / len(ys)

        for p in outline_points:
            px = float(p.get("x_mm", 0))
            py = float(p.get("y_mm", 0))
            # 从中心向外扩展 margin
            dx = px - cx
            dy = py - cy
            dist = math.sqrt(dx * dx + dy * dy)
            if dist > 0.01:
                ex = px + (dx / dist) * margin
                ey = py + (dy / dist) * margin
            else:
                ex = px + margin
                ey = py
            chain.Append(_mm(ex), _mm(ey))
        print(f"[adapt_common] B- zone: 使用实际轮廓多边形 ({len(outline_points)} 顶点)")
    else:
        # Fallback: bbox 矩形
        spec_minx, spec_miny, spec_maxx, spec_maxy = spec_bounds
        chain.Append(_mm(spec_minx - margin), _mm(spec_miny - margin))
        chain.Append(_mm(spec_maxx + margin), _mm(spec_miny - margin))
        chain.Append(_mm(spec_maxx + margin), _mm(spec_maxy + margin))
        chain.Append(_mm(spec_minx - margin), _mm(spec_maxy + margin))
        print("[adapt_common] B- zone: 使用 bbox 矩形 (无轮廓数据)")

    chain.SetClosed(True)
    zone.AddPolygon(chain)
    board.Add(zone)
    if b_net is not None:
        zone.SetNet(b_net)
    else:
        # ★ SWIG 对象无效时用 netcode 直接设置
        zone.SetNetCode(b_nc)


def fill_zone_and_fix_stubs(board, zone_layer: int,
                            spec_cx: float, spec_cy: float) -> None:
    """填充 zone 并为未被 zone 覆盖的 B- 焊盘添加连接短桩。"""
    global _cached_b_netcode

    # ★ 优先使用缓存的 B- netcode（SWIG 对象在 rebuild_b_zone 后可能失效）
    b_nc = _cached_b_netcode
    if b_nc < 0:
        # 回退：尝试 FindNet
        b_net = board.FindNet("B-")
        if b_net is None:
            return
        try:
            b_nc = b_net.GetNetCode()
            _cached_b_netcode = b_nc
        except AttributeError:
            print("[adapt_common] WARNING: FindNet('B-') 返回无效 SWIG 对象且无缓存，跳过 zone 填充")
            return

    filler = pcbnew.ZONE_FILLER(board)
    filler.Fill(list(board.Zones()))
    board.BuildConnectivity()

    # 查找 B- zone 对象（使用安全方式访问，避免 SWIG 类型问题）
    zone_obj = None
    for z in board.Zones():
        try:
            if z.GetNetCode() == b_nc:
                zone_obj = z
                break
        except AttributeError:
            # ★ SWIG 原始指针：尝试通过 pcbnew.ZONE 重新包装
            try:
                z_proper = pcbnew.ZONE(z)
                if z_proper.GetNetCode() == b_nc:
                    zone_obj = z_proper
                    break
            except Exception:
                continue
    if zone_obj is None:
        print(f"[adapt_common] WARNING: 未找到 B- zone (netcode={b_nc})，跳过短桩修复")
        return

    stubs = 0
    for fp in board.GetFootprints():
        for pad in fp.Pads():
            try:
                pad_nc = pad.GetNetCode()
            except AttributeError:
                continue
            if pad_nc != b_nc:
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
            t.SetNetCode(b_nc)
            board.Add(t)
            stubs += 1

    if stubs:
        filler.Fill(list(board.Zones()))
        board.BuildConnectivity()
        print(f"[adapt_common] 补充 {stubs} 个 B- 短桩")


def avoid_terminal_overlap(board, terminals: list[dict], margin: float = 1.5) -> int:
    """将挤入端子焊盘区域的内部元件推开，避免重叠。

    Args:
        margin: 元件中心到端子区域边缘的最小间距 (mm)

    Returns:
        被移动的元件数量
    """
    # 收集所有端子占地区域 (x1, y1, x2, y2)
    forbidden = []
    for t in terminals:
        pos = t.get("position", {})
        tx = float(pos.get("x_mm", 0))
        ty = float(pos.get("y_mm", 0))
        hw = max(float(t.get("width_mm", 2.0)) / 2 + margin, 2.0)
        hh = max(float(t.get("height_mm", 2.0)) / 2 + margin, 2.0)
        forbidden.append((tx - hw, ty - hh, tx + hw, ty + hh))

    def _overlaps(px: float, py: float) -> tuple[bool, float, float]:
        """检查 (px,py) 是否落入任一禁止区域；返回 (冲突, 推挤dx, 推挤dy)。"""
        push_dx, push_dy = 0.0, 0.0
        conflict = False
        for x1, y1, x2, y2 in forbidden:
            if x1 <= px <= x2 and y1 <= py <= y2:
                conflict = True
                # 向最近边界外推
                dl, dr = px - x1, x2 - px
                db, du = py - y1, y2 - py
                m = min(dl, dr, db, du)
                if m == dl:
                    push_dx = -(dl + 0.5)
                elif m == dr:
                    push_dx = dr + 0.5
                elif m == db:
                    push_dy = -(db + 0.5)
                else:
                    push_dy = du + 0.5
                break
        return conflict, push_dx, push_dy

    moved = 0
    # 只处理非端子元件
    for fp in board.GetFootprints():
        if fp.GetReference().startswith("TP"):
            continue
        p = fp.GetPosition()
        px, py = pcbnew.ToMM(p.x), pcbnew.ToMM(p.y)
        conflict, dx, dy = _overlaps(px, py)
        if conflict:
            fp.SetPosition(pcbnew.VECTOR2I(_mm(px + dx), _mm(py + dy)))
            moved += 1

    if moved:
        print(f"[adapt_common] 推开 {moved} 个与端子重叠的内部元件")
    return moved


def space_internal_references(board, min_gap: float = 0.3) -> int:
    """统一缩小所有文本并放到元件正下方空闲位置（约束在 PCB 板框内）。

    - 统一缩小所有文本层尺寸（Reference/Value/F.Fab/B.Fab 等丝印与装配层文字
      均设为字高 0.6mm），保证全部文字大小一致。
    - 参考文本统一优先放在元件“正下方”（居中），位置一致、整齐；
      仅当正下方被相邻元件本体/板框占据时才依次尝试其他方向。
    - 隐藏 Value，保留 Reference。
    """
    # 文本尺寸参数（mm）——尽量小以适配紧凑布局，所有层统一
    text_h = 0.6       # 字高
    text_w = 0.6       # 字宽
    text_th = 0.10     # 笔画粗细
    thw = 0.7          # 文本占用半宽（估算）
    thh = 0.4          # 文本占用半高（估算）

    # ★ 统一所有文本尺寸：Reference/Value + 封装图形文本（F.Fab/B.Fab/F.SilkS 等）
    for fp in board.GetFootprints():
        fp.Value().SetVisible(False)
        ref = fp.Reference()
        ref.SetVisible(True)
        ref.SetTextSize(pcbnew.VECTOR2I(_mm(text_w), _mm(text_h)))
        ref.SetTextThickness(_mm(text_th))
        # 统一封装内其他文本（fab 层值/参考、丝印说明等）的尺寸
        for gi in fp.GraphicalItems():
            if gi.GetClass() == "PCB_TEXT":
                gi.SetTextSize(pcbnew.VECTOR2I(_mm(text_w), _mm(text_h)))
                gi.SetTextThickness(_mm(text_th))

    # 板框边界（约束文本在板内，留 0.3mm 边距）
    bbox = board.GetBoardEdgesBoundingBox()
    if bbox.GetWidth() and bbox.GetHeight():
        b_minx = bbox.GetX() / 1e6 + 0.3
        b_maxx = (bbox.GetX() + bbox.GetWidth()) / 1e6 - 0.3
        b_miny = bbox.GetY() / 1e6 + 0.3
        b_maxy = (bbox.GetY() + bbox.GetHeight()) / 1e6 - 0.3
    else:
        b_minx = b_miny = -1e9
        b_maxx = b_maxy = 1e9

    # 收集占用区域（元件本体包围盒，不含文本）
    occupied: list[tuple[float, float, float, float]] = []
    for fp in board.GetFootprints():
        bb = fp.GetBoundingBox(False, False)
        if bb.GetWidth() and bb.GetHeight():
            occupied.append((bb.GetX() / 1e6, bb.GetY() / 1e6,
                             (bb.GetX() + bb.GetWidth()) / 1e6,
                             (bb.GetY() + bb.GetHeight()) / 1e6))

    def _is_free(x: float, y: float) -> bool:
        # 必须在板框内
        if not (b_minx + thw <= x <= b_maxx - thw and b_miny + thh <= y <= b_maxy - thh):
            return False
        # 不与任何占用区域重叠
        for x1, y1, x2, y2 in occupied:
            if (x - thw < x2 and x + thw > x1 and y - thh < y2 and y + thh > y1):
                return False
        return True

    adjusted = 0
    for fp in board.GetFootprints():
        if fp.GetReference().startswith("TP"):
            continue
        p = fp.GetPosition()
        cx, cy = pcbnew.ToMM(p.x), pcbnew.ToMM(p.y)
        bb = fp.GetBoundingBox(False, False)
        hw = (bb.GetWidth() / 1e6) / 2 if bb.GetWidth() else 1.5
        hh = (bb.GetHeight() / 1e6) / 2 if bb.GetHeight() else 1.0

        # ★ 统一优先“正下方”（居中），再依次尝试其他方向作为后备
        gap = 0.35
        candidates = [
            (cx, cy + hh + gap),            # 正下方（首选，保证一致）
            (cx, cy + hh + gap + 0.5),      # 正下方更远
            (cx + hw + 0.4, cy + hh + 0.3), # 右下
            (cx - hw - 0.4, cy + hh + 0.3), # 左下
            (cx + hw + 0.4, cy),            # 正右
            (cx - hw - 0.4, cy),            # 正左
            (cx, cy - hh - gap),            # 正上方
        ]
        placed = False
        for nx, ny in candidates:
            if _is_free(nx, ny):
                fp.Reference().SetPosition(pcbnew.VECTOR2I(_mm(nx), _mm(ny)))
                occupied.append((nx - thw, ny - thh, nx + thw, ny + thh))
                adjusted += 1
                placed = True
                break
        if not placed:
            # 无空闲位：放到元件正下方并钳制到板内
            nx = max(b_minx + thw, min(b_maxx - thw, cx))
            ny = max(b_miny + thh, min(b_maxy - thh, cy + hh + gap))
            fp.Reference().SetPosition(pcbnew.VECTOR2I(_mm(nx), _mm(ny)))
            occupied.append((nx - thw, ny - thh, nx + thw, ny + thh))

    print(f"[adapt_common] 统一文本尺寸(字高{text_h}mm,含 fab/silk 层)，调整 {adjusted} 个参考文本到元件下方")
    return adjusted


# ── 孔槽放置 ──────────────────────────────────────────────────────────────

def place_holes(board, holes: list[dict]) -> int:
    """在识别位置放置 NPTH 孔/槽（Edge.Cuts 内挖孔）。

    holes 来自 spec["holes"]，每个包含:
      - hole_type: round|slot|irregular|groove|protrusion
      - center: {x_mm, y_mm}
      - polygon: [{x_mm, y_mm}, ...]
      - bbox: {x_mm, y_mm, width_mm, height_mm}

    groove/protrusion 已体现在板框轮廓中，跳过。
    """
    if not holes:
        return 0

    placed = 0
    for hole in holes:
        htype = hole.get("hole_type", "round")
        if htype in ("groove", "protrusion"):
            continue  # 已体现在板框轮廓中

        center = hole.get("center", {})
        cx = float(center.get("x_mm", 0))
        cy = float(center.get("y_mm", 0))
        bbox = hole.get("bbox", {})
        w = float(bbox.get("width_mm", 1.0))
        h = float(bbox.get("height_mm", 1.0))

        if htype == "round":
            # 画圆孔（Edge.Cuts 圆）
            circle = pcbnew.PCB_SHAPE(board)
            circle.SetShape(pcbnew.SHAPE_T_CIRCLE)
            circle.SetCenter(pcbnew.VECTOR2I(_mm(cx), _mm(cy)))
            circle.SetEnd(pcbnew.VECTOR2I(_mm(cx + w / 2), _mm(cy)))
            circle.SetLayer(pcbnew.Edge_Cuts)
            circle.SetWidth(_mm(0.1))
            board.Add(circle)
            placed += 1
        elif htype == "slot":
            # 画槽（粗线段）
            seg = pcbnew.PCB_SHAPE(board)
            seg.SetShape(pcbnew.SHAPE_T_SEGMENT)
            seg.SetStart(pcbnew.VECTOR2I(_mm(cx - w / 2), _mm(cy)))
            seg.SetEnd(pcbnew.VECTOR2I(_mm(cx + w / 2), _mm(cy)))
            seg.SetLayer(pcbnew.Edge_Cuts)
            seg.SetWidth(_mm(h))
            board.Add(seg)
            placed += 1
        else:
            # irregular: 用 polygon 画轮廓
            poly = hole.get("polygon", [])
            if len(poly) >= 3:
                for i in range(len(poly)):
                    p1 = poly[i]
                    p2 = poly[(i + 1) % len(poly)]
                    seg = pcbnew.PCB_SHAPE(board)
                    seg.SetShape(pcbnew.SHAPE_T_SEGMENT)
                    seg.SetStart(pcbnew.VECTOR2I(
                        _mm(float(p1.get("x_mm", 0))),
                        _mm(float(p1.get("y_mm", 0)))))
                    seg.SetEnd(pcbnew.VECTOR2I(
                        _mm(float(p2.get("x_mm", 0))),
                        _mm(float(p2.get("y_mm", 0)))))
                    seg.SetLayer(pcbnew.Edge_Cuts)
                    seg.SetWidth(_mm(0.1))
                    board.Add(seg)
                placed += 1

    if placed:
        print(f"[adapt_common] 放置 {placed} 个孔槽")
    return placed


# ── 内部元件布局 ──────────────────────────────────────────────────────────

# 电路拓扑排列顺序（减少走线交叉）
_TOPO_ORDER = ["R1", "U1", "R2", "Q1", "C1", "C2", "C3", "R3", "R4"]

# 封装尺寸估算 (宽mm, 高mm)
_FP_SIZES = {
    "SOT-23-6": (2.9, 2.8),
    "TSSOP-8_4.4x3mm_P0.65mm": (4.4, 3.0),
    "R_0603_1608Metric": (1.6, 0.8),
    "C_0603_1608Metric": (1.6, 0.8),
}

# 方向优化：重要元件（IC/MOS/二极管）保持横向，小型被动元件（R/C/L/保险丝）可竖放
_KEEP_HORIZONTAL_PREFIXES = ("U", "Q", "D", "K", "T", "IC")
_VERTICAL_PREFIXES = ("R", "C", "L", "F", "FB", "RT")


def _ref_prefix(ref: str) -> str:
    """提取参考标号的字母前缀（如 R1→R, U1→U, FB1→FB）。"""
    out = []
    for ch in ref:
        if ch.isalpha():
            out.append(ch)
        elif out:
            break
    return "".join(out).upper()


def _should_rotate_vertical(fp) -> bool:
    """判断元件是否应旋转 90° 竖放（小型被动元件以节省横向空间）。

    - IC/MOS/二极管等重要元件（U/Q/D...）保持横向，便于识别与散热
    - 电阻/电容/电感/保险丝等被动元件（R/C/L/F...）竖放
    - 其他：短边<2mm 且长边<5mm 的小元件也竖放
    """
    prefix = _ref_prefix(fp.GetReference())
    if prefix in _KEEP_HORIZONTAL_PREFIXES:
        return False
    if prefix in _VERTICAL_PREFIXES:
        return True
    w, h = _get_fp_size_mm(fp)
    return min(w, h) < 2.0 and max(w, h) < 5.0


def _get_fp_size_mm(fp) -> tuple[float, float]:
    """获取封装实际占用尺寸（mm）。

    ★ 优先使用真实包围盒（含 courtyard/丝印，不含文本），保证布局后元件视觉上不重叠。
    旧版用封装本体尺寸（如 0603=1.6mm）会低估实际占位（含 courtyard 约3.0mm）导致重叠。
    """
    try:
        bb = fp.GetBoundingBox(False, False)  # 排除文本
        w = bb.GetWidth() / 1e6
        h = bb.GetHeight() / 1e6
        if w > 0.1 and h > 0.1:
            return (w, h)
    except Exception:
        pass
    fpid = str(fp.GetFPID().GetLibItemName())
    if fpid in _FP_SIZES:
        return _FP_SIZES[fpid]
    # fallback: 从焊盘包围盒估算
    pads = list(fp.Pads())
    if pads:
        xs = [pcbnew.ToMM(p.GetPosition().x) for p in pads]
        ys = [pcbnew.ToMM(p.GetPosition().y) for p in pads]
        w = (max(xs) - min(xs)) + 1.0
        h = (max(ys) - min(ys)) + 1.0
        return (max(w, 1.0), max(h, 1.0))
    return (2.0, 2.0)


def place_internal_components(
    board,
    outline_pts: list[tuple[float, float]],
    terminals: list[dict],
    holes: list[dict],
    min_gap: float = 0.2,
) -> int:
    """货架式多行布局 + 碰撞检测，确保元件不重叠且避开端子禁区。

    算法:
    1. 获取内部元件（排除端子封装），按拓扑顺序排列
    2. 禁区 = 正面端子 bbox(+margin) + 孔槽（背面 SMD 焊盘不阻挡正面元件）
    3. 货架式分行：按拓扑顺序从左到右填充，超宽自动换行，整体垂直居中
    4. 碰撞检测迭代推开禁区
    5. 行内最终间距保证 + 板框约束

    ★ 尺寸使用真实包围盒（含 courtyard），避免视觉重叠。
    """
    # 获取内部元件（排除端子焊盘封装）
    _term_refs = set()
    for t in terminals:
        label = terminal_label(t) or t.get("id", "")
        if label:
            _term_refs.add(label)
            for i in range(1, 10):
                _term_refs.add(f"{label}_{i}")

    def _is_terminal_fp(fp) -> bool:
        ref = fp.GetReference()
        if ref.startswith("TP"):
            return True
        if ref in _term_refs:
            return True
        fpid = str(fp.GetFPID().GetLibItemName())
        if "TestPoint" in fpid or "PinHeader" in fpid:
            return True
        return False

    fps = [fp for fp in board.GetFootprints() if not _is_terminal_fp(fp)]
    if not fps:
        return 0

    # 按拓扑顺序排列
    def _sort_key(fp):
        ref = fp.GetReference()
        if ref in _TOPO_ORDER:
            return _TOPO_ORDER.index(ref)
        return len(_TOPO_ORDER) + 1  # 未知元件排最后

    fps.sort(key=_sort_key)

    # 板框 bbox
    xs = [p[0] for p in outline_pts]
    ys = [p[1] for p in outline_pts]
    board_minx, board_maxx = min(xs), max(xs)
    board_miny, board_maxy = min(ys), max(ys)
    board_cx = (board_minx + board_maxx) / 2
    board_cy = (board_miny + board_maxy) / 2

    inset = 1.0
    avail_minx = board_minx + inset
    avail_maxx = board_maxx - inset
    avail_w = avail_maxx - avail_minx

    # 禁区：仅正面端子 + 孔槽（背面 SMD 焊盘位于另一层，不阻挡正面元件）
    forbidden: list[tuple[float, float, float, float]] = []
    term_margin = 1.2
    for t in terminals:
        if t.get("side") == "back":
            continue
        pos = t.get("position", {})
        tx = float(pos.get("x_mm", 0))
        ty = float(pos.get("y_mm", 0))
        hw = float(t.get("width_mm", 2.0)) / 2 + term_margin
        hh = float(t.get("height_mm", 2.0)) / 2 + term_margin
        forbidden.append((tx - hw, ty - hh, tx + hw, ty + hh))
    hole_margin = 0.5
    for hole in holes:
        bbox = hole.get("bbox", {})
        hx = float(bbox.get("x_mm", 0))
        hy = float(bbox.get("y_mm", 0))
        hw = float(bbox.get("width_mm", 1.0)) / 2 + hole_margin
        hh = float(bbox.get("height_mm", 1.0)) / 2 + hole_margin
        forbidden.append((hx - hw, hy - hh, hx + hw, hy + hh))

    # ★ 方向优化：小型被动元件（R/C/L...）旋转 90° 竖放，节省横向空间；
    #   重要元件（IC/MOS）保持横向。旋转后再取包围盒尺寸，布局按竖放尺寸计算。
    rotated_count = 0
    for fp in fps:
        if _should_rotate_vertical(fp):
            fp.Rotate(fp.GetPosition(), pcbnew.EDA_ANGLE(90.0))
            # 参考文本保持水平，便于阅读
            fp.Reference().SetTextAngle(pcbnew.EDA_ANGLE(0.0))
            rotated_count += 1

    # 每个元件的真实包围盒尺寸（已含旋转后的竖放尺寸）
    fp_sizes = [(fp, _get_fp_size_mm(fp)) for fp in fps]
    n = len(fps)

    # ── 货架式分行：按拓扑顺序从左到右填充，超宽换行 ──
    rows: list[list[int]] = []
    cur: list[int] = []
    cur_w = 0.0
    for i in range(n):
        w = fp_sizes[i][1][0]
        need = w + (min_gap if cur else 0.0)
        if cur and cur_w + need > avail_w:
            rows.append(cur)
            cur = []
            cur_w = 0.0
            need = w
        cur.append(i)
        cur_w += need
    if cur:
        rows.append(cur)

    # 每行高度 = 行内最大元件高；整体垂直居中
    row_h = [max(fp_sizes[i][1][1] for i in r) for r in rows]
    total_h = sum(row_h) + min_gap * (len(rows) - 1)
    y_top = board_cy - total_h / 2

    positions: list[list[float]] = [[0.0, 0.0] for _ in range(n)]
    y_cursor = y_top
    for row in rows:
        rh = row_h[rows.index(row)]
        row_cy = y_cursor + rh / 2
        row_w = sum(fp_sizes[i][1][0] for i in row) + min_gap * (len(row) - 1)
        x_cursor = board_cx - row_w / 2  # 行内水平居中
        for i in row:
            w = fp_sizes[i][1][0]
            positions[i] = [x_cursor + w / 2, row_cy]
            x_cursor += w + min_gap
        y_cursor += rh + min_gap

    # ── 碰撞检测：推开禁区 + 板框约束 ──
    for _iteration in range(30):
        moved_any = False
        for i in range(n):
            w, h = fp_sizes[i][1]
            px, py = positions[i]
            for x1, y1, x2, y2 in forbidden:
                if (px + w / 2 > x1 and px - w / 2 < x2 and
                        py + h / 2 > y1 and py - h / 2 < y2):
                    dl = px - w / 2 - x1
                    dr = x2 - (px + w / 2)
                    db = py - h / 2 - y1
                    du = y2 - (py + h / 2)
                    m = min(abs(dl), abs(dr), abs(db), abs(du))
                    if m == abs(dl):
                        positions[i][0] = x1 - w / 2 - 0.2
                    elif m == abs(dr):
                        positions[i][0] = x2 + w / 2 + 0.2
                    elif m == abs(db):
                        positions[i][1] = y1 - h / 2 - 0.2
                    else:
                        positions[i][1] = y2 + h / 2 + 0.2
                    moved_any = True
                    break
        for i in range(n):
            w, h = fp_sizes[i][1]
            positions[i][0] = max(board_minx + inset + w / 2,
                                  min(board_maxx - inset - w / 2, positions[i][0]))
            positions[i][1] = max(board_miny + inset + h / 2,
                                  min(board_maxy - inset - h / 2, positions[i][1]))
        if not moved_any:
            break

    # ── 行内最终间距保证：每行按 X 排序，只向右推 ──
    for row in rows:
        order = sorted(row, key=lambda i: positions[i][0])
        for k in range(1, len(order)):
            i = order[k - 1]
            j = order[k]
            wi = fp_sizes[i][1][0]
            wj = fp_sizes[j][1][0]
            min_dist = wi / 2 + wj / 2 + min_gap
            if positions[j][0] - positions[i][0] < min_dist:
                positions[j][0] = positions[i][0] + min_dist
        for i in row:
            w = fp_sizes[i][1][0]
            positions[i][0] = max(board_minx + inset + w / 2,
                                  min(board_maxx - inset - w / 2, positions[i][0]))

    # 应用位置
    for i, fp in enumerate(fps):
        fp.SetPosition(pcbnew.VECTOR2I(
            _mm(positions[i][0]), _mm(positions[i][1])))

    print(f"[adapt_common] 布局 {n} 个内部元件（{len(rows)} 行货架式布局，{rotated_count} 个竖放，真实包围盒尺寸）")
    return n


# ── 分通道曼哈顿布线 ──────────────────────────────────────────────────────

def center_board(board) -> None:
    """将整个 PCB 设计平移到原点居中，使板框中心位于 (0,0)。

    这样在 KiCad 中打开时，板子会显示在画布中央而非左上角。
    """
    # 获取板框边界（Edge.Cuts 层图形的包围盒）
    bbox = board.GetBoardEdgesBoundingBox()
    if not bbox.GetWidth() or not bbox.GetHeight():
        return
    cx = bbox.GetX() + bbox.GetWidth() / 2
    cy = bbox.GetY() + bbox.GetHeight() / 2
    # 平移整个板子，使中心移到原点
    offset = pcbnew.VECTOR2I(int(-cx), int(-cy))
    board.Move(offset)
    print(f"[adapt_common] 板框居中: 偏移 ({-cx / 1e6:.1f}, {-cy / 1e6:.1f}) mm")


def route_nets_manhattan(board, zone_layer: int) -> int:
    """双层隔离布线，避免不同网络走线短路。

    ★ 策略（针对高密度双面板）：
      - B- 网络：不显式走线，由 zone_layer(B.Cu) 上的地平面 zone 连接所有 B- 焊盘，
        仅为正面 B- SMD 焊盘添加过孔连接到 B.Cu 地平面。
      - 其余信号网络：在 F.Cu（元件面）走线。F.Cu 相对空旷（仅焊盘/元件），
        采用“错开中点”的折线连接同网络焊盘，避免多网络走线在中部重叠交叉。
      - 线宽：电源网络 0.4mm，信号网络 0.2mm（满足 0.2mm 间距规则）。

    这样信号走线(F.Cu)与地平面(B.Cu)分层隔离，从根本上消除原单层通道布线的
    通道重叠短路、竖直引线穿越其他通道、zone 与信号争层等问题。
    """
    # 收集每个网络的焊盘
    net_pads: dict[int, list[tuple[float, float, bool]]] = {}  # net_code -> [(x,y,is_smd)]
    for fp in board.GetFootprints():
        for pad in fp.Pads():
            nc = pad.GetNetCode()
            if nc == 0:
                continue
            p = pad.GetPosition()
            x, y = pcbnew.ToMM(p.x), pcbnew.ToMM(p.y)
            is_smd = pad.GetAttribute() == pcbnew.PAD_ATTRIB_SMD
            net_pads.setdefault(nc, []).append((x, y, is_smd))

    # ── B- 网络：地平面 zone 连接，仅为正面 SMD 焊盘加过孔 ──
    b_net = board.FindNet("B-")
    b_nc = b_net.GetNetCode() if b_net else -1
    if b_nc > 0 and b_nc in net_pads:
        b_via = 0
        for x, y, is_smd in net_pads[b_nc]:
            if is_smd:
                via = pcbnew.PCB_VIA(board)
                via.SetPosition(pcbnew.VECTOR2I(_mm(x), _mm(y)))
                via.SetWidth(_mm(0.8))
                via.SetDrill(_mm(0.4))
                via.SetViaType(pcbnew.VIATYPE_THROUGH)
                board.Add(via)
                via.SetNetCode(b_nc)
                b_via += 1
        print(f"[adapt_common] B- 网络: {b_via} 过孔连接地平面（zone 覆盖，无显式走线）")

    # ── 信号网络：F.Cu 贪心布线（带几何间距检查，保证不短路）──
    routable = [(nc, pads) for nc, pads in net_pads.items() if nc != b_nc and len(pads) >= 2]
    routable.sort(key=lambda item: board.FindNet(item[0]).GetNetname() if board.FindNet(item[0]) else "")
    if not routable:
        return 0

    fcu = pcbnew.F_Cu
    clearance = 0.2  # mm 最小间距

    # 收集所有焊盘障碍（旋转矩形 OBB：x, y, 半宽, 半高, 角度rad, 网络码）
    pad_obs: list[tuple[float, float, float, float, float, int]] = []
    for fp in board.GetFootprints():
        for pad in fp.Pads():
            nc = pad.GetNetCode()
            if nc == 0:
                continue
            p = pad.GetPosition()
            sz = pad.GetSize()
            pad_obs.append((pcbnew.ToMM(p.x), pcbnew.ToMM(p.y),
                            sz.x / 1e6 / 2, sz.y / 1e6 / 2,
                            math.radians(pad.GetOrientation().AsDegrees()), nc))

    def _closest_on_seg(p, a, b):
        ax, ay, bx, by = a[0], a[1], b[0], b[1]
        dx, dy = bx - ax, by - ay
        L2 = dx * dx + dy * dy
        t = 0.0 if L2 < 1e-9 else max(0.0, min(1.0, ((p[0] - ax) * dx + (p[1] - ay) * dy) / L2))
        return (ax + t * dx, ay + t * dy)

    def _pt_seg_dist(p, a, b):
        c = _closest_on_seg(p, a, b)
        return math.hypot(p[0] - c[0], p[1] - c[1])

    def _pt_obb_dist(px, py, ob) -> float:
        """点到旋转矩形的距离（内部为 0）。"""
        cx, cy, hw, hh, ang, _ = ob
        c, s = math.cos(ang), math.sin(ang)
        lx = (px - cx) * c + (py - cy) * s
        ly = -(px - cx) * s + (py - cy) * c
        ddx = max(abs(lx) - hw, 0.0)
        ddy = max(abs(ly) - hh, 0.0)
        return math.hypot(ddx, ddy)

    def _seg_obb_clear(a, b, ob, req) -> bool:
        """线段 ab 与旋转矩形 ob 的间距是否 >= req（分离轴定理，保守）。"""
        cx, cy, hw, hh, ang, _ = ob
        c, s = math.cos(ang), math.sin(ang)
        # 变换到 OBB 局部坐标
        p0 = ((a[0] - cx) * c + (a[1] - cy) * s, -(a[0] - cx) * s + (a[1] - cy) * c)
        p1 = ((b[0] - cx) * c + (b[1] - cy) * s, -(b[0] - cx) * s + (b[1] - cy) * c)
        ex, ey = hw + req, hh + req
        # 轴 1：OBB 法向。线段投影区间与 [-ex,ex] 重叠？
        lo, hi = min(p0[0], p1[0]), max(p0[0], p1[0])
        if hi < -ex or lo > ex:
            return True
        # 轴 2：OBB 切向
        lo, hi = min(p0[1], p1[1]), max(p0[1], p1[1])
        if hi < -ey or lo > ey:
            return True
        # 轴 3：线段法向。矩形投影半径 vs 线段到原点距离
        dx, dy = p1[0] - p0[0], p1[1] - p0[1]
        L = math.hypot(dx, dy)
        if L < 1e-9:
            return math.hypot(p0[0], p0[1]) > req  # 点：退化为点距
        nx, ny = -dy / L, dx / L
        d_seg = abs(p0[0] * nx + p0[1] * ny)
        r_rect = ex * abs(nx) + ey * abs(ny)
        return d_seg > r_rect

    def seg_seg_dist(a, b, c, d) -> float:
        def ccw(p, q, r):
            return (q[1] - p[1]) * (r[0] - q[0]) - (q[0] - p[0]) * (r[1] - q[1])
        def intersect(p1, p2, p3, p4):
            d1, d2 = ccw(p3, p4, p1), ccw(p3, p4, p2)
            d3, d4 = ccw(p1, p2, p3), ccw(p1, p2, p4)
            return ((d1 > 0) != (d2 > 0)) and ((d3 > 0) != (d4 > 0))
        if intersect(a, b, c, d):
            return 0.0
        return min(_pt_seg_dist(a, c, d), _pt_seg_dist(b, c, d),
                   _pt_seg_dist(c, a, b), _pt_seg_dist(d, a, b))

    # 已布走线障碍（同层 F.Cu）：(x1,y1,x2,y2,半宽,网络码)
    tracks: list[tuple[float, float, float, float, float, int]] = []

    def seg_ok(s, nc, w) -> bool:
        """检查线段 s 是否与异网络焊盘/已布走线冲突（旋转矩形精确间距）。"""
        a, b = (s[0], s[1]), (s[2], s[3])
        req = w / 2 + clearance
        for ob in pad_obs:
            if ob[5] == nc:
                continue
            if not _seg_obb_clear(a, b, ob, req):
                return False
        for x1, y1, x2, y2, hw, tnc in tracks:
            if tnc == nc:
                continue
            if seg_seg_dist(a, b, (x1, y1), (x2, y2)) < w / 2 + hw + clearance:
                return False
        return True

    def add_seg(s, nc, w):
        tracks.append((s[0], s[1], s[2], s[3], w / 2, nc))
        t = pcbnew.PCB_TRACK(board)
        t.SetStart(pcbnew.VECTOR2I(_mm(s[0]), _mm(s[1])))
        t.SetEnd(pcbnew.VECTOR2I(_mm(s[2]), _mm(s[3])))
        t.SetWidth(_mm(w))
        t.SetLayer(fcu)
        t.SetNetCode(nc)
        board.Add(t)

    # 候选通道 Y 列表（板框内密集采样，供水平干线选择空旷高度）
    bb = board.GetBoardEdgesBoundingBox()
    if bb.GetHeight():
        y_lo = bb.GetY() / 1e6 + 0.4
        y_hi = (bb.GetY() + bb.GetHeight()) / 1e6 - 0.4
    else:
        y_lo, y_hi = -3.0, 3.0
    chan_ys = [y_lo + (y_hi - y_lo) * k / 12 for k in range(13)]
    # 优先靠近板中心（短引线），再向两侧扩展
    chan_ys.sort(key=lambda y: abs(y - (y_lo + y_hi) / 2))

    routed = 0
    skipped = 0
    for ch_idx, (nc, pads) in enumerate(routable):
        net_obj = board.FindNet(nc)
        net_name = net_obj.GetNetname() if net_obj else ""
        width = 0.4 if net_name in ("B+", "P-", "VDD") else 0.2

        sp = sorted(pads, key=lambda t: t[0])
        net_ok = True
        for i in range(len(sp) - 1):
            ax, ay, _ = sp[i]
            bx, by, _ = sp[i + 1]
            if abs(ax - bx) < 0.05:
                cands = [[(ax, ay, bx, by)]]
            else:
                # 多通道候选：水平干线在不同 Y 高度尝试，找到不与焊盘冲突的空旷通道
                cands = []
                for cy in chan_ys:
                    cands.append([(ax, ay, ax, cy), (ax, cy, bx, cy), (bx, cy, bx, by)])
                cands.append([(ax, ay, bx, ay), (bx, ay, bx, by)])
                cands.append([(ax, ay, ax, by), (ax, by, bx, by)])
            placed_pair = False
            for path in cands:
                path = [s for s in path if not (abs(s[0] - s[2]) < 0.02 and abs(s[1] - s[3]) < 0.02)]
                if path and all(seg_ok(s, nc, width) for s in path):
                    for s in path:
                        add_seg(s, nc, width)
                    placed_pair = True
                    break
            if not placed_pair:
                net_ok = False
                break
        if net_ok:
            routed += 1
        else:
            skipped += 1

    print(f"[adapt_common] 贪心布线 {routed} 个网络（OBB 精确间距检查保证不短路，{skipped} 个因密度过高跳过待手工，B- 地平面@{board.GetLayerName(zone_layer)}）")
    return routed


# ── Freerouting Cloud API 自动布线器 ──────────────────────────────────────

def _freerouting_headers(content_type: str = "application/json") -> dict:
    """构造 Freerouting Cloud API 请求头。"""
    return {
        "Authorization": f"Bearer {_FREEROUTING_API_KEY}",
        "Freerouting-Profile-ID": _FREEROUTING_PROFILE_ID,
        "Freerouting-Environment-Host": _FREEROUTING_HOST,
        "Content-Type": content_type,
        "User-Agent": "PCBDesign-Autorouter/1.0",
        "Accept": "application/json",
    }


def _freerouting_api(method: str, path: str, data: bytes | None = None,
                     content_type: str = "application/json",
                     timeout: int = 30) -> tuple[int, bytes]:
    """调用 Freerouting Cloud API 端点，返回 (status_code, response_body)。"""
    import urllib.request
    import urllib.error

    url = f"{_FREEROUTING_BASE_URL}{path}"
    headers = _freerouting_headers(content_type)
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, resp.read()
    except urllib.error.HTTPError as e:
        body = e.read() if e.fp else b""
        return e.code, body
    except Exception as e:
        print(f"[freerouting-api] 请求失败: {method} {path} -> {e}")
        return 0, str(e).encode()


def _fix_narrow_tracks(board, min_width_mm: float = 0.2) -> None:
    """修复 Freerouting 产生的不合规窄走线。

    Freerouting Cloud 可能使用比 DSN 规则更窄的走线（如 0.15mm 而非 0.2mm）。
    此函数将所有低于 min_width_mm 的走线加宽到最小值。
    """
    min_w = _mm(min_width_mm)
    fixed = 0
    for t in board.GetTracks():
        if t.GetClass() != "PCB_TRACK":
            continue
        try:
            w = t.GetWidth()
        except Exception:
            continue
        if w < min_w:
            t.SetWidth(min_w)
            fixed += 1
    if fixed:
        print(f"[freerouting-api] 修复 {fixed} 段窄走线 (< {min_width_mm}mm → {min_width_mm}mm)")


def _fix_dangling_vias(board) -> None:
    """移除 Freerouting 产生的悬空过孔（只在一层有走线、另一层无连接的过孔）。

    悬空过孔意味着过孔没有起到层间连接作用，是 Freerouting 留下的冗余元素。
    判断逻辑：检查过孔在 F.Cu 和 B.Cu 两面是否都有走线/焊盘连接。
    如果只在一面有连接，说明过孔没有实际用途，可以安全移除。
    """
    removed = 0
    for t in list(board.GetTracks()):
        if t.GetClass() != "PCB_VIA":
            continue
        pos = t.GetPosition()
        nc = t.GetNetCode()
        if nc <= 0:
            continue

        # 收集哪些铜层有走线/焊盘连接到此过孔
        connected_layers = set()
        pos_mm = (pcbnew.ToMM(pos.x), pcbnew.ToMM(pos.y))
        tolerance = 0.15  # mm

        for other in board.GetTracks():
            if other is t or other.GetClass() != "PCB_TRACK":
                continue
            if other.GetNetCode() != nc:
                continue
            layer = other.GetLayer()
            if layer not in (pcbnew.F_Cu, pcbnew.B_Cu):
                continue
            s, e = other.GetStart(), other.GetEnd()
            for pt in (s, e):
                dist_mm = math.hypot(pcbnew.ToMM(pt.x) - pos_mm[0],
                                     pcbnew.ToMM(pt.y) - pos_mm[1])
                if dist_mm < tolerance:
                    connected_layers.add(layer)
                    break

        # 也检查焊盘
        if len(connected_layers) < 2:
            for fp in board.GetFootprints():
                for pad in fp.Pads():
                    if pad.GetNetCode() != nc:
                        continue
                    pp = pad.GetPosition()
                    dist_mm = math.hypot(pcbnew.ToMM(pp.x) - pos_mm[0],
                                         pcbnew.ToMM(pp.y) - pos_mm[1])
                    if dist_mm < tolerance:
                        pad_layer = pad.GetLayer()
                        if pad_layer in (pcbnew.F_Cu, pcbnew.B_Cu):
                            connected_layers.add(pad_layer)
                    if len(connected_layers) >= 2:
                        break
                if len(connected_layers) >= 2:
                    break

        # 只有一层连接 → 过孔没有层间作用，移除
        if len(connected_layers) < 2:
            board.Remove(t)
            removed += 1
    if removed:
        print(f"[freerouting-api] 移除 {removed} 个悬空过孔（仅单层连接，无层间作用）")


def run_freerouting_autorouter(board, max_passes: int = 30, timeout: int = 300) -> bool:
    """调用 Freerouting Cloud API 对 board 布线（导出 DSN → 上传 → 云端布线 → 下载 SES → 导入）。

    相比本地 JAR 方式：
      - 无需安装 Java / 本地 JAR
      - 云端算力更强，布线质量更高
      - 支持更复杂的板子

    Args:
        board: pcbnew.BOARD（应已完成元件布局 + zone 重建，且无旧走线）
        max_passes: 最大布线轮数
        timeout: API 轮询总超时（秒），云端布线可能需要较长时间

    Returns:
        True 表示布线成功并已导入走线；False 表示失败。
    """
    import json as _json
    import time

    workdir = Path(tempfile.mkdtemp(prefix="pcb_route_"))
    dsn = workdir / "design.dsn"
    ses = workdir / "design.ses"
    try:
        # 1) 检查 API 连通性
        status_code, body = _freerouting_api("GET", "/system/status", timeout=15)
        if status_code != 200:
            print(f"[freerouting-api] WARNING: API 不可达 (status={status_code})")
            return False
        print(f"[freerouting-api] API 状态正常: {body[:200].decode(errors='replace')}")

        # 2) 导出 Specctra DSN
        if not pcbnew.ExportSpecctraDSN(board, str(dsn)):
            print("[freerouting-api] WARNING: DSN 导出失败")
            return False
        dsn_size = dsn.stat().st_size
        print(f"[freerouting-api] DSN 导出成功 ({dsn_size / 1024:.1f} KB)")
        # ★ 调试：保存 DSN 副本以便检查约束
        try:
            import shutil
            debug_dsn = Path(os.environ.get("TEMP", "/tmp")) / "pcbdesign_debug.dsn"
            shutil.copy2(str(dsn), str(debug_dsn))
            print(f"[freerouting-api] DSN 调试副本: {debug_dsn}")
        except Exception:
            pass

        # 3) 创建布线会话
        status_code, body = _freerouting_api("POST", "/sessions/create", timeout=30)
        if status_code not in (200, 201):
            print(f"[freerouting-api] WARNING: 创建会话失败 (status={status_code}): "
                  f"{body[:500].decode(errors='replace')}")
            return False
        session_data = _json.loads(body)
        session_id = (session_data.get("id") or session_data.get("sessionId")
                      or session_data.get("session_id", ""))
        if not session_id:
            print(f"[freerouting-api] WARNING: 会话创建返回无有效 ID: {body[:300].decode(errors='replace')}")
            return False
        print(f"[freerouting-api] 会话已创建: {session_id}")

        # 4) 提交布线任务
        enqueue_body = _json.dumps({
            "session_id": session_id,
            "name": "PCBDesign-auto",
            "priority": "NORMAL",
        }).encode()
        status_code, body = _freerouting_api("POST", "/jobs/enqueue",
                                             data=enqueue_body, timeout=30)
        if status_code not in (200, 201):
            print(f"[freerouting-api] WARNING: 提交任务失败 (status={status_code}): "
                  f"{body[:500].decode(errors='replace')}")
            return False
        job_data = _json.loads(body)
        job_id = job_data.get("jobId") or job_data.get("job_id") or job_data.get("id", "")
        print(f"[freerouting-api] 任务已提交: {job_id}")

        # 4b) 设置布线参数（完整适配本地 JAR 版本的微调参数）
        settings_body = _json.dumps({
            "max_passes": max_passes,
            "via_costs": 50,
            "plane_via_costs": 5,
            "fanout_max_passes": 10,
            "start_ripup_costs": 100,
            "trace_pull_tight_accuracy": 1000,
            "automatic_neckdown": True,
            "improvement_threshold": 0.02,
        }).encode()
        status_code, body = _freerouting_api(
            "POST", f"/jobs/{job_id}/settings",
            data=settings_body, timeout=30,
        )
        if status_code not in (200, 201):
            print(f"[freerouting-api] WARNING: 设置参数失败 (status={status_code}): "
                  f"{body[:500].decode(errors='replace')}")
        else:
            resp_data = _json.loads(body) if body else {}
            print(f"[freerouting-api] 布线参数已设置: max_passes={max_passes}, "
                  f"via_costs=50, plane_via_costs=5, start_ripup_costs=100")
            print(f"[freerouting-api] 服务器确认: {body[:300].decode(errors='replace')}")

        # 5) 上传 DSN 文件（Base64 编码的 JSON 格式）
        import base64
        dsn_bytes = dsn.read_bytes()
        dsn_b64 = base64.b64encode(dsn_bytes).decode()
        input_body = _json.dumps({
            "filename": dsn.name,
            "data": dsn_b64,
        }).encode()
        status_code, body = _freerouting_api(
            "POST", f"/jobs/{job_id}/input",
            data=input_body,
            timeout=120,
        )
        if status_code not in (200, 201):
            print(f"[freerouting-api] WARNING: 上传 DSN 失败 (status={status_code})")
            return False
        print(f"[freerouting-api] DSN 已上传 ({len(dsn_bytes) / 1024:.1f} KB)")

        # 6) 启动布线任务
        status_code, body = _freerouting_api("PUT", f"/jobs/{job_id}/start", timeout=30)
        if status_code not in (200, 201, 202):
            print(f"[freerouting-api] WARNING: 启动任务失败 (status={status_code}): "
                  f"{body[:500].decode(errors='replace')}")
            return False
        print("[freerouting-api] 布线任务已启动，等待云端处理...")

        # 7) 轮询等待完成
        poll_interval = 3  # 秒
        elapsed = 0.0
        job_status = "UNKNOWN"
        while elapsed < timeout:
            time.sleep(poll_interval)
            elapsed += poll_interval

            status_code, body = _freerouting_api("GET", f"/jobs/{job_id}", timeout=30)
            if status_code != 200:
                print(f"[freerouting-api] 轮询异常 (status={status_code})，重试...")
                continue

            job_info = _json.loads(body)
            job_status = job_info.get("state", "UNKNOWN")

            if job_status in ("COMPLETED", "FINISHED", "DONE", "TERMINATED"):
                print(f"[freerouting-api] 布线完成 ({elapsed:.0f}s)")
                break
            elif job_status in ("FAILED", "ERROR", "CANCELLED"):
                err_msg = job_info.get("message", job_info.get("error", "未知错误"))
                print(f"[freerouting-api] WARNING: 布线失败: {err_msg}")
                return False
            else:
                stage = job_info.get("stage", "")
                cur_pass = job_info.get("current_pass", "")
                if stage:
                    print(f"[freerouting-api] 布线中... stage={stage} pass={cur_pass} ({elapsed:.0f}s)")
                elif int(elapsed) % 15 == 0:
                    print(f"[freerouting-api] 等待中... state={job_status} ({elapsed:.0f}s)")
        else:
            print(f"[freerouting-api] WARNING: 布线超时 (>{timeout}s, 最后状态={job_status})")
            return False

        # 8) 下载 SES 结果（返回 JSON，data 字段为 Base64 编码）
        status_code, body = _freerouting_api("GET", f"/jobs/{job_id}/output", timeout=60)
        if status_code not in (200, 202):
            print(f"[freerouting-api] WARNING: 下载结果失败 (status={status_code})")
            return False
        import base64
        output_data = _json.loads(body)
        ses_b64 = output_data.get("data", "")
        ses_bytes = base64.b64decode(ses_b64)
        ses.write_bytes(ses_bytes)
        print(f"[freerouting-api] SES 已下载 ({len(ses_bytes) / 1024:.1f} KB)")

        # 9) 导入 SES 布线结果
        if not pcbnew.ImportSpecctraSES(board, str(ses)):
            print("[freerouting-api] WARNING: SES 导入失败")
            return False

        # 10) 后处理：修复 Freerouting 可能产生的不合规线宽和悬空过孔
        _fix_narrow_tracks(board)
        _fix_dangling_vias(board)

        n_tracks = len([t for t in board.GetTracks() if t.GetClass() == "PCB_TRACK"])
        n_vias = len([t for t in board.GetTracks() if t.GetClass() == "PCB_VIA"])
        # 统计各网络布线连接情况
        nets_with_tracks = set()
        for t in board.GetTracks():
            nc = t.GetNetCode()
            if nc > 0:
                nets_with_tracks.add(nc)
        total_nets = len([n for n in board.GetNetsByNetcode() if n > 0]) if hasattr(board, 'GetNetsByNetcode') else 0
        print(f"[freerouting-api] Freerouting Cloud 布线完成: {n_tracks} 走线 + {n_vias} 过孔")
        if total_nets:
            print(f"[freerouting-api] 已布线网络: {len(nets_with_tracks)}/{total_nets}")
        return True
    finally:
        for f in (dsn, ses):
            try:
                if f.exists():
                    f.unlink()
            except OSError:
                pass
        try:
            workdir.rmdir()
        except OSError:
            pass


def remove_redundant_bminus_tracks(board) -> int:
    """删除 Freerouting 为 B- 网络布设的冗余走线（保留过孔）。

    B- 由地平面 zone 负责连接，Freerouting 额外布的 B- 走线不仅多余，
    还可能靠近 B+/信号焊盘引发短路。仅删除走线、保留过孔（正面 B- 焊盘
    需通过过孔连接到 B.Cu 地平面），之后由 zone 填充保证 B- 连通。
    """
    b_net = board.FindNet("B-")
    if b_net is None:
        return 0
    b_nc = b_net.GetNetCode()
    # ★ 缓存 B- netcode（后续 zone 操作后 SWIG 对象可能失效）
    global _cached_b_netcode
    _cached_b_netcode = b_nc
    removed = 0
    for t in list(board.GetTracks()):
        # 仅删除走线（PCB_TRACK），保留过孔（PCB_VIA）以维持层间连接
        if t.GetNetCode() == b_nc and t.GetClass() == "PCB_TRACK":
            board.Remove(t)
            removed += 1
    if removed:
        print(f"[adapt_common] 删除 {removed} 段冗余 B- 走线（保留过孔，地平面 zone 连接）")
    return removed


def _segment_hits_pad(x1: float, y1: float, x2: float, y2: float,
                      pad_cx: float, pad_cy: float, pad_w: float, pad_h: float,
                      margin: float = 0.3) -> bool:
    """判断线段 (x1,y1)→(x2,y2) 是否与焊盘矩形（含 margin）相交。

    使用参数化线段-矩形裁剪（Liáng-Barsky 简化版）。
    """
    half_w = pad_w / 2 + margin
    half_h = pad_h / 2 + margin
    # 矩形边界
    xmin, xmax = pad_cx - half_w, pad_cx + half_w
    ymin, ymax = pad_cy - half_h, pad_cy + half_h
    dx, dy = x2 - x1, y2 - y1
    t_min, t_max = 0.0, 1.0
    for denom, num in [(-dx, x1 - xmin), (dx, xmax - x1),
                        (-dy, y1 - ymin), (dy, ymax - y1)]:
        if abs(denom) < 1e-9:
            if num < 0:
                return False
        else:
            t = num / denom
            if denom < 0:
                t_min = max(t_min, t)
            else:
                t_max = min(t_max, t)
            if t_min > t_max:
                return False
    return True


def _segment_near_segment(x1: float, y1: float, x2: float, y2: float,
                          sx1: float, sy1: float, sx2: float, sy2: float,
                          min_dist: float = 0.3) -> bool:
    """判断两条线段之间的最小距离是否小于 min_dist。

    使用采样法（将每条线段分成 N 段，检查端点到另一条线段的距离）。
    """
    import itertools
    # 将两条线段各分成 4 段，得到 5 个采样点
    n_samples = 5
    pts_a = [(x1 + (x2 - x1) * i / (n_samples - 1),
              y1 + (y2 - y1) * i / (n_samples - 1)) for i in range(n_samples)]
    pts_b = [(sx1 + (sx2 - sx1) * i / (n_samples - 1),
              sy1 + (sy2 - sy1) * i / (n_samples - 1)) for i in range(n_samples)]
    for (ax, ay), (bx, by) in itertools.product(pts_a, pts_b):
        if math.hypot(ax - bx, ay - by) < min_dist:
            return True
    return False


def _track_clears_nets(board, x1: float, y1: float, x2: float, y2: float,
                        b_net_code: int, layer: int, clearance: float = 0.3) -> bool:
    """检查线段 (x1,y1)→(x2,y2) 在指定层是否与其他网络焊盘或走线冲突。

    检查对象：
    1. 其他网络的焊盘（所有层）
    2. 其他网络的走线（同层）
    """
    # 检查焊盘
    for fp in board.GetFootprints():
        for pad in fp.Pads():
            if pad.GetNetCode() == b_net_code:
                continue
            p = pad.GetPosition()
            cx, cy = pcbnew.ToMM(p.x), pcbnew.ToMM(p.y)
            size = pad.GetSize()
            pw, ph = pcbnew.ToMM(size.x), pcbnew.ToMM(size.y)
            if _segment_hits_pad(x1, y1, x2, y2, cx, cy, pw, ph, clearance):
                return False
    # 检查同层其他网络的走线
    for t in board.GetTracks():
        if t.GetClass() != "PCB_TRACK":
            continue
        if t.GetNetCode() == b_net_code:
            continue
        if t.GetLayer() != layer:
            continue
        s, e = t.GetStart(), t.GetEnd()
        sx1, sy1 = pcbnew.ToMM(s.x), pcbnew.ToMM(s.y)
        sx2, sy2 = pcbnew.ToMM(e.x), pcbnew.ToMM(e.y)
        if _segment_near_segment(x1, y1, x2, y2, sx1, sy1, sx2, sy2, clearance + 0.2):
            return False
    return True


def connect_bminus_pads_to_zone(board, zone_layer: int) -> int:
    """显式连接 F.Cu/B.Cu 上未连线的 B- 焊盘到最近的 B- 过孔/走线。

    Freerouting 不处理 B- 网络（由 zone 负责），但 B- 焊盘可能没有走线
    连接到过孔。此函数为每个未连线的 B- 焊盘添加走线到最近的 B- 过孔，
    并确保走线不穿越其他网络的焊盘（碰撞检测）。

    如果直线到最近过孔会被阻挡，则尝试其他过孔或 L 形走线。
    """
    # ★ 优先使用缓存的 B- netcode（SWIG 对象在 zone 操作后可能失效）
    global _cached_b_netcode
    b_nc = _cached_b_netcode
    if b_nc < 0:
        b_net = board.FindNet("B-")
        if b_net is None:
            print("[adapt_common] B- 网络不存在，跳过焊盘连接")
            return 0
        try:
            b_nc = b_net.GetNetCode()
            _cached_b_netcode = b_nc
        except AttributeError:
            # KiCad 10 的 FindNet 可能返回无效对象，用焊盘反查 netcode
            print(f"[adapt_common] FindNet 返回无效对象，尝试备用查找...")
            for fp in board.GetFootprints():
                for pad in fp.Pads():
                    try:
                        if pad.GetNetname() == "B-":
                            b_nc = pad.GetNetCode()
                            _cached_b_netcode = b_nc
                            break
                    except Exception:
                        continue
                if b_nc > 0:
                    break
            if b_nc < 0:
                print("[adapt_common] 无法找到 B- 网络代码，跳过")
                return 0
            print(f"[adapt_common] 备用查找成功: B- netcode={b_nc}")

    # 收集所有 B- 过孔位置（PCB_VIA 连接 F.Cu ↔ B.Cu）
    b_vias: list[tuple[float, float]] = []
    for t in board.GetTracks():
        if t.GetClass() == "PCB_VIA" and t.GetNetCode() == b_nc:
            p = t.GetPosition()
            b_vias.append((pcbnew.ToMM(p.x), pcbnew.ToMM(p.y)))

    # 收集所有 B- 焊盘位置（F.Cu 和 B.Cu 都处理）
    target_pads: list[tuple[float, float, int, object]] = []
    for fp in board.GetFootprints():
        for pad in fp.Pads():
            if pad.GetNetCode() != b_nc:
                continue
            pad_layer = pad.GetLayer()
            if pad_layer not in (pcbnew.F_Cu, pcbnew.B_Cu):
                continue
            p = pad.GetPosition()
            target_pads.append((pcbnew.ToMM(p.x), pcbnew.ToMM(p.y), pad_layer, pad))

    if not target_pads:
        return 0

    # 按距离排序过孔，优先尝试近的
    connected = 0
    for px, py, pad_layer, pad in target_pads:
        # 检查焊盘同层是否已有 B- 走线端点
        has_track_same = False
        for t in board.GetTracks():
            if t.GetClass() != "PCB_TRACK" or t.GetNetCode() != b_nc:
                continue
            if t.GetLayer() != pad_layer:
                continue
            s, e = t.GetStart(), t.GetEnd()
            for pt in [s, e]:
                if math.hypot(pcbnew.ToMM(pt.x) - px, pcbnew.ToMM(pt.y) - py) < 0.5:
                    has_track_same = True
                    break
            if has_track_same:
                break
        if has_track_same:
            continue  # 已有连接，跳过

        # 按距离排序所有 B- 过孔
        sorted_vias = sorted(b_vias, key=lambda v: math.hypot(px - v[0], py - v[1]))

        track_added = False
        for vx, vy in sorted_vias[:5]:  # 最多尝试 5 个最近过孔
            dist = math.hypot(px - vx, py - vy)
            if dist < 0.3:
                continue  # 过孔太近，跳过
            if dist > 20.0:
                break

            # 检查直线是否穿越其他网络焊盘
            if _track_clears_nets(board, px, py, vx, vy, b_nc, pad_layer, clearance=0.35):
                # 路径清晰，直接画走线
                t = pcbnew.PCB_TRACK(board)
                t.SetStart(pcbnew.VECTOR2I(_mm(px), _mm(py)))
                t.SetEnd(pcbnew.VECTOR2I(_mm(vx), _mm(vy)))
                t.SetWidth(_mm(0.4))
                t.SetLayer(pad_layer)
                t.SetNetCode(b_nc)
                board.Add(t)
                connected += 1
                track_added = True
                break
            else:
                # 尝试 L 形走线（先水平再垂直 / 先垂直再水平）
                for corner in [(vx, py), (px, vy)]:
                    cx, cy = corner
                    seg1_ok = _track_clears_nets(board, px, py, cx, cy, b_nc, pad_layer, 0.35)
                    seg2_ok = _track_clears_nets(board, cx, cy, vx, vy, b_nc, pad_layer, 0.35)
                    if seg1_ok and seg2_ok:
                        t1 = pcbnew.PCB_TRACK(board)
                        t1.SetStart(pcbnew.VECTOR2I(_mm(px), _mm(py)))
                        t1.SetEnd(pcbnew.VECTOR2I(_mm(cx), _mm(cy)))
                        t1.SetWidth(_mm(0.4))
                        t1.SetLayer(pad_layer)
                        t1.SetNetCode(b_nc)
                        board.Add(t1)
                        t2 = pcbnew.PCB_TRACK(board)
                        t2.SetStart(pcbnew.VECTOR2I(_mm(cx), _mm(cy)))
                        t2.SetEnd(pcbnew.VECTOR2I(_mm(vx), _mm(vy)))
                        t2.SetWidth(_mm(0.4))
                        t2.SetLayer(pad_layer)
                        t2.SetNetCode(b_nc)
                        board.Add(t2)
                        connected += 1
                        track_added = True
                        break
                if track_added:
                    break

        if not track_added:
            # 所有路径都被阻挡 → 在焊盘位置直接放过孔（最短连接）
            via = pcbnew.PCB_VIA(board)
            via.SetPosition(pcbnew.VECTOR2I(_mm(px), _mm(py)))
            via.SetWidth(_mm(0.8))
            via.SetDrill(_mm(0.4))
            via.SetViaType(pcbnew.VIATYPE_THROUGH)
            via.SetNetCode(b_nc)
            board.Add(via)
            b_vias.append((px, py))
            connected += 1

    if connected:
        board.BuildConnectivity()
        print(f"[adapt_common] 连接 {connected} 个 B- 焊盘到过孔（保证地平面 zone 连通）")
    return connected


# ── 连接感知布局优化（模拟退火）────────────────────────────────────────

def _get_fp_pad_positions(fp) -> list[tuple[float, float, int]]:
    """获取封装所有焊盘的全局坐标 (x_mm, y_mm, net_code)。"""
    pads = []
    for pad in fp.Pads():
        p = pad.GetPosition()
        pads.append((pcbnew.ToMM(p.x), pcbnew.ToMM(p.y), pad.GetNetCode()))
    return pads


def _compute_layout_cost(board, outline_pts: list[tuple[float, float]]) -> tuple[float, float, float]:
    """计算布局综合成本：(hpwl, congestion_penalty, total_cost)。

    - hpwl: 所有网络半外围线长之和（曼哈顿距离）
    - congestion_penalty: 不同网络焊盘之间的间距惩罚（越近惩罚越重）
    - total_cost: hpwl + congestion_penalty * weight

    短路惩罚确保优化器不会将不同网络的焊盘挤在一起，
    为 Freerouting 留出足够的走线通道。
    """
    # 收集所有焊盘位置
    all_pads: list[tuple[float, float, int]] = []  # (x, y, net_code)
    net_pads: dict[int, list[tuple[float, float]]] = {}
    for fp in board.GetFootprints():
        for pad in fp.Pads():
            nc = pad.GetNetCode()
            if nc == 0:
                continue
            p = pad.GetPosition()
            px, py = pcbnew.ToMM(p.x), pcbnew.ToMM(p.y)
            all_pads.append((px, py, nc))
            net_pads.setdefault(nc, []).append((px, py))

    # HPWL
    hpwl = 0.0
    for pads in net_pads.values():
        if len(pads) < 2:
            continue
        xs = [p[0] for p in pads]
        ys = [p[1] for p in pads]
        hpwl += (max(xs) - min(xs)) + (max(ys) - min(ys))

    # 短路惩罚：检查不同网络焊盘之间的最小间距
    # 如果异网络焊盘间距 < min_clearance，产生惩罚
    min_clearance = 1.5  # mm，最小安全间距（含走线空间，保护板紧凑需保证足够间距）
    penalty = 0.0
    # 只检查不同网络之间的焊盘对（按网络分组减少计算量）
    net_list = list(net_pads.items())
    for i in range(len(net_list)):
        nc_a, pads_a = net_list[i]
        for j in range(i + 1, len(net_list)):
            nc_b, pads_b = net_list[j]
            # 计算两组焊盘之间的最小距离
            min_dist = float("inf")
            for ax, ay in pads_a:
                for bx, by in pads_b:
                    d = math.hypot(ax - bx, ay - by)
                    if d < min_dist:
                        min_dist = d
            # 如果最小距离小于安全间距，产生惩罚
            if min_dist < min_clearance:
                gap = min_clearance - min_dist
                # 惩罚与间距缺口的立方成正比（越近惩罚越重，几乎硬约束）
                penalty += gap * gap * gap * 500.0

    total = hpwl + penalty
    return hpwl, penalty, total


def _point_in_polygon_safe(px: float, py: float, polygon: list[tuple[float, float]]) -> bool:
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


def optimize_component_placement(
    board,
    outline_pts: list[tuple[float, float]],
    iterations: int = 300,
    seed: int = 42,
) -> int:
    """模拟退火布局优化，最小化总走线长度（HPWL）。

    针对电池保护板场景（1-5 串/并，10-15 个元件）：
      - 端子焊盘（TP_xxx）位置固定不动
      - 内部元件（U1/Q1/R1/C1...）可自由移动+旋转
      - 约束：板框内、不重叠
      - 目标：最小化所有网络的曼哈顿走线长度之和

    优化后 Freerouting 拿到的初始布局已是布线友好的，
    大幅减少短路/间距违规。

    Args:
        board: pcbnew.BOARD
        outline_pts: PCB 板框轮廓 [(x_mm, y_mm), ...]
        iterations: 退火迭代次数
        seed: 随机种子

    Returns:
        移动的元件数量
    """
    import random
    random.seed(seed)

    # 收集可动元件（排除端子焊盘）
    movable = []
    for fp in board.GetFootprints():
        ref = fp.GetReference()
        if ref.startswith("TP"):
            continue
        fpid = str(fp.GetFPID().GetLibItemName())
        if "TestPoint" in fpid or "PinHeader" in fpid:
            continue
        movable.append(fp)

    if not movable or len(outline_pts) < 3:
        return 0

    # 板框边界
    xs = [p[0] for p in outline_pts]
    ys = [p[1] for p in outline_pts]
    b_minx, b_maxx = min(xs), max(xs)
    b_miny, b_maxy = min(ys), max(ys)
    inset = 1.0  # 边距

    # 初始成本
    init_hpwl, init_penalty, init_cost = _compute_layout_cost(board, outline_pts)

    # 退火参数
    board_diag = math.hypot(b_maxx - b_minx, b_maxy - b_miny)
    T0 = board_diag * 0.15  # 初始温度：板框对角线的 15%
    T_min = 0.05  # 最小温度：0.05mm
    alpha = 0.96  # 冷却系数

    T = T0
    moved_count = 0
    improvements = 0

    for iteration in range(iterations):
        if T < T_min:
            break

        # 随机选择一个元件
        fp = random.choice(movable)
        p = fp.GetPosition()
        old_x, old_y = pcbnew.ToMM(p.x), pcbnew.ToMM(p.y)
        old_angle = fp.GetOrientation().AsDegrees()

        # 生成候选移动：幅度随温度降低
        move_scale = T * 0.5
        dx = random.gauss(0, move_scale)
        dy = random.gauss(0, move_scale)
        new_x = old_x + dx
        new_y = old_y + dy

        # 偶尔尝试旋转（20% 概率）
        do_rotate = random.random() < 0.2
        rot_delta = random.choice([-90, 90, 180]) if do_rotate else 0

        # 应用候选位置
        fp.SetPosition(pcbnew.VECTOR2I(_mm(new_x), _mm(new_y)))
        if do_rotate:
            fp.Rotate(fp.GetPosition(), pcbnew.EDA_ANGLE(rot_delta))

        # 约束检查：元件中心必须在板框内
        in_bounds = (b_minx + inset <= new_x <= b_maxx - inset and
                     b_miny + inset <= new_y <= b_maxy - inset)

        # 约束检查：不与其他元件重叠（包围盒检查 + 走线通道余量）
        overlap = False
        if in_bounds:
            fp_bb = fp.GetBoundingBox(False, False)
            fp_w = fp_bb.GetWidth() / 1e6 if fp_bb.GetWidth() else 2.0
            fp_h = fp_bb.GetHeight() / 1e6 if fp_bb.GetHeight() else 1.5
            # 走线通道余量：元件之间至少保留 1.5mm 走线空间
            route_margin = 1.5
            for other in movable:
                if other is fp:
                    continue
                op = other.GetPosition()
                ox, oy = pcbnew.ToMM(op.x), pcbnew.ToMM(op.y)
                obb = other.GetBoundingBox(False, False)
                ow = obb.GetWidth() / 1e6 if obb.GetWidth() else 2.0
                oh = obb.GetHeight() / 1e6 if obb.GetHeight() else 1.5
                # 包围盒重叠检查（含走线通道余量）
                need_x = (fp_w + ow) / 2 + route_margin
                need_y = (fp_h + oh) / 2 + route_margin
                if (abs(new_x - ox) < need_x and abs(new_y - oy) < need_y):
                    overlap = True
                    break

        # 计算新成本（HPWL + 短路惩罚）
        new_hpwl, new_penalty, new_cost = _compute_layout_cost(board, outline_pts)
        delta = new_cost - init_cost

        # 接受/拒绝
        accept = False
        if in_bounds and not overlap:
            if delta < 0:
                accept = True
            else:
                # Metropolis 准则：以一定概率接受较差解
                try:
                    if random.random() < math.exp(-delta / max(T, 0.01)):
                        accept = True
                except (OverflowError, ZeroDivisionError):
                    accept = False

        if accept:
            init_hpwl, init_penalty, init_cost = new_hpwl, new_penalty, new_cost
            moved_count += 1
            if delta < -0.01:
                improvements += 1
        else:
            # 恢复原位置和旋转
            fp.SetPosition(pcbnew.VECTOR2I(_mm(old_x), _mm(old_y)))
            if do_rotate:
                fp.Rotate(fp.GetPosition(), pcbnew.EDA_ANGLE(-rot_delta))

        # 冷却
        T *= alpha

    final_hpwl, final_penalty, final_cost = _compute_layout_cost(board, outline_pts)
    hpwl_reduction = ((init_hpwl - final_hpwl) / max(init_hpwl, 0.01)) * 100
    print(f"[adapt_common] 布局优化: {iterations} 轮退火, "
          f"{moved_count} 次移动, HPWL {init_hpwl:.1f}→{final_hpwl:.1f}mm "
          f"(↓{hpwl_reduction:.0f}%), 短路惩罚 {init_penalty:.1f}→{final_penalty:.1f}")
    return moved_count

