"""焊盘对齐验证测试脚本。

模拟 Web 前端操作流程前 4 步：
  1. 上传正反面照片（通过 /api/simulate 读取 input/*.jpg）
  2. 矫正预览（标定）
  3. 识别轮廓（extract-pcb）
  4. 一键识别孔槽/焊盘/元器件（仅焊盘）

测试条件:
  - 正面: ID 和 T 焊盘纵向对齐排列（相同 x 坐标）
  - 正面: 3 个 P+ 焊盘，纵向对齐 + 纵向均匀排列
  - 正面: 3 个 P- 焊盘，纵向对齐 + 纵向均匀排列
  - 正面: 3 个 P+ 与 3 个 P- 在同一列共线（x 坐标一致），且 P+ 全部在 P- 上方
  - 背面: 2 个 B+ 和 2 个 B- 焊盘呈 2x2 grid 排列

用法:
  .venv/Scripts/python.exe tests/test_pad_alignment.py

要求: battery_designer 服务已启动在 http://127.0.0.1:8000
"""
from __future__ import annotations

import json
import math
import sys
import time
from collections import defaultdict
from pathlib import Path

import httpx

BASE = "http://127.0.0.1:8000"
TIMEOUT = 300
TOLERANCE_MM = 0.8  # 对齐容差（mm）

FRAME_W_MM = 40.0
FRAME_H_MM = 25.0


# ── 工具函数 ──

def step(n: int, title: str):
    print(f"\n{'='*60}")
    print(f"  Step {n}: {title}")
    print(f"{'='*60}")


def ok(msg: str):
    print(f"  [OK] {msg}")


def fail(msg: str):
    print(f"  [FAIL] {msg}")


def info(msg: str):
    print(f"  [INFO] {msg}")


def warn(msg: str):
    print(f"  [WARN] {msg}")


def get_centers(candidates: list[dict], label: str) -> list[tuple[float, float]]:
    """提取指定 label 所有焊盘的 (x_mm, y_mm) 中心点列表。"""
    points: list[tuple[float, float]] = []
    for pad in candidates:
        if (pad.get("label", "") or "").upper().strip() == label.upper():
            region = pad.get("visible_region") or (pad.get("matched_regions") or [{}])[0]
            center = region.get("center", {})
            x = center.get("x_mm")
            y = center.get("y_mm")
            if x is not None and y is not None:
                points.append((x, y))
    return points


def get_sizes(candidates: list[dict], label: str) -> list[tuple[float, float]]:
    """提取指定 label 所有焊盘的 (width_mm, height_mm) 尺寸列表。"""
    sizes: list[tuple[float, float]] = []
    for pad in candidates:
        if (pad.get("label", "") or "").upper().strip() == label.upper():
            w = pad.get("width_mm")
            h = pad.get("height_mm")
            if w is not None and h is not None:
                sizes.append((w, h))
    return sizes


def check_same_x(points: list[tuple[float, float]], label: str) -> bool:
    """检查所有点是否共享相同 x 坐标（纵向对齐）。"""
    if len(points) < 2:
        return len(points) > 0
    xs = [p[0] for p in points]
    spread = max(xs) - min(xs)
    if spread <= TOLERANCE_MM:
        ok(f"  {label}: {len(points)} 个焊盘纵向对齐, x 范围 [{min(xs):.2f}, {max(xs):.2f}], 偏差={spread:.2f}mm")
        return True
    else:
        fail(f"  {label}: x 偏差过大 ({spread:.2f}mm > {TOLERANCE_MM}mm), 非纵向对齐")
        return False


def check_even_spacing(points: list[tuple[float, float]], label: str) -> bool:
    """检查 y 方向上是否均匀排列。"""
    if len(points) < 3:
        return len(points) >= 2
    sorted_pts = sorted(points, key=lambda p: p[1])
    gaps = [sorted_pts[i + 1][1] - sorted_pts[i][1] for i in range(len(sorted_pts) - 1)]
    avg_gap = sum(gaps) / len(gaps)
    max_deviation = max(abs(g - avg_gap) for g in gaps) if gaps else 0
    if avg_gap > 0 and max_deviation / avg_gap <= 0.25:
        ok(f"  {label}: {len(points)} 个焊盘纵向均匀排列, 间距={gaps}, 平均={avg_gap:.2f}mm, 最大偏差={max_deviation:.2f}mm ({max_deviation/avg_gap*100:.0f}%)")
        return True
    else:
        fail(f"  {label}: 间距不均匀, gaps={gaps}, avg={avg_gap:.2f}mm, max_dev={max_deviation:.2f}mm")
        return False


def check_horizontal_pairing(plus: list[tuple[float, float]], minus: list[tuple[float, float]],
                             label_p: str = "B+", label_m: str = "B-") -> bool:
    """检查两组焊盘是否逐对横向对齐（y 坐标匹配）。用于背面 B+/B- 2x2 grid 验证。"""
    if len(plus) != len(minus):
        fail(f"  {label_p}/{label_m}: 数量不匹配 ({len(plus)} vs {len(minus)})")
        return False
    p_sorted = sorted(plus, key=lambda p: p[1])
    m_sorted = sorted(minus, key=lambda p: p[1])
    all_ok = True
    for i, ((px, py), (mx, my)) in enumerate(zip(p_sorted, m_sorted)):
        dy = abs(py - my)
        if dy <= TOLERANCE_MM:
            ok(f"  {label_p}[{i}] <=> {label_m}[{i}]: y 对齐 (y={py:.2f} vs {my:.2f}, dy={dy:.2f}mm)")
        else:
            fail(f"  {label_p}[{i}] <=> {label_m}[{i}]: y 不对齐 (y={py:.2f} vs {my:.2f}, dy={dy:.2f}mm > {TOLERANCE_MM}mm)")
            all_ok = False
    return all_ok


def check_2x2_grid(b_plus: list[tuple[float, float]], b_minus: list[tuple[float, float]]) -> bool:
    """检查 B+ 和 B- 是否呈 2x2 grid 排列。

    2x2 grid 条件:
      - B+ 和 B- 各有 2 个
      - 同组内 x 对齐（纵向排列）
      - 两组间逐对 y 对齐（横向对齐）
    """
    all_ok = True
    if len(b_plus) != 2:
        fail(f"  B+: 期望 2 个，实际 {len(b_plus)} 个")
        all_ok = False
    if len(b_minus) != 2:
        fail(f"  B-: 期望 2 个，实际 {len(b_minus)} 个")
        all_ok = False
    if not all_ok:
        return False

    # B+ 同 x（纵向）
    bp_ok = check_same_x(b_plus, "B+")
    # B- 同 x（纵向）
    bm_ok = check_same_x(b_minus, "B-")

    # B+ 和 B- 横向对齐（同一行 y 一致）
    pair_ok = check_horizontal_pairing(b_plus, b_minus)

    all_ok = bp_ok and bm_ok and pair_ok
    if all_ok:
        ok("  背面焊盘: 2x2 grid 验证通过")
    else:
        fail("  背面焊盘: 2x2 grid 验证失败")
    return all_ok


# ── 主流程 ──

def main() -> int:
    t0 = time.time()
    client = httpx.Client(base_url=BASE, timeout=TIMEOUT)
    errors: list[str] = []
    all_results: list[bool] = []

    # ════════════════════════════════════════════════════════════
    # Step 1: 上传 + 矫正预览
    # ════════════════════════════════════════════════════════════
    step(1, "上传正反面照片 + 矫正预览 (/api/simulate)")
    try:
        resp = client.post("/api/simulate", data={
            "frame_w_mm": str(FRAME_W_MM),
            "frame_h_mm": str(FRAME_H_MM),
        })
        resp.raise_for_status()
        sim = resp.json()
        steps_data = sim.get("steps", [])
        cal_ids: dict[str, str] = {}
        for s in steps_data:
            side = s["side"]
            if s.get("calibration_success"):
                cal_ids[side] = s["calibration_id"]
                ok(f"{side}: 标定成功 (id={s['calibration_id'][:12]}..., ppm={s.get('pixels_per_mm', 0):.1f})")
            else:
                fail(f"{side}: 标定失败 - {s.get('calibration_error_msg', '未知错误')}")
                errors.append(f"Step1 {side} 标定失败")
        if len(cal_ids) < 2:
            fail("正反面标定未全部完成，后续流程无法继续")
            _summary(t0, errors)
            return 1
    except Exception as e:
        fail(f"请求异常: {e}")
        _summary(t0, [f"Step1 异常: {e}"])
        return 1

    # ════════════════════════════════════════════════════════════
    # Step 2: 识别轮廓
    # ════════════════════════════════════════════════════════════
    step(2, "识别轮廓 (extract-pcb)")
    outlines: dict[str, list] = {}
    try:
        for side in ["front", "back"]:
            t1 = time.time()
            resp = client.post("/api/vision/extract-pcb", data={
                "calibration_id": cal_ids[side],
            })
            resp.raise_for_status()
            data = resp.json()
            outline = data.get("outline", [])
            outlines[side] = outline
            ok(f"{side}: 轮廓 {len(outline)} 顶点 ({time.time()-t1:.1f}s)")
            if len(outline) < 3:
                fail(f"{side}: 轮廓顶点不足 3 个")
                errors.append(f"Step2 {side} 轮廓无效")
    except Exception as e:
        fail(f"请求异常: {e}")
        errors.append(f"Step2 异常: {e}")

    # 交叉校验
    if "front" in cal_ids and "back" in cal_ids and len(outlines.get("front", [])) >= 3:
        try:
            info("执行正反面交叉校验...")
            for side, other_side in [("front", "back"), ("back", "front")]:
                resp = client.post("/api/vision/extract-pcb", data={
                    "calibration_id": cal_ids[side],
                    "other_calibration_id": cal_ids[other_side],
                })
                if resp.status_code == 200:
                    data2 = resp.json()
                    if data2.get("outline"):
                        outlines[side] = data2["outline"]
            ok(f"交叉校验完成: front={len(outlines.get('front',[]))} pts, back={len(outlines.get('back',[]))} pts")
        except Exception as e:
            info(f"交叉校验跳过（非致命）: {e}")

    # ════════════════════════════════════════════════════════════
    # Step 3: 焊盘识别
    # ════════════════════════════════════════════════════════════
    step(3, "焊盘识别 (detect-terminals)")
    pads_data: dict[str, dict] = {}
    for side in ["front", "back"]:
        try:
            t1 = time.time()
            resp = client.post("/api/vision/detect-terminals", data={
                "calibration_id": cal_ids[side],
                "side": side,
                "debug": "true",
            })
            resp.raise_for_status()
            pads_data[side] = resp.json()
            candidates = pads_data[side].get("candidates", [])
            details = ", ".join(
                f"{c.get('label','?')}(x={c.get('matched_regions',[{}])[0].get('center',{}).get('x_mm','?'):.1f}, "
                f"y={c.get('matched_regions',[{}])[0].get('center',{}).get('y_mm','?'):.1f})"
                for c in candidates
            )
            ok(f"{side}: {len(candidates)} 个焊盘 ({time.time()-t1:.1f}s)")
            if details:
                info(f"  详情: {details}")
        except Exception as e:
            fail(f"{side} 焊盘检测失败: {e}")
            errors.append(f"Step3 {side} 焊盘: {e}")

    # ════════════════════════════════════════════════════════════
    # Step 3.5: 调试输出 —— 每一步的焊盘信息
    # ════════════════════════════════════════════════════════════
    print()
    print("=" * 80)
    print("  [DEBUG] 焊盘检测全流程分步数据")
    print("=" * 80)

    for side in ["front", "back"]:
        data = pads_data.get(side, {})
        if not data:
            continue
        stages = data.get("_debug_stages", [])
        if not stages:
            warn(f"  {side}: 无调试数据 (debug 模式未生效?)")
            continue

        print(f"\n{'─' * 70}")
        print(f"  [{side.upper()}] 板尺寸: {data.get('coordinate_system', {}).get('pcb_width_mm', '?')} × {data.get('coordinate_system', {}).get('pcb_height_mm', '?')} mm")
        print(f"  [{side.upper()}] 总共 {len(stages)} 个阶段")

        last_cands = None
        for s in stages:
            name = s["stage"]
            count = s["count"]
            cands = s["candidates"]

            print(f"\n  >>> {name} ({count} 个焊盘)")

            # Show pads sorted by y (with explicit x for reference)
            sorted_cands = sorted(cands, key=lambda c: (c.get("y_mm", 0) or 0))

            for c in sorted_cands:
                label = c.get("label", "?")
                x = c.get("x_mm")
                y = c.get("y_mm")
                w = c.get("width_mm")
                h = c.get("height_mm")
                conf = c.get("confidence")
                src = c.get("source", "")
                diag = c.get("diagnostic_verified", "")

                x_str = f"{x:.2f}" if x is not None else "?"
                y_str = f"{y:.2f}" if y is not None else "?"
                w_str = f"{w:.2f}" if w is not None else "?"
                h_str = f"{h:.2f}" if h is not None else "?"
                size_str = f"{w_str}×{h_str}" if (w is not None or h is not None) else "?×?"
                conf_str = f"{conf:.2f}" if conf is not None else "?"
                extra = []
                if src:
                    extra.append(src)
                if diag:
                    extra.append(diag)
                extra_str = f" [{', '.join(extra)}]" if extra else ""

                print(f"    {label:4s}  x={x_str:>7s}  y={y_str:>7s}  size={size_str:>11s}  conf={conf_str}{extra_str}")

            # Diff with previous stage (only for consecutive stages with same label count)
            if last_cands:
                # Compare pads of the same label
                prev_by_label = {}
                for pc in last_cands:
                    lb = pc.get("label", "")
                    prev_by_label.setdefault(lb, []).append(pc)

                curr_by_label = {}
                for cc in cands:
                    lb = cc.get("label", "")
                    curr_by_label.setdefault(lb, []).append(cc)

                diffs = []
                for lb in sorted(set(list(prev_by_label.keys()) + list(curr_by_label.keys()))):
                    prevs = prev_by_label.get(lb, [])
                    currs = curr_by_label.get(lb, [])
                    if len(prevs) != len(currs):
                        diffs.append(f"{lb}: {len(prevs)}→{len(currs)}个")
                    elif prevs and currs:
                        max_dx, max_dy = 0.0, 0.0
                        for pc, cc in zip(sorted(prevs, key=lambda p: p.get("y_mm") or 0),
                                          sorted(currs, key=lambda p: p.get("y_mm") or 0)):
                            dx = abs((cc.get("x_mm") or 0) - (pc.get("x_mm") or 0))
                            dy = abs((cc.get("y_mm") or 0) - (pc.get("y_mm") or 0))
                            max_dx = max(max_dx, dx)
                            max_dy = max(max_dy, dy)
                        if max_dx > 0.01 or max_dy > 0.01:
                            diffs.append(f"{lb}: maxΔx={max_dx:.2f}, maxΔy={max_dy:.2f}mm")
                if diffs:
                    print(f"    → 变化: {'; '.join(diffs)}")
                else:
                    print(f"    → 无变化")

            last_cands = cands

    print(f"\n{'─' * 70}")
    print("  [DEBUG END]")
    print("=" * 80)
    print()

    # ════════════════════════════════════════════════════════════
    # 焊盘有效性验证（边界零容差 + AI 视觉自检）
    # ════════════════════════════════════════════════════════════
    print(f"\n{'='*60}")
    print("  [VALIDATION] 焊盘有效性验证: PCB 边界(零容差) + AI 视觉自检")
    print(f"{'='*60}")
    validation_errors: list[str] = []

    for side in ["front", "back"]:
        data = pads_data.get(side, {})
        if not data:
            continue
        candidates = data.get("candidates", [])
        cs = data.get("coordinate_system", {})
        pcb_w = cs.get("pcb_width_mm", 0)
        pcb_h = cs.get("pcb_height_mm", 0)

        # ── 3.6a 每个焊盘的边界验证 (零容差) ──
        print(f"\n  [{side.upper()}] -- PCB 边界验证 (PCB {pcb_w:.1f}x{pcb_h:.1f}mm, 零容差)")
        boundary_ok = True
        for pad in candidates:
            label = pad.get("label", "?")
            vp = pad.get("visible_position", {})
            cx = vp.get("x_mm")
            cy = vp.get("y_mm")

            # Check center — 零容差: 任何超出 0mm 即失败
            if cx is not None and cy is not None:
                if cx < 0 or cx > pcb_w or cy < 0 or cy > pcb_h:
                    fail(f"  {side}/{label}: 中心 ({cx:.2f}, {cy:.2f}) 超出 PCB ({pcb_w:.1f}x{pcb_h:.1f}) 边界!")
                    boundary_ok = False
                    validation_errors.append(f"边界越界: {side}/{label} 中心({cx:.2f},{cy:.2f})")

            # Check polygon vertices — 零容差
            regions = pad.get("matched_regions", [])
            if regions:
                poly = regions[0].get("polygon") or []
                for vi, pt in enumerate(poly):
                    px = pt.get("x_mm")
                    py = pt.get("y_mm")
                    if px is not None and py is not None:
                        if px < 0 or px > pcb_w or py < 0 or py > pcb_h:
                            fail(f"  {side}/{label}: 顶点[{vi}] ({px:.2f}, {py:.2f}) 超出 PCB 边界!")
                            boundary_ok = False
                            validation_errors.append(f"顶点越界: {side}/{label}[{vi}] ({px:.2f},{py:.2f})")

        if boundary_ok:
            ok(f"  {side}: 所有 {len(candidates)} 个焊盘边界验证通过 (零容差)")
        all_results.append(boundary_ok)

        # ── 3.6b AI 焊盘视觉自检 (VLM 判断焊盘裁切区域是否为真实焊盘) ──
        print(f"\n  [{side.upper()}] -- AI 视觉自检 (VLM 判断焊盘区域真实性)")
        print(f"  [INFO] 调用 VLM 逐个验证，耗时较长 (每个焊盘约 2-5s)...")

        calib_id = cal_ids.get(side)
        if not calib_id:
            fail(f"  {side}: 缺少 calibration_id，跳过视觉自检")
            all_results.append(False)
            continue

        visual_ok = True
        try:
            r = httpx.post(
                f"{BASE}/api/vision/verify-pad-regions",
                data={"calibration_id": calib_id, "side": side},
                timeout=TIMEOUT,
            )
            if not r.is_success:
                fail(f"  {side}: VLM 视觉自检 API 调用失败 (HTTP {r.status_code})")
                visual_ok = False
            else:
                vdata = r.json()
                verified = vdata.get("verified", 0)
                failed = vdata.get("failed", 0)
                results = vdata.get("results", [])

                for v in results:
                    label = v.get("label", "?")
                    if v.get("ok"):
                        info(f"    {label}: [OK] (VLM 置信度 {v.get('confidence', 0):.2f})")
                    else:
                        issues = v.get("issues", [])
                        fail(f"    {label}: [FAIL] VLM 判定不合格 — {issues}")
                        validation_errors.append(f"AI视觉: {side}/{label}: {issues}")
                        visual_ok = False

                if failed == 0:
                    ok(f"  {side}: 全部 {verified} 个焊盘 AI 视觉自检通过")
                else:
                    fail(f"  {side}: {failed}/{verified + failed} 个焊盘 AI 视觉自检未通过!")
        except Exception as exc:
            fail(f"  {side}: VLM 视觉自检异常: {exc}")
            visual_ok = False

        all_results.append(visual_ok)

    if not validation_errors:
        ok("所有焊盘 PCB 边界 + AI 视觉自检通过")
    else:
        fail(f"共 {len(validation_errors)} 个问题: " + "; ".join(validation_errors[:5]))
    print()

    # ════════════════════════════════════════════════════════════
    # Step 4: 焊盘对齐验证
    # ════════════════════════════════════════════════════════════
    step(4, "焊盘对齐验证")

    # ── 正面验证 ──
    front_candidates = pads_data.get("front", {}).get("candidates", [])
    if not front_candidates:
        fail("正面无焊盘数据，跳过验证")
        _summary(t0, errors + ["无正面焊盘数据"])
        return 1

    print("\n  [正面焊盘对齐验证]")
    print("  " + "-" * 50)

    # 4.1 ID + T 纵向对齐
    info("4.1 ID 和 T 焊盘纵向对齐")
    id_points = get_centers(front_candidates, "ID")
    t_points = get_centers(front_candidates, "T")
    # 也尝试 TH（某些情况下 T 焊盘标记为 TH）
    if not t_points:
        t_points = get_centers(front_candidates, "TH")

    id_t_ok = True
    if not id_points:
        warn("  正面未识别到 ID 焊盘，跳过 ID 验证")
        id_t_ok = False
    else:
        id_t_ok = check_same_x(id_points, "ID") and id_t_ok

    if not t_points:
        warn("  正面未识别到 T/TH 焊盘，跳过 T 验证")
        id_t_ok = False
    else:
        id_t_ok = check_same_x(t_points, "T") and id_t_ok

    # ID 和 T 应该在同一个 x 坐标（纵向对齐）
    if id_points and t_points:
        dx = abs(id_points[0][0] - t_points[0][0])
        if dx <= TOLERANCE_MM:
            ok(f"  ID <=> T: 纵向对齐 (ID x={id_points[0][0]:.2f}, T x={t_points[0][0]:.2f}, dx={dx:.2f}mm)")
        else:
            fail(f"  ID <=> T: 纵向不对齐 (ID x={id_points[0][0]:.2f}, T x={t_points[0][0]:.2f}, dx={dx:.2f}mm > {TOLERANCE_MM}mm)")
            id_t_ok = False
    all_results.append(id_t_ok)

    # 4.2 P+ 焊盘验证
    print()
    info("4.2 正面 P+ 焊盘验证（3 个 + 纵向对齐 + 纵向均匀）")
    p_plus_points = get_centers(front_candidates, "P+")
    p_plus_ok = True
    if len(p_plus_points) != 3:
        fail(f"  P+: 期望 3 个，实际 {len(p_plus_points)} 个")
        p_plus_ok = False
    else:
        p_plus_ok = check_same_x(p_plus_points, "P+") and p_plus_ok
        p_plus_ok = check_even_spacing(p_plus_points, "P+") and p_plus_ok
    all_results.append(p_plus_ok)

    # 4.3 P- 焊盘验证
    print()
    info("4.3 正面 P- 焊盘验证（3 个 + 纵向对齐 + 纵向均匀）")
    p_minus_points = get_centers(front_candidates, "P-")
    p_minus_ok = True
    if len(p_minus_points) != 3:
        fail(f"  P-: 期望 3 个，实际 {len(p_minus_points)} 个")
        p_minus_ok = False
    else:
        p_minus_ok = check_same_x(p_minus_points, "P-") and p_minus_ok
        p_minus_ok = check_even_spacing(p_minus_points, "P-") and p_minus_ok
    all_results.append(p_minus_ok)

    # 4.4 P+ 和 P- 同列共线 + 尺寸一致 + P+ 在上、P- 在下
    print()
    info("4.4 正面 P+ 和 P- 同列共线（x 一致），尺寸一致，P+ 全部在 P- 上方")
    p_p_col_ok = True
    if p_plus_ok and p_minus_ok:
        # 检查所有 P+/P- 是否在同一列（x 一致）
        all_pp = p_plus_points + p_minus_points
        xs = [p[0] for p in all_pp]
        x_spread = max(xs) - min(xs)
        if x_spread <= TOLERANCE_MM:
            ok(f"  P+ & P-: {len(all_pp)} 个焊盘同一列共线, x 范围 [{min(xs):.2f}, {max(xs):.2f}], 偏差={x_spread:.2f}mm")
        else:
            fail(f"  P+ & P-: x 偏差过大 ({x_spread:.2f}mm > {TOLERANCE_MM}mm)")
            p_p_col_ok = False

        # 检查 P+ 全部在 P- 上方（P+ max y < P- min y）
        p_plus_ys = [p[1] for p in p_plus_points]
        p_minus_ys = [p[1] for p in p_minus_points]
        if max(p_plus_ys) < min(p_minus_ys):
            ok(f"  P+ 在上(y=[{min(p_plus_ys):.2f},{max(p_plus_ys):.2f}]), P- 在下(y=[{min(p_minus_ys):.2f},{max(p_minus_ys):.2f}])")
        else:
            fail(f"  P+ 与 P- y 范围重叠 (P+ y=[{min(p_plus_ys):.2f},{max(p_plus_ys):.2f}], P- y=[{min(p_minus_ys):.2f},{max(p_minus_ys):.2f}])")
            p_p_col_ok = False

        # 检查焊盘尺寸一致性（所有 P+ 和 P- 的宽高应一致）
        p_plus_sizes = get_sizes(front_candidates, "P+")
        p_minus_sizes = get_sizes(front_candidates, "P-")
        all_sizes = p_plus_sizes + p_minus_sizes
        all_labels = ["P+"] * len(p_plus_sizes) + ["P-"] * len(p_minus_sizes)

        if all_sizes:
            widths = [s[0] for s in all_sizes]
            heights = [s[1] for s in all_sizes]
            w_min, w_max = min(widths), max(widths)
            h_min, h_max = min(heights), max(heights)
            w_spread = w_max - w_min
            h_spread = h_max - h_min

            # Use 15% relative tolerance for size (VLM polygon imprecision)
            avg_w = sum(widths) / len(widths) if widths else 0
            avg_h = sum(heights) / len(heights) if heights else 0
            size_tol = 0.15  # 15% relative tolerance
            w_ok = w_spread <= max(avg_w * size_tol, TOLERANCE_MM * 0.5)
            h_ok = h_spread <= max(avg_h * size_tol, TOLERANCE_MM * 0.5)

            detail_parts = []
            for label, (w, h_) in zip(all_labels, all_sizes):
                detail_parts.append(f"{label}[{w:.2f}×{h_:.2f}]")
            detail = " ".join(detail_parts)

            if w_ok and h_ok:
                ok(f"  P+/P- 焊盘尺寸一致: avg={avg_w:.2f}×{avg_h:.2f}mm, w偏差={w_spread:.2f}mm, h偏差={h_spread:.2f}mm | {detail}")
            else:
                issues = []
                if not w_ok:
                    issues.append(f"宽度不一致 ({w_min:.2f}~{w_max:.2f}, spread={w_spread:.2f}mm)")
                if not h_ok:
                    issues.append(f"高度不一致 ({h_min:.2f}~{h_max:.2f}, spread={h_spread:.2f}mm)")
                fail(f"  P+/P- 焊盘尺寸不一致: {'; '.join(issues)} | {detail}")
                p_p_col_ok = False
        else:
            warn("  P+/P- 无尺寸数据，跳过尺寸一致性验证")
    else:
        warn("  P+ 或 P- 数量不符，跳过共线验证")
        p_p_col_ok = False
    all_results.append(p_p_col_ok)

    # ── 背面验证 ──
    back_candidates = pads_data.get("back", {}).get("candidates", [])
    print()
    print("\n  [背面焊盘对齐验证]")
    print("  " + "-" * 50)

    if not back_candidates:
        fail("背面无焊盘数据，跳过验证")
        all_results.append(False)
    else:
        info("4.5 背面 B+/B- 焊盘 2x2 grid 验证")
        b_plus_points = get_centers(back_candidates, "B+")
        b_minus_points = get_centers(back_candidates, "B-")
        back_ok = check_2x2_grid(b_plus_points, b_minus_points)
        all_results.append(back_ok)

    # ════════════════════════════════════════════════════════════
    # 汇总
    # ════════════════════════════════════════════════════════════
    _summary(t0, errors)
    passed = all(r for r in all_results)
    if passed:
        print("  [RESULT] 所有焊盘对齐验证通过!")
    else:
        print("  [RESULT] 部分焊盘对齐验证未通过，请检查上方 [FAIL] 信息")
    return 0 if passed else 1


def _summary(t0: float, errors: list[str]):
    elapsed = time.time() - t0
    print(f"\n{'='*60}")
    print(f"  测试完成 (总耗时 {elapsed:.1f}s)")
    print(f"{'='*60}")
    if errors:
        print(f"\n  [WARN] 共 {len(errors)} 个流程错误:")
        for i, e in enumerate(errors, 1):
            print(f"     {i}. {e}")
    print()


if __name__ == "__main__":
    sys.exit(main())
