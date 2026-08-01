"""完整流程端到端测试脚本。

模拟 Web 前端的全部操作流程：
  1. 上传正反面照片（通过 /api/simulate 读取 input/*.jpg）
  2. 矫正预览（标定）
  3. 识别轮廓（extract-pcb）
  4. 一键识别孔槽/焊盘/元器件
  5. 电芯参数查询
  6. IC 解析
  7. 生成 KiCad 工程文件

用法:
  .venv/Scripts/python.exe scripts/test_full_flow.py

输出: 每步结果 + 最终生成文件路径
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import httpx

BASE = "http://127.0.0.1:8000"
TIMEOUT = 300  # 单个请求最长等待（秒）

# ── 测试参数 ──
CELL_MANUFACTURER = "亿纬锂能"
CELL_MODEL = "INR18650-26V"
IC_MODEL = "3M1B/N607"
MOS_MODEL = "8205A"
FRAME_W_MM = 60.0
FRAME_H_MM = 30.0


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


def main() -> int:
    t0 = time.time()
    client = httpx.Client(base_url=BASE, timeout=TIMEOUT)
    errors: list[str] = []

    # ── Step 1: 上传 + 矫正预览（/api/simulate）──
    step(1, "上传正反面照片 + 矫正预览")
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
            _summary(t0, errors, None)
            return 1
    except Exception as e:
        fail(f"请求异常: {e}")
        _summary(t0, [f"Step1 异常: {e}"], None)
        return 1

    # ── Step 2: 识别轮廓（extract-pcb）──
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
            grooves = data.get("grooves", [])
            ok(f"{side}: 轮廓 {len(outline)} 顶点, {len(grooves)} 槽口 ({time.time()-t1:.1f}s)")
            if len(outline) < 3:
                fail(f"{side}: 轮廓顶点不足 3 个")
                errors.append(f"Step2 {side} 轮廓无效")
    except Exception as e:
        fail(f"请求异常: {e}")
        errors.append(f"Step2 异常: {e}")

    # 交叉校验（可选，pass 2）
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

    # ── Step 3: 一键识别（孔槽 + 焊盘 + 元器件）──
    step(3, "一键识别孔槽/焊盘/元器件")
    pads_data: dict[str, dict] = {}
    components_data: dict[str, dict] = {}
    holes_data: dict[str, dict] = {}

    for side in ["front", "back"]:
        if side not in cal_ids:
            continue
        # 孔槽
        try:
            t1 = time.time()
            outline_json = json.dumps({"outline": outlines.get(side, [])})
            resp = client.post("/api/vision/detect-holes", data={
                "calibration_id": cal_ids[side],
                "outline_json": outline_json,
            })
            resp.raise_for_status()
            holes_data[side] = resp.json()
            hc = holes_data[side].get("hole_count", len(holes_data[side].get("holes", [])))
            ok(f"{side} 孔槽: {hc} 个 ({time.time()-t1:.1f}s)")
        except Exception as e:
            fail(f"{side} 孔槽检测失败: {e}")
            errors.append(f"Step3 {side} 孔槽: {e}")

        # 焊盘
        try:
            t1 = time.time()
            resp = client.post("/api/vision/detect-terminals", data={
                "calibration_id": cal_ids[side],
                "side": side,
            })
            resp.raise_for_status()
            pads_data[side] = resp.json()
            pc = pads_data[side].get("candidate_count", len(pads_data[side].get("candidates", [])))
            ok(f"{side} 焊盘: {pc} 个 ({time.time()-t1:.1f}s)")
        except Exception as e:
            fail(f"{side} 焊盘检测失败: {e}")
            errors.append(f"Step3 {side} 焊盘: {e}")

        # 元器件
        try:
            t1 = time.time()
            resp = client.post("/api/vision/detect-components", data={
                "calibration_id": cal_ids[side],
                "side": side,
            })
            resp.raise_for_status()
            components_data[side] = resp.json()
            cc = len(components_data[side].get("components", []))
            ok(f"{side} 元器件: {cc} 个 ({time.time()-t1:.1f}s)")
        except Exception as e:
            fail(f"{side} 元器件检测失败: {e}")
            errors.append(f"Step3 {side} 元器件: {e}")

    # ── Step 4: 电芯参数查询 ──
    step(4, f"电芯参数查询 ({CELL_MANUFACTURER} {CELL_MODEL})")
    cell_params = None
    try:
        t1 = time.time()
        resp = client.post("/api/cell/lookup", json={
            "manufacturer": CELL_MANUFACTURER,
            "model": CELL_MODEL,
        })
        resp.raise_for_status()
        cell_params = resp.json()
        ok(f"电芯: {cell_params.get('manufacturer','')} {cell_params.get('model','')} "
           f"({cell_params.get('nominal_capacity_mah','')}mAh, {cell_params.get('nominal_voltage_v','')}V) "
           f"({time.time()-t1:.1f}s)")
    except Exception as e:
        fail(f"电芯查询失败: {e}")
        errors.append(f"Step4 电芯: {e}")

    # ── Step 5: IC 解析 ──
    step(5, f"IC 解析 ({IC_MODEL})")
    ic_device = None
    try:
        t1 = time.time()
        resp = client.get("/api/ic/resolve", params={"model": IC_MODEL})
        resp.raise_for_status()
        ic_device = resp.json()
        ok(f"IC: {ic_device.get('full_mpn','')} ({ic_device.get('manufacturer','')}) "
           f"封装={ic_device.get('package','')} ({time.time()-t1:.1f}s)")
    except Exception as e:
        fail(f"IC 解析失败: {e}")
        errors.append(f"Step5 IC: {e}")

    # ── Step 6: 构建 DesignSpec + 创建项目 ──
    step(6, "创建项目")
    project_id = None
    try:
        # 构建 terminals（与前端 buildTerminals 逻辑一致）
        LABEL_ROLES = {
            "B+": {"roles": ["battery"], "polarity": "positive"},
            "B-": {"roles": ["battery"], "polarity": "negative"},
            "P+": {"roles": ["charge", "discharge"], "polarity": "positive"},
            "P-": {"roles": ["charge", "discharge"], "polarity": "negative"},
            "C+": {"roles": ["charge"], "polarity": "positive"},
            "C-": {"roles": ["charge"], "polarity": "negative"},
            "TH": {"roles": ["temperature"], "polarity": None},
            "NTC": {"roles": ["temperature"], "polarity": None},
            "ID": {"roles": ["identification"], "polarity": None},
        }
        terminals = []
        idx = 0
        for side in ["front", "back"]:
            pd = pads_data.get(side, {})
            candidates = pd.get("candidates", [])
            off = pd.get("coordinate_system", {}).get("crop_offset_mm", {"x": 0, "y": 0})
            for pad in candidates:
                label = (pad.get("label", "") or "").upper().strip()
                mapping = LABEL_ROLES.get(label)
                if not mapping:
                    continue
                region = pad.get("visible_region") or (pad.get("matched_regions") or [{}])[0]
                center = region.get("center", {})
                poly = region.get("polygon", [])
                cx = center.get("x_mm")
                cy = center.get("y_mm")
                if cx is None or cy is None:
                    continue
                # 宽高
                w, h = 2.0, 2.0
                if len(poly) >= 3:
                    xs = [p["x_mm"] for p in poly]
                    ys = [p["y_mm"] for p in poly]
                    w = max(max(xs) - min(xs), 0.5)
                    h = max(max(ys) - min(ys), 0.5)
                idx += 1
                terminals.append({
                    "id": f"T{idx}_{label.replace('+','P').replace('-','N')}",
                    "position": {"x_mm": cx + off.get("x", 0), "y_mm": cy + off.get("y", 0)},
                    "roles": mapping["roles"],
                    "polarity": mapping["polarity"],
                    "side": side,
                    "shape": "rect",
                    "width_mm": min(w, 50),
                    "height_mm": min(h, 50),
                    "source_region": {
                        "type": "solder_pad",
                        "visual_class": "pad",
                        "shape": "rect",
                        "center": {"x_mm": cx + off.get("x", 0), "y_mm": cy + off.get("y", 0)},
                        "bbox": {
                            "x_mm": cx + off.get("x", 0) - w / 2,
                            "y_mm": cy + off.get("y", 0) - h / 2,
                            "width_mm": w, "height_mm": h,
                        },
                        "polygon": [{"x_mm": p["x_mm"] + off.get("x", 0),
                                     "y_mm": p["y_mm"] + off.get("y", 0)} for p in poly],
                        "source": "vlm",
                    },
                })

        # 合并元器件
        detected_components = []
        for side in ["front", "back"]:
            cd = components_data.get(side, {})
            for c in cd.get("components", []):
                detected_components.append({
                    "type": c.get("type", "other"),
                    "silkscreen": c.get("silkscreen", ""),
                    "package": c.get("package", ""),
                    "confidence": c.get("confidence", 0.5),
                })

        ic_mpn = ic_device.get("full_mpn", IC_MODEL) if ic_device else IC_MODEL
        outline_pts = outlines.get("front", [])

        spec = {
            "name": f"抄板_{ic_mpn}_1S_{time.strftime('%Y-%m-%d')}",
            "protection_ic": ic_mpn,
            "battery": {"count": 1, "connection": "series", "battery_type": "18650"},
            "mos_count": 2,
            "mos_mpn": MOS_MODEL,
            "outline": {"points": outline_pts, "source": "photo", "confirmed": True},
            "terminals": terminals,
            "photo_capture": {
                "front_calibration_id": cal_ids.get("front"),
                "back_calibration_id": cal_ids.get("back"),
            },
            "detected_components": detected_components,
        }

        resp = client.post("/api/projects", json=spec)
        resp.raise_for_status()
        proj = resp.json()
        project_id = proj.get("id") or proj.get("project_id")
        ok(f"项目已创建: {project_id}")
        info(f"端口拓扑: {proj.get('port_topology', 'unknown')}")
        if proj.get("directory"):
            info(f"项目目录: {proj['directory']}")
    except Exception as e:
        fail(f"创建项目失败: {e}")
        errors.append(f"Step6 创建项目: {e}")
        _summary(t0, errors, None)
        return 1

    # ── Step 7: 生成 KiCad 工程文件 ──
    step(7, "生成 KiCad 工程文件 (manufacturing)")
    output_dir = None
    try:
        t1 = time.time()
        resp = client.post(f"/api/projects/{project_id}/manufacturing")
        # 处理候选模板审批
        if resp.status_code != 200:
            err_data = resp.json() if resp.headers.get("content-type", "").startswith("application/json") else {}
            if (err_data.get("error", {}).get("code", "") == "CANDIDATE_APPROVAL_REQUIRED"):
                info("候选模板需要审批，自动审批中...")
                client.post(f"/api/projects/{project_id}/approve-candidate")
                resp = client.post(f"/api/projects/{project_id}/manufacturing")

        if resp.status_code != 200:
            err_data = resp.json() if resp.headers.get("content-type", "").startswith("application/json") else {}
            err_msg = err_data.get("error", {}).get("message", resp.text[:200])
            fail(f"制造文件生成失败: {err_msg}")
            errors.append(f"Step7 manufacturing: {err_msg}")
        else:
            mdata = resp.json()
            files = mdata.get("manifest", {}).get("files", [])
            ok(f"生成完成! 共 {len(files)} 个文件 ({time.time()-t1:.1f}s)")
            # 输出文件列表
            print(f"\n  [FILES] 生成文件清单:")
            for f in files:
                print(f"     - {f.get('path', f)}")
            if mdata.get("package"):
                print(f"\n  [ZIP] 完整包: {mdata['package']}")
            # 推断输出目录
            output_dir = Path("work/projects") / project_id / "output" / "kicad"
            info(f"KiCad 输出目录: {output_dir.resolve()}")
    except Exception as e:
        fail(f"生成异常: {e}")
        errors.append(f"Step7 异常: {e}")

    # ── 汇总 ──
    _summary(t0, errors, output_dir)
    return 1 if errors else 0


def _summary(t0: float, errors: list[str], output_dir: Path | None):
    elapsed = time.time() - t0
    print(f"\n{'='*60}")
    print(f"  测试完成 (总耗时 {elapsed:.1f}s)")
    print(f"{'='*60}")
    if errors:
        print(f"\n  [WARN] 共 {len(errors)} 个错误:")
        for i, e in enumerate(errors, 1):
            print(f"     {i}. {e}")
    else:
        print(f"\n  [PASS] 全部通过！")
    if output_dir and output_dir.exists():
        print(f"\n  [DIR] KiCad 工程文件路径:")
        print(f"     {output_dir.resolve()}")
        pcb_files = list(output_dir.glob("*.kicad_pcb"))
        for f in pcb_files:
            print(f"     [PCB] {f.resolve()}")
    print()


if __name__ == "__main__":
    sys.exit(main())
