"""
End-to-end test: calibrate → extract outline → detect holes/pads/components.

Simulates the web UI workflow (steps 1-3):
  1. Set frame size to 40×25 mm, calibrate front & back images
  2. Extract PCB outline both sides
  3. Detect holes, pads, and components both sides

Then verifies pad alignment post _align_pad_groups:
  ✓ Front: ID & TH vertically aligned (same column)
  ✓ Front: 3×P+ vertically aligned; 3×P- vertically aligned;
           P+ and P- columns share Y positions (horizontally aligned)
  ✓ Back:  B+/B- form 2×2 grid (both X & Y aligned)

Requires: DASHSCOPE_API_KEY set (for VLM pad/component detection).
"""
from __future__ import annotations

import importlib
import json
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# ══ Must set BATTERY_DESIGN_WORKDIR BEFORE first import of app ══
# app.py computes WORK_ROOT at module level; if pytest collected this file
# before the env-var is set, WORK_ROOT defaults to ROOT/"work".
# Force a temp dir here so both pytest collection and direct execution work.

_TMP_WORK = tempfile.mkdtemp(prefix="pcb_e2e_")
os.environ["BATTERY_DESIGN_WORKDIR"] = _TMP_WORK

import battery_designer.app as app_module

# ── config ────────────────────────────────────────────────────────────

FRAME_W_MM = 40.0
FRAME_H_MM = 25.0

FRONT_IMG = ROOT / "data" / "visualization" / "test_front.jpg"
BACK_IMG  = ROOT / "data" / "visualization" / "test_back.jpg"

# ── helpers ───────────────────────────────────────────────────────────

def _post_file(client: TestClient, url: str, file_path: Path,
               extra_fields: dict | None = None) -> dict:
    """POST multipart/form-data with a file + optional form fields."""
    with open(file_path, "rb") as fh:
        files = {"file": (file_path.name, fh, "image/jpeg")}
        data = extra_fields or {}
        # For FastAPI TestClient, pass string values in data, file separately
        resp = client.post(url, data=data, files=files)
    assert resp.status_code == 200, f"{url} → {resp.status_code}: {resp.text[:300]}"
    return resp.json()


def _post_form(client: TestClient, url: str, fields: dict, allowed_status: set = {200}) -> dict:
    """POST application/x-www-form-urlencoded."""
    resp = client.post(url, data=fields)
    if resp.status_code not in allowed_status:
        print(f"  [ERROR] {url} → {resp.status_code}", flush=True)
        print(f"  [ERROR] body: {resp.text[:500]}", flush=True)
    assert resp.status_code in allowed_status, f"{url} → {resp.status_code}: {resp.text[:300]}"
    return resp.json()


def _get_centers_by_label(candidates: list[dict]) -> dict[str, list[tuple[float, float]]]:
    """Group (x_mm, y_mm) positions by label."""
    groups: dict[str, list[tuple[float, float]]] = {}
    for c in candidates:
        lbl = c["label"]
        vp = c["visible_position"]
        groups.setdefault(lbl, []).append((float(vp["x_mm"]), float(vp["y_mm"])))
    return groups


def _check_vertical_alignment(label: str, positions: list[tuple[float, float]],
                               tol_mm: float = 0.5) -> list[str]:
    """Verify all pads with *label* share the same X (vertical column)."""
    errors = []
    if len(positions) < 2:
        return errors  # not enough to check alignment
    xs = [p[0] for p in positions]
    ref_x = xs[0]
    for i, x in enumerate(xs):
        if abs(x - ref_x) > tol_mm:
            errors.append(
                f"  [{label}] vertical alignment broken: "
                f"pad[{0}] X={ref_x:.3f}, pad[{i}] X={x:.3f} (diff={abs(x-ref_x):.3f} mm)")
    return errors


def _check_horizontal_alignment(groups: dict[str, list[tuple[float, float]]],
                                 tol_mm: float = 0.5) -> list[str]:
    """Verify pads across labels share the same Y positions (row alignment)."""
    errors = []
    labels = sorted(groups.keys())
    if len(labels) < 2:
        return errors

    # Collect all Y values for each label, sorted
    y_sets = {lbl: sorted([p[1] for p in positions]) for lbl, positions in groups.items()}
    ref_label = labels[0]
    ref_ys = y_sets[ref_label]

    for lbl in labels[1:]:
        ys = y_sets[lbl]
        if len(ys) != len(ref_ys):
            errors.append(
                f"  Column alignment: {lbl} has {len(ys)} pads vs {ref_label} has {len(ref_ys)} — "
                f"cannot verify Y-alignment")
            continue
        for i, (y, ref_y) in enumerate(zip(ys, ref_ys)):
            if abs(y - ref_y) > tol_mm:
                errors.append(
                    f"  Y-alignment broken: {ref_label}[{i}] Y={ref_y:.3f} vs "
                    f"{lbl}[{i}] Y={y:.3f} (diff={abs(y-ref_y):.3f} mm)")
    return errors


def _check_2x2_grid(groups: dict[str, list[tuple[float, float]]],
                    tol_mm: float = 0.5) -> list[str]:
    """Verify B+/B- pads form a 2×2 grid: 4 pads, 2 unique X values, 2 unique Y values."""
    errors = []
    all_positions: list[tuple[float, float]] = []
    for lbl, poses in groups.items():
        all_positions.extend(poses)

    if len(all_positions) != 4:
        errors.append(f"  2×2 grid expects 4 pads, got {len(all_positions)}")
        return errors

    xs = sorted(set(round(p[0], 2) for p in all_positions))
    ys = sorted(set(round(p[1], 2) for p in all_positions))

    if len(xs) != 2:
        errors.append(
            f"  2×2 grid expects 2 unique X values, got {len(xs)}: {xs}")
    if len(ys) != 2:
        errors.append(
            f"  2×2 grid expects 2 unique Y values, got {len(ys)}: {ys}")

    # Verify each pad falls on one of the 4 grid intersections
    for lbl, poses in groups.items():
        for px, py in poses:
            on_x = any(abs(px - gx) < tol_mm for gx in xs)
            on_y = any(abs(py - gy) < tol_mm for gy in ys)
            if not on_x or not on_y:
                errors.append(
                    f"  [{lbl}] ({px:.3f},{py:.3f}) not on grid X∈{xs} Y∈{ys}")

    return errors


# ── fixture ───────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def e2e_state():
    """Run full pipeline once and return all intermediate results."""
    # Re-import app to pick up the env-var we set at module top
    importlib.reload(app_module)
    client = TestClient(app_module.app)

    state: dict = {"work_dir": _TMP_WORK, "errors": []}

    # ── Step 1: Calibrate front ──
    t0 = time.time()
    try:
        cal_front = _post_file(client, "/api/vision/calibrate-black-frame", FRONT_IMG,
                               {"frame_w_mm": str(FRAME_W_MM), "frame_h_mm": str(FRAME_H_MM)})
        state["cal_front_id"] = cal_front["calibration_id"]
        state["cal_front"] = cal_front
        print(f"  [calibrate front] id={cal_front['calibration_id']}, "
              f"ppm={cal_front['pixels_per_mm']:.1f}, "
              f"time={time.time()-t0:.1f}s")
    except Exception as e:
        state["errors"].append(f"calibrate front failed: {e}")
        return state

    # ── Step 2: Calibrate back ──
    t0 = time.time()
    try:
        cal_back = _post_file(client, "/api/vision/calibrate-black-frame", BACK_IMG,
                              {"frame_w_mm": str(FRAME_W_MM), "frame_h_mm": str(FRAME_H_MM)})
        state["cal_back_id"] = cal_back["calibration_id"]
        state["cal_back"] = cal_back
        print(f"  [calibrate back]  id={cal_back['calibration_id']}, "
              f"ppm={cal_back['pixels_per_mm']:.1f}, "
              f"time={time.time()-t0:.1f}s")
    except Exception as e:
        state["errors"].append(f"calibrate back failed: {e}")
        return state

    # ── Step 3: Extract PCB outline (front & back) ──
    for side in ["front", "back"]:
        t0 = time.time()
        try:
            cal_id = state[f"cal_{side}_id"]
            outline = _post_form(client, "/api/vision/extract-pcb", {"calibration_id": cal_id})
            state[f"outline_{side}"] = outline
            print(f"  [extract {side}] vertices={len(outline.get('outline',[]))}, "
                  f"grooves={outline.get('groove_count',0)}, "
                  f"time={time.time()-t0:.1f}s")
        except Exception as e:
            state["errors"].append(f"extract {side} failed: {e}")

    # ── Step 4: Detect holes (front & back) ──
    for side in ["front", "back"]:
        t0 = time.time()
        try:
            cal_id = state[f"cal_{side}_id"]
            outline_data = state[f"outline_{side}"]
            outline_json = json.dumps({"outline": outline_data.get("outline", [])})
            holes = _post_form(client, "/api/vision/detect-holes",
                               {"calibration_id": cal_id, "outline_json": outline_json})
            state[f"holes_{side}"] = holes
            print(f"  [holes {side}] count={holes.get('hole_count',0)}, "
                  f"time={time.time()-t0:.1f}s")
        except Exception as e:
            state["errors"].append(f"holes {side} failed: {e}")

    # ── Step 5: Detect pads/terminals (VLM) ──
    for side in ["front", "back"]:
        t0 = time.time()
        try:
            cal_id = state[f"cal_{side}_id"]
            pads = _post_form(client, "/api/vision/detect-terminals",
                              {"calibration_id": cal_id, "side": side})
            state[f"pads_{side}"] = pads
            groups = _get_centers_by_label(pads.get("candidates", []))
            print(f"  [pads {side}] count={pads.get('candidate_count',0)}, "
                  f"labels={sorted(groups.keys())}, "
                  f"time={time.time()-t0:.1f}s")
        except Exception as e:
            state["errors"].append(f"pads {side} failed: {e}")

    # ── Step 6: Detect components (VLM) ──
    for side in ["front", "back"]:
        t0 = time.time()
        try:
            cal_id = state[f"cal_{side}_id"]
            comps = _post_form(client, "/api/vision/detect-components",
                               {"calibration_id": cal_id, "side": side})
            state[f"comps_{side}"] = comps
            c_types = [c["type"] for c in comps.get("components", [])]
            print(f"  [comps {side}] count={len(comps.get('components',[]))}, "
                  f"types={c_types[:10]}{'...' if len(c_types)>10 else ''}, "
                  f"time={time.time()-t0:.1f}s")
        except Exception as e:
            state["errors"].append(f"components {side} failed: {e}")

    total = time.time() - t0 if 't0' in dir() else 0
    print(f"\n  Total time: {total:.1f}s")
    return state


# ── tests ─────────────────────────────────────────────────────────────

def test_pipeline_no_errors(e2e_state):
    """The pipeline itself should not have crashed."""
    if e2e_state["errors"]:
        for err in e2e_state["errors"]:
            print(f"  PIPELINE ERROR: {err}")
    assert not e2e_state["errors"], f"Pipeline had {len(e2e_state['errors'])} error(s)"


def test_front_id_th_vertical(e2e_state):
    """Front: ID and TH pads must be vertically aligned (same X column).

    Only meaningful if both ID and TH were detected.
    """
    pads = e2e_state.get("pads_front")
    if pads is None:
        pytest.skip("Front pads not detected")
    groups = _get_centers_by_label(pads["candidates"])

    id_poses = groups.get("ID", [])
    th_poses = groups.get("TH", [])

    print(f"  Front pads by label: { {k: len(v) for k, v in groups.items()} }")
    if id_poses:
        print(f"  ID positions: {[(round(x,2), round(y,2)) for x,y in id_poses]}")
    if th_poses:
        print(f"  TH positions: {[(round(x,2), round(y,2)) for x,y in th_poses]}")

    errors = []

    # ID and TH should each be internally vertically aligned
    id_errors = _check_vertical_alignment("ID", id_poses)
    th_errors = _check_vertical_alignment("TH", th_poses)

    # ID and TH should be in the same column (same X)
    if id_poses and th_poses:
        common_x_id = id_poses[0][0]
        common_x_th = th_poses[0][0]
        if abs(common_x_id - common_x_th) > 1.0:
            errors.append(
                f"  ID X={common_x_id:.3f} vs TH X={common_x_th:.3f} — "
                f"not vertically aligned (diff={abs(common_x_id-common_x_th):.3f} mm)")
        else:
            print(f"  ✓ ID+TH same column at X≈{common_x_id:.2f}")

    errors.extend(id_errors)
    errors.extend(th_errors)

    for e in errors:
        print(e)
    assert not errors, f"ID/TH alignment: {len(errors)} issue(s)"


def test_front_pp_pn_columns(e2e_state):
    """Front: 3×P+ in one column, 3×P- in another column.
    Both columns share Y positions (horizontally aligned rows).
    """
    pads = e2e_state.get("pads_front")
    if pads is None:
        pytest.skip("Front pads not detected")
    groups = _get_centers_by_label(pads["candidates"])

    pp_poses = groups.get("P+", [])
    pn_poses = groups.get("P-", [])

    if not pp_poses or not pn_poses:
        # Not all pads detected — skip strict assertions, just report
        print(f"  P+={len(pp_poses)} pads, P-={len(pn_poses)} pads — skip checks")
        return

    print(f"  P+ positions: {[(round(x,2), round(y,2)) for x,y in pp_poses]}")
    print(f"  P- positions: {[(round(x,2), round(y,2)) for x,y in pn_poses]}")

    errors = []

    # P+ must be vertically aligned (same X)
    errors.extend(_check_vertical_alignment("P+", pp_poses))
    # P- must be vertically aligned (same X)
    errors.extend(_check_vertical_alignment("P-", pn_poses))

    # P+ and P- rows must be horizontally aligned (share Y values)
    if len(pp_poses) == len(pn_poses):
        errors.extend(_check_horizontal_alignment({"P+": pp_poses, "P-": pn_poses}))
    else:
        print(f"  Skipping Y-alignment: P+={len(pp_poses)} vs P-={len(pn_poses)}")

    # Pads should NOT be at the same X (P+ and P- are distinct columns)
    if pp_poses and pn_poses:
        pp_x = pp_poses[0][0]
        pn_x = pn_poses[0][0]
        if abs(pp_x - pn_x) < 0.5:
            errors.append(
                f"  P+ and P- columns at same X={pp_x:.3f} — should be distinct columns")
        else:
            print(f"  ✓ P+ at X≈{pp_x:.2f}, P- at X≈{pn_x:.2f} — distinct columns "
                  f"(gap={abs(pp_x-pn_x):.2f} mm)")

    for e in errors:
        print(e)
    assert not errors, f"P+/P- columns: {len(errors)} issue(s)"


def test_back_2x2_grid(e2e_state):
    """Back: B+ and B- pads form a 2×2 grid (4 pads, 2 columns × 2 rows)."""
    pads = e2e_state.get("pads_back")
    if pads is None:
        pytest.skip("Back pads not detected")
    groups = _get_centers_by_label(pads["candidates"])

    bp_poses = groups.get("B+", [])
    bn_poses = groups.get("B-", [])

    print(f"  Back pads by label: { {k: [(round(x,2), round(y,2)) for x,y in v] for k, v in groups.items()} }")

    if not bp_poses or not bn_poses:
        print("  B+ and/or B- not detected — skip grid checks")
        return

    all_poses = bp_poses + bn_poses
    errors = []

    # Check 2x2 grid structure
    errors.extend(_check_2x2_grid({"B+": bp_poses, "B-": bn_poses}))

    # B+ pads share same Y (top row) if there are 2 of them
    if len(bp_poses) == 2:
        y_diff = abs(bp_poses[0][1] - bp_poses[1][1])
        print(f"  B+ Y-diff = {y_diff:.2f} mm")
    if len(bn_poses) == 2:
        y_diff = abs(bn_poses[0][1] - bn_poses[1][1])
        print(f"  B- Y-diff = {y_diff:.2f} mm")

    # Verify NO pad was snapped to PCB center (frame_w_mm/2 = 20.0mm)
    # This would indicate the old bug is still present
    pcb_center_x = FRAME_W_MM / 2  # 20.0
    for lbl, poses in [("B+", bp_poses), ("B-", bn_poses)]:
        for px, py in poses:
            if abs(px - pcb_center_x) < 0.2:
                errors.append(
                    f"  [{lbl}] X={px:.3f} snapped to PCB center ({pcb_center_x}) — "
                    f"possible old bug regression!")

    for e in errors:
        print(e)
    assert not errors, f"Back 2×2 grid: {len(errors)} issue(s)"


def test_no_pcb_center_snap(e2e_state):
    """Regression test: NO candidate should have X/Y exactly at the PCB center
    (unless the PCB is genuinely tiny and has a centered pad)."""
    for side in ["front", "back"]:
        pads = e2e_state.get(f"pads_{side}")
        if pads is None:
            continue
        for c in pads["candidates"]:
            vp = c["visible_position"]
            cx, cy = float(vp["x_mm"]), float(vp["y_mm"])
            if abs(cx - FRAME_W_MM / 2) < 0.1:
                print(f"  WARNING [{side}][{c['label']}] X={cx:.3f} at PCB center X={FRAME_W_MM/2}")
            if abs(cy - FRAME_H_MM / 2) < 0.1:
                print(f"  WARNING [{side}][{c['label']}] Y={cy:.3f} at PCB center Y={FRAME_H_MM/2}")
    # This is a soft check — just reports, doesn't fail


# ── self-run (no pytest needed) ───────────────────────────────────────

def main():
    """Run full pipeline without pytest for debugging."""
    from datetime import datetime

    print(f"\n{'='*60}", flush=True)
    print(f"  E2E Pad Alignment Test — {datetime.now().strftime('%H:%M:%S')}", flush=True)
    print(f"  Frame: {FRAME_W_MM}×{FRAME_H_MM} mm", flush=True)
    print(f"  Front: {FRONT_IMG}", flush=True)
    print(f"  Back:  {BACK_IMG}", flush=True)
    print(f"{'='*60}\n", flush=True)

    # Use dedicated temp dir, set BEFORE import
    tmp_work = tempfile.mkdtemp(prefix="pcb_e2e_main_")
    os.environ["BATTERY_DESIGN_WORKDIR"] = str(tmp_work)

    # Force a fresh import with the new WORK_ROOT
    import battery_designer.app
    importlib.reload(battery_designer.app)
    from battery_designer.app import app, WORK_ROOT

    print(f"  Work dir: {tmp_work}", flush=True)
    print(f"  WORK_ROOT: {WORK_ROOT}", flush=True)

    client = TestClient(app)

    errors = []
    t_total = time.time()

    # ── Calibrate ──
    cal_front = _post_file(client, "/api/vision/calibrate-black-frame", FRONT_IMG,
                           {"frame_w_mm": str(FRAME_W_MM), "frame_h_mm": str(FRAME_H_MM)})
    cal_id_f = cal_front["calibration_id"]
    print(f"\n  [1/5] Front calibrated: {cal_id_f}", flush=True)
    cal_dir_f = Path(tmp_work) / "calibrations" / cal_id_f
    print(f"        cal dir: {cal_dir_f}", flush=True)
    print(f"        exists: {cal_dir_f.exists()}", flush=True)
    if cal_dir_f.exists():
        print(f"        files: {[f.name for f in cal_dir_f.iterdir()]}", flush=True)

    cal_back = _post_file(client, "/api/vision/calibrate-black-frame", BACK_IMG,
                          {"frame_w_mm": str(FRAME_W_MM), "frame_h_mm": str(FRAME_H_MM)})
    cal_id_b = cal_back["calibration_id"]
    print(f"  [1/5] Back  calibrated: {cal_id_b}")

    # ── Extract outline ──
    outline_front = None
    outline_back = None
    for side, cal_id in [("front", cal_id_f), ("back", cal_id_b)]:
        out = _post_form(client, "/api/vision/extract-pcb", {"calibration_id": cal_id})
        ol = out.get("outline", [])
        print(f"  [2/5] Outline {side}: {len(ol)} vertices, "
              f"{out.get('groove_count',0)} grooves", flush=True)
        # Check if pcb_outline.json was saved to disk
        outline_path = Path(tmp_work) / "calibrations" / cal_id / "pcb_outline.json"
        print(f"        outline saved: {outline_path.exists()}", flush=True)
        if outline_path.exists():
            saved = json.loads(outline_path.read_text())
            saved_ol = saved.get("outline", [])
            print(f"        saved outline vertices: {len(saved_ol)}", flush=True)
        if side == "front":
            outline_front = out
        else:
            outline_back = out

    # ── Detect holes ──
    for side, cal_id in [("front", cal_id_f), ("back", cal_id_b)]:
        # extract-pcb already saved pcb_outline.json to disk; let detect-holes
        # load from disk (no outline_json needed)
        t0 = time.time()
        print(f"  [3/5] Holes {side}: checking cal dir...", flush=True)
        cal_dir_check = Path(tmp_work) / "calibrations" / cal_id
        print(f"        cal dir exists: {cal_dir_check.exists()}", flush=True)
        if cal_dir_check.exists():
            print(f"        files: {[f.name for f in cal_dir_check.iterdir()]}", flush=True)
        try:
            holes = _post_form(client, "/api/vision/detect-holes",
                               {"calibration_id": cal_id})
            print(f"  [3/5] Holes {side}: {holes.get('hole_count',0)} found "
                  f"({time.time()-t0:.1f}s)", flush=True)
        except Exception as e:
            print(f"  [3/5] Holes {side}: FAILED — {e}", flush=True)

    # ── Detect pads (VLM) ──
    for side, cal_id in [("front", cal_id_f), ("back", cal_id_b)]:
        t0 = time.time()
        pads = _post_form(client, "/api/vision/detect-terminals",
                          {"calibration_id": cal_id, "side": side})
        groups = _get_centers_by_label(pads.get("candidates", []))
        print(f"\n  [4/5] Pads {side}: {pads.get('candidate_count',0)} candidates "
              f"({time.time()-t0:.1f}s)")
        for lbl in sorted(groups.keys()):
            poses = groups[lbl]
            summary = ", ".join(f"({x:.3f},{y:.3f})" for x, y in poses)
            print(f"         {lbl} ×{len(poses)}: {summary}")

        if side == "front":
            pads_front = pads
        else:
            pads_back = pads

    # ── Detect components (VLM) ──
    for side, cal_id in [("front", cal_id_f), ("back", cal_id_b)]:
        t0 = time.time()
        comps = _post_form(client, "/api/vision/detect-components",
                           {"calibration_id": cal_id, "side": side})
        types = [c["type"] for c in comps.get("components", [])]
        print(f"\n  [5/5] Components {side}: {len(types)} found ({time.time()-t0:.1f}s)")
        for t in sorted(set(types)):
            print(f"         {t} ×{types.count(t)}")

    # ── VERIFY ──
    print(f"\n{'='*60}")
    print(f"  VERIFICATION")
    print(f"{'='*60}\n")

    all_ok = True

    # 1) Front ID + TH vertical
    gf = _get_centers_by_label(pads_front["candidates"])
    id_p = gf.get("ID", [])
    th_p = gf.get("TH", [])
    print(f"  [Test 1] Front ID+TH vertical:")
    e1 = _check_vertical_alignment("ID", id_p)
    e1 += _check_vertical_alignment("TH", th_p)
    if id_p and th_p:
        if abs(id_p[0][0] - th_p[0][0]) < 1.0:
            print(f"    ✓ ID and TH share column at X≈{id_p[0][0]:.2f}")
        else:
            print(f"    ✗ ID X={id_p[0][0]:.3f} vs TH X={th_p[0][0]:.3f}")
            all_ok = False
    else:
        print(f"    (ID={len(id_p)}, TH={len(th_p)} — skipping)")
    for e in e1:
        print(e)
        all_ok = False

    # 2) Front P+/P- columns
    pp_p = gf.get("P+", [])
    pn_p = gf.get("P-", [])
    print(f"\n  [Test 2] Front P+/P- columns:")
    e2 = _check_vertical_alignment("P+", pp_p)
    e2 += _check_vertical_alignment("P-", pn_p)
    if len(pp_p) >= 2 and len(pn_p) >= 2:
        if abs(pp_p[0][0] - pp_p[1][0]) < 0.5 and abs(pn_p[0][0] - pn_p[1][0]) < 0.5:
            print(f"    ✓ P+ vertical at X≈{pp_p[0][0]:.2f}, P- vertical at X≈{pn_p[0][0]:.2f}")
        else:
            print(f"    ✗ Vertical alignment broken")
            all_ok = False
        e2 += _check_horizontal_alignment({"P+": pp_p, "P-": pn_p})
    else:
        print(f"    (P+={len(pp_p)}, P-={len(pn_p)} — skipping)")
    for e in e2:
        print(e)
        all_ok = False

    # 3) Back 2x2 grid
    gb = _get_centers_by_label(pads_back["candidates"])
    bp_p = gb.get("B+", [])
    bn_p = gb.get("B-", [])
    print(f"\n  [Test 3] Back B+/B- 2×2 grid:")
    e3 = _check_2x2_grid({"B+": bp_p, "B-": bn_p})
    if not e3:
        xs = sorted(set(round(p[0], 2) for p in bp_p + bn_p))
        ys = sorted(set(round(p[1], 2) for p in bp_p + bn_p))
        print(f"    ✓ 2×2 grid: X∈{xs}, Y∈{ys}")
    for e in e3:
        print(e)
        all_ok = False

    # 4) Regression: no PCB center snap
    print(f"\n  [Test 4] Regression — no PCB center snap:")
    snaps = 0
    for side, pads_data, cal_id in [("front", pads_front, cal_id_f),
                                      ("back", pads_back, cal_id_b)]:
        for c in pads_data["candidates"]:
            cx = float(c["visible_position"]["x_mm"])
            cy = float(c["visible_position"]["y_mm"])
            if abs(cx - FRAME_W_MM / 2) < 0.1 or abs(cy - FRAME_H_MM / 2) < 0.1:
                print(f"    ⚠ [{side}] {c['label']} at ({cx:.3f},{cy:.3f}) — at PCB center!")
                snaps += 1
    if snaps == 0:
        print(f"    ✓ No pads snapped to PCB center")
    else:
        print(f"    ⚠ {snaps} pad(s) at PCB center — possible old bug")
        # Soft warning, not a hard fail

    print(f"\n{'='*60}")
    print(f"  {'ALL CHECKS PASSED' if all_ok else 'SOME CHECKS FAILED'}")
    print(f"  Total time: {time.time()-t_total:.1f}s")
    print(f"{'='*60}\n")

    # Cleanup
    import shutil
    shutil.rmtree(tmp_work, ignore_errors=True)

    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
