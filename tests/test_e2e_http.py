"""
Simplified end-to-end test: calibrate -> extract outline -> detect pads.
Uses requests over HTTP to the already-running server at 127.0.0.1:8000.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
BASE_URL = "http://127.0.0.1:8000"

FRAME_W_MM = 40.0
FRAME_H_MM = 25.0
FRONT_IMG = ROOT / "data" / "visualization" / "test_front.jpg"
BACK_IMG = ROOT / "data" / "visualization" / "test_back.jpg"


def _group_centers(candidates: list[dict]) -> dict[str, list[tuple[float, float]]]:
    groups: dict[str, list[tuple[float, float]]] = {}
    for c in candidates:
        lbl = c["label"]
        vp = c.get("visible_position", c.get("center_mm", {"x_mm": 0, "y_mm": 0}))
        groups.setdefault(lbl, []).append((float(vp["x_mm"]), float(vp["y_mm"])))
    return groups


def main():
    print(f"\n" + "=" * 60)
    print(f"  E2E Pad Alignment Test (via HTTP)")
    print(f"  Frame: {FRAME_W_MM}x{FRAME_H_MM} mm")
    print(f"=" * 60 + "\n")

    t_total = time.time()
    results: dict[str, list[str]] = {"pass": [], "fail": [], "skip": []}

    # -- Step 1: Calibrate front --
    t0 = time.time()
    with open(FRONT_IMG, "rb") as fh:
        resp = requests.post(
            f"{BASE_URL}/api/vision/calibrate-black-frame",
            data={"frame_w_mm": str(FRAME_W_MM), "frame_h_mm": str(FRAME_H_MM)},
            files={"file": (FRONT_IMG.name, fh, "image/jpeg")},
        )
    assert resp.status_code == 200, f"calibrate front: {resp.status_code} {resp.text[:200]}"
    cal_f = resp.json()
    cal_id_f = cal_f["calibration_id"]
    print(f"  [1] Front calibrated: {cal_id_f} ({time.time()-t0:.1f}s)")

    # -- Step 2: Calibrate back --
    t0 = time.time()
    with open(BACK_IMG, "rb") as fh:
        resp = requests.post(
            f"{BASE_URL}/api/vision/calibrate-black-frame",
            data={"frame_w_mm": str(FRAME_W_MM), "frame_h_mm": str(FRAME_H_MM)},
            files={"file": (BACK_IMG.name, fh, "image/jpeg")},
        )
    assert resp.status_code == 200, f"calibrate back: {resp.status_code} {resp.text[:200]}"
    cal_b = resp.json()
    cal_id_b = cal_b["calibration_id"]
    print(f"  [1] Back  calibrated: {cal_id_b} ({time.time()-t0:.1f}s)")

    # -- Step 3: Extract PCB outline --
    for side, cal_id in [("front", cal_id_f), ("back", cal_id_b)]:
        t0 = time.time()
        resp = requests.post(f"{BASE_URL}/api/vision/extract-pcb",
                             data={"calibration_id": cal_id})
        assert resp.status_code == 200, f"extract {side}: {resp.status_code} {resp.text[:200]}"
        out = resp.json()
        ol = out.get("outline", [])
        print(f"  [2] Outline {side}: {len(ol)} vertices, "
              f"{out.get('groove_count',0)} grooves "
              f"({time.time()-t0:.1f}s)")

    # -- Step 4: Detect holes --
    for side, cal_id in [("front", cal_id_f), ("back", cal_id_b)]:
        t0 = time.time()
        resp = requests.post(f"{BASE_URL}/api/vision/detect-holes",
                             data={"calibration_id": cal_id})
        if resp.status_code == 200:
            holes = resp.json()
            print(f"  [3] Holes {side}: {holes.get('hole_count',0)} found "
                  f"({time.time()-t0:.1f}s)")
        else:
            err = resp.json().get("error", {})
            print(f"  [3] Holes {side}: SKIP ({resp.status_code}) - "
                  f"{err.get('message','')}")

    # -- Step 5: Detect pads/terminals --
    pads_front_raw = None
    pads_back_raw = None
    for side, cal_id in [("front", cal_id_f), ("back", cal_id_b)]:
        t0 = time.time()
        resp = requests.post(f"{BASE_URL}/api/vision/detect-terminals",
                             data={"calibration_id": cal_id, "side": side})
        assert resp.status_code == 200, f"pads {side}: {resp.status_code} {resp.text[:200]}"
        pads = resp.json()
        groups = _group_centers(pads.get("candidates", []))
        labels = sorted(groups.keys())
        print(f"  [4] Pads {side}: {pads.get('candidate_count',0)} candidates "
              f"({time.time()-t0:.1f}s)")
        for lbl in labels:
            poses = groups[lbl]
            print(f"       {lbl} x{len(poses)}:")
            for x, y in poses:
                print(f"         ({x:.3f}, {y:.3f})")
        if side == "front":
            pads_front_raw = pads
        else:
            pads_back_raw = pads

    # -- VERIFICATION --
    print(f"\n" + "=" * 60)
    print(f"  VERIFICATION")
    print(f"=" * 60 + "\n")

    def _check_vertical(label, poses, tol=0.5):
        if len(poses) < 2:
            return None
        xs = [p[0] for p in poses]
        ref = xs[0]
        diffs = [abs(x - ref) for x in xs]
        if max(diffs) > tol:
            return f"  FAIL [{label}] X varies: {xs} (max diff={max(diffs):.3f})"
        return f"  OK [{label}] vertical at X~{ref:.3f}"

    def _check_horizontal(labels_groups, tol=0.5):
        labels = sorted(labels_groups.keys())
        if len(labels) < 2:
            return None
        y_sets = {l: sorted([p[1] for p in poses]) for l, poses in labels_groups.items()}
        ref_l = labels[0]
        ref_ys = y_sets[ref_l]
        for lbl in labels[1:]:
            if len(y_sets[lbl]) != len(ref_ys):
                return f"  FAIL {ref_l} has {len(ref_ys)} rows vs {lbl} has {len(y_sets[lbl])}"
            for i, (y, ry) in enumerate(zip(y_sets[lbl], ref_ys)):
                if abs(y - ry) > tol:
                    return f"  FAIL Y-align: {ref_l}[{i}]({ry:.3f}) vs {lbl}[{i}]({y:.3f}) diff={abs(y-ry):.3f}"
        return f"  OK {ref_l} & {', '.join(labels[1:])} share Y positions"

    def _check_2x2_grid(groups):
        all_pos = []
        for lbl, poses in groups.items():
            all_pos.extend(poses)
        if len(all_pos) != 4:
            return f"  FAIL Expected 4 pads for 2x2, got {len(all_pos)}"
        xs = sorted(set(round(p[0], 2) for p in all_pos))
        ys = sorted(set(round(p[1], 2) for p in all_pos))
        if len(xs) != 2 or len(ys) != 2:
            return f"  FAIL Expected 2x2 grid, got X in {xs}, Y in {ys}"
        return f"  OK 2x2 grid: X in {xs}, Y in {ys}"

    def _check_no_center_snap(groups):
        cx, cy = FRAME_W_MM / 2, FRAME_H_MM / 2
        snaps = []
        for lbl, poses in groups.items():
            for px, py in poses:
                if abs(px - cx) < 0.1 or abs(py - cy) < 0.1:
                    snaps.append(f"    WARN [{lbl}] ({px:.3f},{py:.3f}) at PCB center")
        return snaps

    if pads_front_raw:
        gf = _group_centers(pads_front_raw["candidates"])
        print(f"  Front pads: { {k: len(v) for k, v in gf.items()} }")
        # Test 1: ID + TH vertical
        msg = _check_vertical("ID", gf.get("ID", []))
        if msg: print(msg)
        msg = _check_vertical("TH", gf.get("TH", []))
        if msg: print(msg)
        if gf.get("ID") and gf.get("TH"):
            id_x = gf["ID"][0][0]
            th_x = gf["TH"][0][0]
            if abs(id_x - th_x) < 1.0:
                results["pass"].append(f"ID+TH same column at X~{id_x:.2f}")
            else:
                results["fail"].append(f"ID X={id_x:.3f} vs TH X={th_x:.3f}")

        # Test 2: P+/P- columns
        pp_msg = _check_vertical("P+", gf.get("P+", []))
        pn_msg = _check_vertical("P-", gf.get("P-", []))
        if pp_msg: print(pp_msg)
        if pn_msg: print(pn_msg)
        if gf.get("P+") and gf.get("P-"):
            h_msg = _check_horizontal({"P+": gf["P+"], "P-": gf["P-"]})
            if h_msg: print(h_msg)
            if pp_msg is None or "OK" in pp_msg:
                results["pass"].append("P+ vertically aligned")
            if pn_msg is None or "OK" in pn_msg:
                results["pass"].append("P- vertically aligned")
            if h_msg and "OK" in h_msg:
                results["pass"].append("P+ & P- Y-aligned")
            elif h_msg and "FAIL" in h_msg:
                results["fail"].append(h_msg)

        # Check no center snap
        snaps = _check_no_center_snap(gf)
        for s in snaps:
            print(s)

    if pads_back_raw:
        gb = _group_centers(pads_back_raw["candidates"])
        print(f"\n  Back pads: { {k: len(v) for k, v in gb.items()} }")
        msg = _check_2x2_grid({"B+": gb.get("B+", []), "B-": gb.get("B-", [])})
        if msg:
            print(msg)
            if "OK" in msg:
                results["pass"].append("Back 2x2 grid")
            else:
                results["fail"].append(msg)

        snaps = _check_no_center_snap(gb)
        for s in snaps:
            print(s)

    print(f"\n" + "=" * 60)
    for cat in ["pass", "fail", "skip"]:
        for item in results[cat]:
            marker = {"pass": "PASS", "fail": "FAIL", "skip": "SKIP"}[cat]
            print(f"  [{marker}] {item}")
    print(f"  Total time: {time.time()-t_total:.1f}s")
    print(f"=" * 60 + "\n")

    return 0 if not results["fail"] else 1


if __name__ == "__main__":
    sys.exit(main())
