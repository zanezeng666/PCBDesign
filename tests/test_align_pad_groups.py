"""
Test _align_pad_groups with synthetic scenarios mirroring real PCB layouts.

Scenarios:
  1. Back side 2x2 grid (B+/B-):  4 identical pads, both X & Y aligned
  2. Front side ID + TH:          vertically aligned, different labels
  3. Front side P+/P- columns:    3 P+ vertically, 3 P- vertically,
                                  6 pads same size, columns also Y-aligned
"""

import sys
import os
import json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from battery_designer.app import _align_pad_groups


# ── Helpers ────────────────────────────────────────────────────────────

def _make_candidate(label: str, cx_mm: float, cy_mm: float,
                    w_mm: float = 6.0, h_mm: float = 4.0,
                    idx: int = 0) -> dict:
    """Build one pad candidate in the format expected by _align_pad_groups."""
    hw, hh = w_mm / 2, h_mm / 2
    poly = [
        {"x_mm": round(cx_mm - hw, 3), "y_mm": round(cy_mm - hh, 3)},
        {"x_mm": round(cx_mm + hw, 3), "y_mm": round(cy_mm - hh, 3)},
        {"x_mm": round(cx_mm + hw, 3), "y_mm": round(cy_mm + hh, 3)},
        {"x_mm": round(cx_mm - hw, 3), "y_mm": round(cy_mm + hh, 3)},
    ]
    region = {
        "type": "solder_pad",
        "visual_class": "metallic",
        "shape": "rect",
        "center": {"x_mm": cx_mm, "y_mm": cy_mm},
        "bbox": {
            "x_mm": round(cx_mm - hw, 3),
            "y_mm": round(cy_mm - hh, 3),
            "width_mm": w_mm,
            "height_mm": h_mm,
        },
        "polygon": poly,
        "source": "vlm",
    }
    cand = {
        "id": f"{label}_{idx + 1}",
        "label": label,
        "visible_position": {"x_mm": cx_mm, "y_mm": cy_mm},
        "visible_region": region,
        "matched_regions": [region],
        "_cv_corner_radius_px": None,
    }
    return cand


def _build_result(candidates: list[dict]) -> dict:
    return {"candidates": candidates}


def _get_centers(result: dict) -> list[tuple[str, float, float]]:
    """Return (label, x_mm, y_mm) sorted by x then y."""
    out = []
    for c in result["candidates"]:
        vp = c["visible_position"]
        out.append((c["label"], vp["x_mm"], vp["y_mm"]))
    return sorted(out, key=lambda x: (x[1], x[2], x[0]))


def _check(label: str, expected: list[tuple[float, float]],
           result: dict, tolerance: float = 0.01) -> list[str]:
    """Verify pads with *label* are at *expected* positions (unordered match)."""
    actual = [(c["visible_position"]["x_mm"], c["visible_position"]["y_mm"])
              for c in result["candidates"] if c["label"] == label]
    errors = []
    if len(actual) != len(expected):
        errors.append(
            f"  [{label}] count mismatch: got {len(actual)}, expected {len(expected)}")
        return errors
    # Sort both for comparison
    actual_s = sorted(actual, key=lambda p: (p[0], p[1]))
    expected_s = sorted(expected, key=lambda p: (p[0], p[1]))
    for i, ((ax, ay), (ex, ey)) in enumerate(zip(actual_s, expected_s)):
        if abs(ax - ex) > tolerance or abs(ay - ey) > tolerance:
            errors.append(
                f"  [{label}][{i}] got ({ax:.3f},{ay:.3f}) expected ({ex:.3f},{ey:.3f})")
    return errors


# ── Test 1: Back side 2×2 grid ────────────────────────────────────────

def test_back_2x2_grid():
    """4 identical pads on back: B+_left, B+_right, B-_left, B-_right.

    Layout (grid):
         B+_L(50,80)          B+_R(110,80)
         B-_L(50,140)         B-_R(110,140)

    The 2x2 grid means pads are BOTH row-aligned and column-aligned.
    Grouping must produce correct vertical OR horizontal groups — either
    is fine, but NO pad should move to the PCB centre.
    """
    pcb_w_mm = 160.0
    pcb_h_mm = 180.0

    pads = [
        _make_candidate("B+", 50.0, 80.0, 6.0, 4.0, 0),
        _make_candidate("B+", 110.0, 80.0, 6.0, 4.0, 1),
        _make_candidate("B-", 50.0, 140.0, 6.0, 4.0, 0),
        _make_candidate("B-", 110.0, 140.0, 6.0, 4.0, 1),
    ]
    result = _build_result(pads)
    aligned = _align_pad_groups(result, pcb_w_mm=pcb_w_mm, pcb_h_mm=pcb_h_mm)

    errors = []
    centers = _get_centers(aligned)

    # X should be ~50 or ~110 (NOT snapped to PCB center at 80)
    for label, cx, cy in centers:
        if abs(cx - 80.0) < 0.1:  # near PCB center → BUG
            errors.append(
                f"  [{label}] X={cx:.3f} — snapped to PCB center (80), should be 50 or 110")
        elif abs(cx - 50.0) > 1.0 and abs(cx - 110.0) > 1.0:
            errors.append(
                f"  [{label}] X={cx:.3f} — too far from expected 50 or 110")

    # Y should be ~80 or ~140 (not shifted significantly)
    for label, cx, cy in centers:
        if abs(cy - 80.0) > 0.5 and abs(cy - 140.0) > 0.5:
            errors.append(
                f"  [{label}] Y={cy:.3f} — too far from expected 80 or 140")

    return errors


# ── Test 2: Front side ID + TH vertically aligned ─────────────────────

def test_front_id_th():
    """ID and TH (T) are vertically aligned but DIFFERENT pads.

    Layout:
         ID (60, 100)
         TH (60, 160)

    Should form ONE vertical group. X must stay at 60 (NOT PCB center).
    Y must preserve ~100 and ~160.
    """
    pcb_w_mm = 160.0
    pcb_h_mm = 180.0

    pads = [
        _make_candidate("ID", 60.0, 100.0, 4.0, 3.0, 0),
        _make_candidate("TH", 60.0, 160.0, 5.0, 4.0, 0),
    ]
    result = _build_result(pads)
    aligned = _align_pad_groups(result, pcb_w_mm=pcb_w_mm, pcb_h_mm=pcb_h_mm)

    errors = []

    # Both must have been grouped (same X → vertical group)
    # X must stay at ~60
    for c in aligned["candidates"]:
        vp = c["visible_position"]
        if abs(vp["x_mm"] - 60.0) > 0.5:
            errors.append(
                f"  [{c['label']}] X={vp['x_mm']:.3f} — expected ~60 (NOT PCB center)")

    # Y should be ~100 and ~160 (symmetrized preserves midpoint)
    for c in aligned["candidates"]:
        vp = c["visible_position"]
        if not (95 < vp["y_mm"] < 105) and not (155 < vp["y_mm"] < 165):
            errors.append(
                f"  [{c['label']}] Y={vp['y_mm']:.3f} — expected ~100 or ~160")

    return errors


# ── Test 3: Front side 3×P+ and 3×P- columns ─────────────────────────

def test_front_pp_pn_columns():
    """3 P+ in left column, 3 P- in right column, all same size,
    Y-aligned rows (columns share Y positions).

    Layout:
         P+_1(30,60)         P-_1(140,60)
         P+_2(30,100)        P-_2(140,100)
         P+_3(30,140)        P-_3(140,140)

    Must form TWO separate vertical groups (not one giant merged group).
    P+ column X = ~30, P- column X = ~140.
    Y spacing = ~40mm uniform.
    """
    pcb_w_mm = 170.0
    pcb_h_mm = 160.0

    pads = [
        _make_candidate("P+", 30.0, 60.0, 6.0, 3.5, 0),
        _make_candidate("P+", 30.0, 100.0, 6.0, 3.5, 1),
        _make_candidate("P+", 30.0, 140.0, 6.0, 3.5, 2),
        _make_candidate("P-", 140.0, 60.0, 6.0, 3.5, 0),
        _make_candidate("P-", 140.0, 100.0, 6.0, 3.5, 1),
        _make_candidate("P-", 140.0, 140.0, 6.0, 3.5, 2),
    ]
    result = _build_result(pads)
    aligned = _align_pad_groups(result, pcb_w_mm=pcb_w_mm, pcb_h_mm=pcb_h_mm)

    errors = []

    # Check P+ column: X must be ~30
    for c in aligned["candidates"]:
        if c["label"] == "P+":
            vp = c["visible_position"]
            if abs(vp["x_mm"] - 30.0) > 0.5:
                errors.append(
                    f"  [P+] X={vp['x_mm']:.3f} — expected ~30")

    # Check P- column: X must be ~140
    for c in aligned["candidates"]:
        if c["label"] == "P-":
            vp = c["visible_position"]
            if abs(vp["x_mm"] - 140.0) > 0.5:
                errors.append(
                    f"  [P-] X={vp['x_mm']:.3f} — expected ~140")

    # Check Y positions: ~60, ~100, ~140
    for c in aligned["candidates"]:
        vp = c["visible_position"]
        close_to = any(abs(vp["y_mm"] - y) < 1.0 for y in [60, 100, 140])
        if not close_to:
            errors.append(
                f"  [{c['label']}] Y={vp['y_mm']:.3f} — expected ~60/~100/~140")

    # Both columns should have exactly 3 pads each (not merged into 1 group of 6)
    p_plus = [c for c in aligned["candidates"] if c["label"] == "P+"]
    p_minus = [c for c in aligned["candidates"] if c["label"] == "P-"]
    if len(p_plus) != 3:
        errors.append(f"  P+ count = {len(p_plus)} — expected 3")
    if len(p_minus) != 3:
        errors.append(f"  P- count = {len(p_minus)} — expected 3")

    return errors


# ── Test 4: tolerance should NOT merge nearby-but-distinct columns ─────

def test_tolerance_separate_columns():
    """Two distinct columns 8mm apart should NOT be merged.

    tol = typical * 0.6 = 6.0 * 0.6 = 3.6mm.
    8mm gap > 3.6mm → should stay separate.
    """
    pads = [
        _make_candidate("B+", 50.0, 100.0, 6.0, 4.0, 0),
        _make_candidate("B+", 58.0, 100.0, 6.0, 4.0, 1),  # 8mm apart
    ]
    result = _build_result(pads)
    aligned = _align_pad_groups(result, pcb_w_mm=100.0, pcb_h_mm=150.0)

    errors = []
    centers = _get_centers(aligned)

    for label, cx, cy in centers:
        if abs(cx - 50.0) > 1.0 and abs(cx - 58.0) > 1.0:
            errors.append(
                f"  [{label}] X={cx:.3f} — should stay at 50 or 58 (not merged)")

    return errors


# ── Runner ─────────────────────────────────────────────────────────────

def run_all():
    tests = [
        ("Back 2x2 grid (B+/B-)", test_back_2x2_grid),
        ("Front ID + TH vertical", test_front_id_th),
        ("Front P+/P- columns", test_front_pp_pn_columns),
        ("Tolerance: separate columns", test_tolerance_separate_columns),
    ]

    all_passed = True
    for name, fn in tests:
        errors = fn()
        status = "PASS" if not errors else "FAIL"
        if errors:
            all_passed = False
        print(f"\n{'─' * 60}")
        print(f"  {status}: {name}")
        for e in errors:
            print(e)

    print(f"\n{'═' * 60}")
    print(f"  {'ALL TESTS PASSED' if all_passed else 'SOME TESTS FAILED'}")
    print(f"{'═' * 60}\n")

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(run_all())
