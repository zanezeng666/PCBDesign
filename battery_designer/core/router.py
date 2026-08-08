"""System health and optimization dashboard API endpoints.

Moved from ``routers/system.py``.  Uses shared singletons from
``core.singletons`` and paths from ``core.config``.
"""

from __future__ import annotations

import json
import re

from fastapi import APIRouter
from fastapi.responses import FileResponse

from .config import STATIC_ROOT, DATA_ROOT
from .singletons import catalog, pipeline

router = APIRouter()


@router.get("/api/health")
def health():
    return {
        "status": "ok",
        "kicad": pipeline.diagnose(),
        "resolver_configured": bool(catalog.resolver_endpoint),
    }


@router.get("/optimization")
async def optimization_dashboard():
    """Serve the self-optimization debug dashboard."""
    return FileResponse(STATIC_ROOT / "optimization_dashboard.html")


@router.get("/api/optimization/summary")
async def optimization_summary():
    """Return optimization history summary for the dashboard."""
    opt_dir = DATA_ROOT / "optimization"
    viz_dir = DATA_ROOT / "visualization"

    rounds: list[dict] = []
    round_pattern = re.compile(r"report_round_(\d+)_(front|back)\.json")

    round_files: dict[int, dict[str, object]] = {}
    if opt_dir.exists():
        for f in sorted(opt_dir.iterdir()):
            m = round_pattern.match(f.name)
            if m:
                rnd = int(m.group(1))
                side = m.group(2)
                round_files.setdefault(rnd, {})[side] = f

    for rnd in sorted(round_files.keys()):
        files = round_files[rnd]
        if "front" not in files and "back" not in files:
            continue

        front_data = None
        back_data = None
        try:
            if "front" in files:
                front_data = json.loads(files["front"].read_text(encoding="utf-8"))
            if "back" in files:
                back_data = json.loads(files["back"].read_text(encoding="utf-8"))
        except Exception:
            continue

        report = front_data or back_data or {}
        if isinstance(report, dict) and "composite_score" in report:
            comp = report["composite_score"]
            total = round(comp.get("total", 0), 1) if isinstance(comp, dict) else 0
        else:
            comp = {}
            total = 0

        front_score = 0
        back_score = 0
        if front_data and isinstance(front_data, dict):
            fc = front_data.get("composite_score", {})
            front_score = round(fc.get("total", 0), 1) if isinstance(fc, dict) else 0
        if back_data and isinstance(back_data, dict):
            bc = back_data.get("composite_score", {})
            back_score = round(bc.get("total", 0), 1) if isinstance(bc, dict) else 0

        iou_mean = 0
        for side_data in [front_data, back_data]:
            if side_data and isinstance(side_data, dict):
                iou = side_data.get("iou", {})
                if isinstance(iou, dict):
                    iou_mean = max(iou_mean, iou.get("mean", 0))

        matches_quality = {"excellent": 0, "good": 0, "fair": 0, "poor": 0}
        for side_data in [front_data, back_data]:
            if side_data and isinstance(side_data, dict):
                iou = side_data.get("iou", {})
                if isinstance(iou, dict) and "values" in iou:
                    for v in iou["values"]:
                        if v >= 0.8:
                            matches_quality["excellent"] += 1
                        elif v >= 0.6:
                            matches_quality["good"] += 1
                        elif v >= 0.3:
                            matches_quality["fair"] += 1
                        else:
                            matches_quality["poor"] += 1

        diagnosis_parts = []
        for label, side_data in [("Front", front_data), ("Back", back_data)]:
            if side_data and isinstance(side_data, dict):
                cd = side_data.get("center_offset", {})
                if isinstance(cd, dict):
                    mean_off = cd.get("mean_px", 0)
                    if mean_off > 5:
                        diagnosis_parts.append(f"{label} offset:{mean_off:.1f}px")
                rec = side_data.get("recall", {})
                if isinstance(rec, dict) and rec.get("value", 1) < 0.8:
                    diagnosis_parts.append(f"{label} recall:{rec['value']:.1%}")

        round_entry = {
            "round": rnd,
            "front_score": front_score,
            "back_score": back_score,
            "composite_score": total,
            "iou_mean": round(iou_mean, 1),
            "diagnosis": "; ".join(diagnosis_parts) if diagnosis_parts else "OK",
            "matches_quality": matches_quality,
        }

        params_file = opt_dir / f"params_round_{rnd:03d}.json"
        if params_file.exists():
            try:
                round_entry["params"] = json.loads(
                    params_file.read_text(encoding="utf-8")
                )
            except Exception:
                round_entry["params"] = {}

        viz_files = []
        if viz_dir.exists():
            for side in ["front", "back"]:
                viz_file = viz_dir / f"round_{rnd:03d}_{side}.jpg"
                if viz_file.exists():
                    viz_files.append(f"data/visualization/round_{rnd:03d}_{side}.jpg")
        round_entry["viz_images"] = viz_files

        rounds.append(round_entry)

    best_score = max((r["composite_score"] for r in rounds), default=0)

    best_params = {}
    best_params_file = opt_dir / "best_params.json"
    if best_params_file.exists():
        try:
            best_params = json.loads(best_params_file.read_text(encoding="utf-8"))
        except Exception:
            pass

    return {
        "rounds": rounds,
        "best_score": best_score,
        "best_params": best_params,
        "total_rounds": len(rounds),
    }
