from __future__ import annotations

from typing import Any

from .catalog import DevicePackage
from .models import DesignSpec
from .mos import MosfetSelection


def assess_overcurrent_target(spec: DesignSpec, device: DevicePackage, selection: MosfetSelection) -> dict[str, Any]:
    target = spec.limits.overcurrent_trip_a
    threshold_typ = device.parameters.get("discharge_overcurrent_detection_v_typ")
    threshold_min = device.parameters.get("discharge_overcurrent_detection_v_min_25c")
    threshold_max = device.parameters.get("discharge_overcurrent_detection_v_max_25c")
    if target is None:
        return {"status": "not_requested", "verified": False}
    if threshold_typ is None or threshold_min is None or threshold_max is None:
        return {
            "status": "insufficient_ic_data",
            "verified": False,
            "target_a": target,
            "warning": "The resolved IC package has no complete overcurrent threshold range.",
        }

    switches = 2 if selection.option.dual_series_switch else 1
    cold_max_path_r = selection.option.rds_on_max_ohm * switches / selection.package_count
    required_path_r = float(threshold_typ) / target
    earliest_trip = float(threshold_min) / cold_max_path_r
    nominal_from_max_r = float(threshold_typ) / cold_max_path_r
    latest_from_max_r = float(threshold_max) / cold_max_path_r
    conservative_match = earliest_trip >= target
    return {
        "status": "conservative_match" if conservative_match else "mismatch",
        "verified": False,
        "target_a": target,
        "ic_threshold_v": {"min_25c": threshold_min, "typ": threshold_typ, "max_25c": threshold_max},
        "mosfet": selection.option.mpn,
        "package_count": selection.package_count,
        "cold_max_path_resistance_ohm": round(cold_max_path_r, 6),
        "required_path_resistance_ohm_typ": round(required_path_r, 6),
        "trip_current_a_using_cold_max_r": {
            "min_threshold": round(earliest_trip, 3),
            "typ_threshold": round(nominal_from_max_r, 3),
            "max_threshold": round(latest_from_max_r, 3),
        },
        "conservative_target_met": conservative_match,
        "warning": (
            "The requested trip current is above the conservative minimum; it can trip early."
            if not conservative_match
            else "A complete Rds(on) tolerance and temperature model is still required before validation."
        ),
    }
