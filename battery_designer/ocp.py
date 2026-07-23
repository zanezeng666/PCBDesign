from __future__ import annotations

from typing import Any

from .mos import MosfetSelection


def evaluate_oc_protection(
    mos_selection: MosfetSelection,
    *,
    dischg_oc_detection_v_typ: float | None = None,
    dischg_oc_detection_v_min_25c: float | None = None,
    dischg_oc_detection_v_max_25c: float | None = None,
) -> dict[str, Any]:
    """Evaluate the overcurrent protection characteristics for a given MOSFET configuration.

    Returns a diagnostic dict describing where the hardware OC trip sits.
    No user target is required — the trip is purely a function of
    IC detection threshold ÷ (MOSFET Rds(on) / parallel count).
    """
    threshold_typ = dischg_oc_detection_v_typ
    threshold_min = dischg_oc_detection_v_min_25c
    threshold_max = dischg_oc_detection_v_max_25c

    switches = 2 if mos_selection.option.dual_series_switch else 1
    cold_max_path_r = mos_selection.option.rds_on_max_ohm * switches / mos_selection.package_count

    if threshold_typ is None:
        return {
            "status": "no_ic_data",
            "verified": False,
            "warning": "IC overcurrent-detection threshold is unknown.",
        }

    earliest_trip = float(threshold_min or threshold_typ) / cold_max_path_r
    nominal_trip = float(threshold_typ) / cold_max_path_r
    latest_trip = float(threshold_max or threshold_typ) / cold_max_path_r

    return {
        "status": "evaluated",
        "verified": False,
        "mosfet_mpn": mos_selection.option.mpn,
        "package_count": mos_selection.package_count,
        "cold_max_path_resistance_ohm": round(cold_max_path_r, 6),
        "ic_threshold_v": {
            "min_25c": threshold_min,
            "typ": threshold_typ,
            "max_25c": threshold_max,
        },
        "trip_current_a_cold_max_r": {
            "min_threshold": round(earliest_trip, 1),
            "typ_threshold": round(nominal_trip, 1),
            "max_threshold": round(latest_trip, 1),
        },
    }


# ――  legacy compat  ――――――――――――――――――――――――――――――――――――――

def assess_overcurrent_target(selection: MosfetSelection, target_a: float, device_params: dict[str, Any]) -> dict[str, Any]:
    """Legacy wrapper: check whether a *specific* OC trip target is met."""
    result = evaluate_oc_protection(
        selection,
        dischg_oc_detection_v_typ=device_params.get("discharge_overcurrent_detection_v_typ"),
        dischg_oc_detection_v_min_25c=device_params.get("discharge_overcurrent_detection_v_min_25c"),
        dischg_oc_detection_v_max_25c=device_params.get("discharge_overcurrent_detection_v_max_25c"),
    )
    if result["status"] != "evaluated":
        result["target_a"] = target_a
        return result

    trip_typ = result["trip_current_a_cold_max_r"]["typ_threshold"]
    result["target_a"] = target_a
    result["status"] = "conservative_match" if result["trip_current_a_cold_max_r"]["min_threshold"] >= target_a else "mismatch"
    result["verified"] = False
    result["warning"] = (
        "Trip current meets the conservative target." if result["status"] == "conservative_match"
        else f"OC trip typ ~{trip_typ} A is below target {target_a} A. May trip early under tolerance."
    )
    return result
