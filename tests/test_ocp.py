from __future__ import annotations

from pathlib import Path

from battery_designer.catalog import IcCatalog
from battery_designer.mos import derive_config_from_count, get_mosfet
from battery_designer.ocp import evaluate_oc_protection


def test_hy2113_fs8205a_oc_protection_evaluated(tmp_path: Path):
    catalog = IcCatalog(Path("data/ic_catalog"), tmp_path)
    device = catalog.resolve("HY2113-MB1B")

    mosfet = get_mosfet("FS8205A")
    selection = derive_config_from_count(2, mosfet)

    result = evaluate_oc_protection(
        selection,
        dischg_oc_detection_v_typ=device.parameters.get("discharge_overcurrent_detection_v_typ"),
        dischg_oc_detection_v_min_25c=device.parameters.get("discharge_overcurrent_detection_v_min_25c"),
        dischg_oc_detection_v_max_25c=device.parameters.get("discharge_overcurrent_detection_v_max_25c"),
    )
    assert result["status"] == "evaluated"
    assert result["package_count"] == 2
    trip = result["trip_current_a_cold_max_r"]["typ_threshold"]
    # HY2113 OC detection 0.25 V, FS8205A max Rds(on) 35 mΩ × 2 switches / 2 pkgs
    expected = 0.25 / 0.035  # = 7.14A (cold, typical threshold)
    assert abs(trip - expected) < 0.15, f"Expected ~{expected}A, got {trip}A"
