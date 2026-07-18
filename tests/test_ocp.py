from __future__ import annotations

from pathlib import Path

from battery_designer.catalog import IcCatalog
from battery_designer.models import DesignSpec
from battery_designer.mos import select_mosfets
from battery_designer.ocp import assess_overcurrent_target


def test_hy2113_fs8205a_reports_8a_conservative_mismatch(common_spec, tmp_path: Path):
    data = common_spec.model_dump(mode="json")
    data["protection_ic"] = "HY2113-MB1B"
    data["limits"]["overcurrent_trip_a"] = 8
    spec = DesignSpec.model_validate(data)
    device = IcCatalog(Path("data/ic_catalog"), tmp_path).resolve("HY2113-MB1B")
    result = assess_overcurrent_target(spec, device, select_mosfets(spec.limits))
    assert result["status"] == "mismatch"
    assert result["required_path_resistance_ohm_typ"] == 0.03125
