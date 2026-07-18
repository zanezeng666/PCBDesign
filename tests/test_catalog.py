from __future__ import annotations

from pathlib import Path

import pytest

from battery_designer.catalog import IcCatalog, normalize_mpn, validate_ic_for_design
from battery_designer.errors import DesignError


def test_normalize_mpn():
    assert normalize_mpn("dw01-g ") == "DW01G"


def test_local_catalog_resolves_alias(tmp_path: Path):
    catalog = IcCatalog(Path("data/ic_catalog"), tmp_path)
    device = catalog.resolve("dw01g")
    assert device.full_mpn == "DW01-G"
    assert device.manufacturer.startswith("Fortune")


def test_hy2113_mb1b_resolves_marking_and_parameters(tmp_path: Path):
    catalog = IcCatalog(Path("data/ic_catalog"), tmp_path)
    device = catalog.resolve("3M1B")
    assert device.full_mpn == "HY2113-MB1B"
    assert device.pins["2"] == "CS"
    assert device.parameters["discharge_overcurrent_detection_v_typ"] == 0.250
    assert device.marking["second_line"] == "date_code"


def test_catalog_rejects_wrong_series(tmp_path: Path):
    device = IcCatalog(Path("data/ic_catalog"), tmp_path).resolve("DW01-G")
    with pytest.raises(DesignError) as error:
        validate_ic_for_design(device, 2, "common")
    assert error.value.code == "IC_SERIES_MISMATCH"


def test_catalog_rejects_unreviewed_port_topology(tmp_path: Path):
    device = IcCatalog(Path("data/ic_catalog"), tmp_path).resolve("DW01-G")
    with pytest.raises(DesignError) as error:
        validate_ic_for_design(device, 1, "separate")
    assert error.value.code == "IC_PORT_TOPOLOGY_MISMATCH"
