from __future__ import annotations

import pytest
from pydantic import ValidationError

from battery_designer.models import DesignSpec


def test_common_port_is_derived_from_shared_roles(common_spec):
    assert common_spec.port_topology == "common"
    assert common_spec.battery.series_cells == 1
    assert common_spec.battery.parallel_cells == 3


def test_separate_port_is_derived_from_distinct_terminals(common_spec):
    data = common_spec.model_dump(mode="json")
    data["terminals"] = [item for item in data["terminals"] if item["id"] not in {"PP", "PN"}]
    data["terminals"].extend(
        [
            {"id": "CP", "position": {"x_mm": 28, "y_mm": 3}, "roles": ["charge"], "polarity": "positive", "side": "front", "shape": "circle", "width_mm": 2, "height_mm": 2},
            {"id": "CN", "position": {"x_mm": 28, "y_mm": 12}, "roles": ["charge"], "polarity": "negative", "side": "front", "shape": "circle", "width_mm": 2, "height_mm": 2},
            {"id": "PP2", "position": {"x_mm": 36, "y_mm": 3}, "roles": ["discharge"], "polarity": "positive", "side": "front", "shape": "circle", "width_mm": 2, "height_mm": 2},
            {"id": "PN2", "position": {"x_mm": 36, "y_mm": 12}, "roles": ["discharge"], "polarity": "negative", "side": "front", "shape": "circle", "width_mm": 2, "height_mm": 2},
        ]
    )
    assert DesignSpec.model_validate(data).port_topology == "separate"


def test_terminal_outside_outline_is_rejected(common_spec):
    data = common_spec.model_dump(mode="json")
    data["terminals"][0]["position"] = {"x_mm": 100, "y_mm": 100}
    with pytest.raises(ValidationError, match="outside"):
        DesignSpec.model_validate(data)


def test_series_and_parallel_cannot_be_combined(common_spec):
    data = common_spec.model_dump(mode="json")
    data["battery"]["connection"] = "series_parallel"
    with pytest.raises(ValidationError):
        DesignSpec.model_validate(data)


def test_back_terminal_requires_back_photo_calibration(common_spec):
    data = common_spec.model_dump(mode="json")
    data["terminals"][0]["side"] = "back"
    with pytest.raises(ValidationError, match="back-side terminals"):
        DesignSpec.model_validate(data)


def test_back_terminal_accepts_calibrated_back_photo(common_spec):
    data = common_spec.model_dump(mode="json")
    data["terminals"][0]["side"] = "back"
    data["photo_capture"] = {
        "front_calibration_id": "a" * 32,
        "back_calibration_id": "b" * 32,
        "back_transform": "mirror_x",
        "alignment_error_mm": 0.2,
    }
    assert DesignSpec.model_validate(data).terminals[0].side.value == "back"


def test_auxiliary_terminal_can_be_unpolarized(common_spec):
    data = common_spec.model_dump(mode="json")
    data["terminals"].append({
        "id": "TH", "position": {"x_mm": 20, "y_mm": 8},
        "roles": ["temperature"], "polarity": None,
        "side": "front", "shape": "oval", "width_mm": 2, "height_mm": 1,
    })
    assert DesignSpec.model_validate(data).terminals[-1].polarity is None


def test_overcurrent_trip_must_exceed_continuous_current(common_spec):
    data = common_spec.model_dump(mode="json")
    data["limits"]["overcurrent_trip_a"] = data["limits"]["continuous_current_a"]
    with pytest.raises(ValidationError, match="overcurrent trip"):
        DesignSpec.model_validate(data)
