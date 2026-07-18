from __future__ import annotations

import pytest

from battery_designer.models import DesignSpec


@pytest.fixture
def common_spec() -> DesignSpec:
    return DesignSpec.model_validate(
        {
            "name": "test-board",
            "protection_ic": "DW01-G",
            "battery": {
                "count": 3,
                "connection": "parallel",
                "cell_min_v": 3.0,
                "cell_nominal_v": 3.7,
                "cell_max_v": 4.2,
            },
            "limits": {
                "continuous_current_a": 2,
                "peak_current_a": 5,
                "peak_duration_s": 3,
                "ambient_temp_c": 25,
                "max_temp_rise_c": 40,
            },
            "outline": {
                "confirmed": True,
                "points": [
                    {"x_mm": 0, "y_mm": 0},
                    {"x_mm": 40, "y_mm": 0},
                    {"x_mm": 40, "y_mm": 15},
                    {"x_mm": 0, "y_mm": 15},
                ],
            },
            "terminals": [
                {"id": "BP", "position": {"x_mm": 4, "y_mm": 4}, "roles": ["battery"], "polarity": "positive", "width_mm": 2, "height_mm": 2},
                {"id": "BN", "position": {"x_mm": 4, "y_mm": 11}, "roles": ["battery"], "polarity": "negative", "width_mm": 2, "height_mm": 2},
                {"id": "PP", "position": {"x_mm": 35, "y_mm": 4}, "roles": ["charge", "discharge"], "polarity": "positive", "width_mm": 2, "height_mm": 2},
                {"id": "PN", "position": {"x_mm": 35, "y_mm": 11}, "roles": ["charge", "discharge"], "polarity": "negative", "width_mm": 2, "height_mm": 2},
            ],
        }
    )
