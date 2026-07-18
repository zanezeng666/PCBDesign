from __future__ import annotations

from dataclasses import asdict, dataclass
from math import ceil

from .errors import DesignError
from .models import ElectricalLimits


@dataclass(frozen=True)
class MosfetOption:
    mpn: str
    package: str
    dual_series_switch: bool
    rds_on_max_ohm: float
    rds_temperature_factor: float
    thermal_resistance_c_per_w: float
    pulse_current_a: float
    lcsc: str | None = None
    status: str = "candidate"


@dataclass(frozen=True)
class MosfetSelection:
    option: MosfetOption
    package_count: int
    hot_resistance_per_path_ohm: float
    loss_per_package_w: float
    estimated_temp_rise_c: float
    continuous_margin: float
    peak_margin: float

    def as_dict(self) -> dict:
        data = asdict(self)
        data["option"] = asdict(self.option)
        return data


DEFAULT_MOSFETS = [
    MosfetOption(
        mpn="FS8205A",
        package="TSSOP-8",
        dual_series_switch=True,
        rds_on_max_ohm=0.035,
        rds_temperature_factor=1.6,
        thermal_resistance_c_per_w=125.0,
        pulse_current_a=25.0,
        status="candidate",
    )
]


def select_mosfets(limits: ElectricalLimits, options: list[MosfetOption] | None = None, max_packages: int = 16) -> MosfetSelection:
    options = options or DEFAULT_MOSFETS
    candidates: list[MosfetSelection] = []
    for option in options:
        switches = 2 if option.dual_series_switch else 1
        hot_path_r = option.rds_on_max_ohm * option.rds_temperature_factor * switches
        for count in range(1, max_packages + 1):
            per_package_current = limits.continuous_current_a / count
            loss = per_package_current**2 * hot_path_r
            rise = loss * option.thermal_resistance_c_per_w
            peak_per_package = limits.peak_current_a / count
            if rise <= limits.max_temp_rise_c and peak_per_package <= option.pulse_current_a * 0.7:
                candidates.append(
                    MosfetSelection(
                        option=option,
                        package_count=count,
                        hot_resistance_per_path_ohm=hot_path_r / count,
                        loss_per_package_w=loss,
                        estimated_temp_rise_c=rise,
                        continuous_margin=limits.max_temp_rise_c / max(rise, 1e-9),
                        peak_margin=(option.pulse_current_a * 0.7) / max(peak_per_package, 1e-9),
                    )
                )
                break
    if not candidates:
        raise DesignError(
            "MOS_SELECTION_FAILED",
            "No candidate MOSFET configuration meets current and temperature constraints.",
            {"max_packages": max_packages},
        )
    return min(candidates, key=lambda item: (item.package_count, item.estimated_temp_rise_c))
