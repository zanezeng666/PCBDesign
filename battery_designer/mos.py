from __future__ import annotations

from dataclasses import asdict, dataclass
from math import sqrt

from .errors import DesignError
from .models import ElectricalLimits

# ── MOSFET database keyed by MPN ──────────────────────────────
# Typical companion MOSFETs for common protection ICs.
# FS8205A: dual N-channel, standard for DW01 / HY2113 / TP4056.


@dataclass(frozen=True)
class MosfetOption:
    mpn: str
    package: str
    dual_series_switch: bool
    rds_on_max_ohm: float
    rds_temperature_factor: float        # hot_rds = rds_on_max * this
    thermal_resistance_c_per_w: float    # junction-to-ambient
    pulse_current_a: float               # per FET, 300 µs pulse
    lcsc: str | None = None
    status: str = "candidate"


@dataclass(frozen=True)
class MosfetSelection:
    option: MosfetOption
    package_count: int
    hot_resistance_per_path_ohm: float
    loss_per_package_w: float
    estimated_temp_rise_c: float
    continuous_current_a: float

    def as_dict(self) -> dict:
        data = asdict(self)
        data["option"] = asdict(self.option)
        return data


# ―――――――――――――――――――――――――――――――――――――――――――――――――――――――――
MOSFET_DATABASE: dict[str, MosfetOption] = {
    "FS8205A": MosfetOption(
        mpn="FS8205A",
        package="TSSOP-8",
        dual_series_switch=True,
        rds_on_max_ohm=0.035,
        rds_temperature_factor=1.6,
        thermal_resistance_c_per_w=125.0,
        pulse_current_a=25.0,
        status="candidate",
    ),
}

DEFAULT_MOSFET_MPN = "FS8205A"

# ―――――――――――――――――――――――――――――――――――――――――――――――――――――――――
#  Public API
# ―――――――――――――――――――――――――――――――――――――――――――――――――――――――――


def get_mosfet(mpn: str) -> MosfetOption:
    """Look up a MOSFET by MPN; fall back to the default."""
    option = MOSFET_DATABASE.get(mpn) or MOSFET_DATABASE[DEFAULT_MOSFET_MPN]
    return option


def list_available_mpns() -> list[str]:
    return sorted(MOSFET_DATABASE.keys())


def derive_config_from_count(
    mos_count: int,
    mosfet: MosfetOption | None = None,
    *,
    max_temp_rise_c: float = 40.0,
    max_packages: int = 16,
) -> MosfetSelection:
    """Evaluate thermal and continuous-current characteristics for a fixed MOSFET count.

    *hot* resistance includes the temperature derating factor.
    Continuous current is estimated by the self-heating limit:
        Iₘₐₓ = mos_count × √(ΔTₘₐₓ / (R_hot_single × θⱼₐ))
    """
    mosfet = mosfet or get_mosfet(DEFAULT_MOSFET_MPN)

    if mos_count > max_packages:
        raise DesignError(
            "MOS_COUNT_EXCEEDED",
            f"Requested {mos_count} MOSFET packages, maximum is {max_packages}.",
            {"mos_count": mos_count, "max_packages": max_packages},
        )

    switches = 2 if mosfet.dual_series_switch else 1
    hot_r_single = mosfet.rds_on_max_ohm * mosfet.rds_temperature_factor * switches

    # Per-package current that hits the temperature rise limit
    max_per_package_current = sqrt(max_temp_rise_c / (hot_r_single * mosfet.thermal_resistance_c_per_w))
    continuous = mos_count * max_per_package_current

    # Actual loss & rise at the computed continuous current
    per_pkg_current = continuous / mos_count
    loss_per_pkg = per_pkg_current ** 2 * hot_r_single
    rise = loss_per_pkg * mosfet.thermal_resistance_c_per_w

    return MosfetSelection(
        option=mosfet,
        package_count=mos_count,
        hot_resistance_per_path_ohm=hot_r_single / mos_count,
        loss_per_package_w=loss_per_pkg,
        estimated_temp_rise_c=rise,
        continuous_current_a=round(continuous, 2),
    )


def derive_peak_current(
    selection: MosfetSelection,
    *,
    derating: float = 0.7,
) -> float:
    """Peak current this configuration can handle (300 µs pulse, derated)."""
    return round(selection.package_count * selection.option.pulse_current_a * derating, 1)


def derive_oc_trip(
    selection: MosfetSelection,
    dischg_oc_detection_v: float = 0.25,
    *,
    use_hot_resistance: bool = False,
) -> float:
    """Overcurrent trip threshold.

    OC trip = IC_detection_voltage / Rds_on (total path).

    By default uses *cold* resistance (room temp) → the *worst-case* trip point.
    Set use_hot_resistance=True for typical operating-temperature estimate.
    """
    mosfet = selection.option
    switches = 2 if mosfet.dual_series_switch else 1
    per_package_r = mosfet.rds_on_max_ohm * switches
    if use_hot_resistance:
        per_package_r *= mosfet.rds_temperature_factor
    total_r = per_package_r / selection.package_count
    return round(dischg_oc_detection_v / total_r, 1)


def derive_electrical_limits(
    selection: MosfetSelection,
    dischg_oc_detection_v: float = 0.25,
    *,
    peak_duration_s: float = 0.3,
    ambient_temp_c: float = 25.0,
    max_temp_rise_c: float = 40.0,
) -> ElectricalLimits:
    """Derive full ElectricalLimits from MOSFET configuration + IC threshold."""
    peak = derive_peak_current(selection)
    oc_trip = derive_oc_trip(selection, dischg_oc_detection_v)

    return ElectricalLimits(
        continuous_current_a=selection.continuous_current_a,
        peak_current_a=peak,
        peak_duration_s=peak_duration_s,
        ambient_temp_c=ambient_temp_c,
        max_temp_rise_c=max_temp_rise_c,
        overcurrent_trip_a=oc_trip,
    )


# ―――――――――――――――――――――――――――――――――――――――――――――――――――――――――
#  Legacy alias (backward compat)
# ―――――――――――――――――――――――――――――――――――――――――――――――――――――――――


def select_mosfets(
    limits: ElectricalLimits,
    options: list[MosfetOption] | None = None,
    max_packages: int = 16,
) -> MosfetSelection:
    """Find the minimum package count that meets the given current limits."""
    options = options or [MOSFET_DATABASE[DEFAULT_MOSFET_MPN]]
    candidates: list[MosfetSelection] = []
    for option in options:
        switches = 2 if option.dual_series_switch else 1
        hot_path_r = option.rds_on_max_ohm * option.rds_temperature_factor * switches
        for count in range(1, max_packages + 1):
            per_pkg = limits.continuous_current_a / count
            loss = per_pkg ** 2 * hot_path_r
            rise = loss * option.thermal_resistance_c_per_w
            peak_pkg = limits.peak_current_a / count
            if rise <= limits.max_temp_rise_c and peak_pkg <= option.pulse_current_a * 0.7:
                candidates.append(
                    MosfetSelection(
                        option=option,
                        package_count=count,
                        hot_resistance_per_path_ohm=hot_path_r / count,
                        loss_per_package_w=loss,
                        estimated_temp_rise_c=rise,
                        continuous_current_a=limits.continuous_current_a,
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
