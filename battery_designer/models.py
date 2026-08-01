from __future__ import annotations

from enum import Enum
from math import isfinite
from typing_extensions import Annotated

from pydantic import BaseModel, Field, model_validator


# ── Battery-type → per-cell voltage lookup ──
# (min, nominal, max) in Volts
BATTERY_TYPE_VOLTAGES: dict[str, tuple[float, float, float]] = {
    "18650": (3.0, 3.7, 4.2),
    "21700": (3.0, 3.7, 4.2),
    "LiPo": (3.3, 3.7, 4.2),
    "LFP": (2.5, 3.2, 3.65),
}


class ConnectionMode(str, Enum):
    SERIES = "series"
    PARALLEL = "parallel"


class TerminalRole(str, Enum):
    BATTERY = "battery"
    CHARGE = "charge"
    DISCHARGE = "discharge"
    TEMPERATURE = "temperature"
    IDENTIFICATION = "identification"
    AUXILIARY = "auxiliary"


class Polarity(str, Enum):
    POSITIVE = "positive"
    NEGATIVE = "negative"


class BoardSide(str, Enum):
    FRONT = "front"
    BACK = "back"


class BackTransform(str, Enum):
    MIRROR_X = "mirror_x"
    MIRROR_Y = "mirror_y"
    ROTATE_180 = "rotate_180"
    NONE = "none"


class PadShape(str, Enum):
    CIRCLE = "circle"
    RECT = "rect"
    OVAL = "oval"
    ROUNDED_RECT = "rounded_rect"
    CUSTOM = "custom"


class TemplateStatus(str, Enum):
    CANDIDATE = "candidate"
    VALIDATED = "validated"


Mm = Annotated[float, Field(ge=0)]


class Point(BaseModel):
    x_mm: float
    y_mm: float

    @model_validator(mode="after")
    def finite(self) -> "Point":
        if not isfinite(self.x_mm) or not isfinite(self.y_mm):
            raise ValueError("coordinates must be finite")
        return self


class RegionBounds(BaseModel):
    x_mm: float
    y_mm: float
    width_mm: float = Field(gt=0)
    height_mm: float = Field(gt=0)


class TerminalRegion(BaseModel):
    type: str = Field(pattern=r"^(solder_pad|hole|board_outline)$")
    visual_class: str
    shape: PadShape
    center: Point
    bbox: RegionBounds
    polygon: list[Point] = Field(min_length=3, max_length=100)
    source: str


class HoleRegion(BaseModel):
    """A single hole, slot, or edge groove detected in the PCB board."""
    id: str = Field(pattern=r"^[A-Za-z0-9_-]{1,64}$")
    hole_type: str = Field(pattern=r"^(round|slot|irregular|groove|protrusion)$")
    center: Point
    bbox: RegionBounds
    polygon: list[Point] = Field(min_length=3, max_length=100)
    confidence: float = Field(ge=0, le=1, default=0.8)
    source: str = "vlm"


class BoardOutline(BaseModel):
    points: list[Point] = Field(min_length=3, max_length=500)
    source: str = "photo"
    confirmed: bool = False

    @model_validator(mode="after")
    def valid_polygon(self) -> "BoardOutline":
        raw = [(p.x_mm, p.y_mm) for p in self.points]
        if len(set(raw)) < 3:
            raise ValueError("board outline needs at least three distinct points")
        area = abs(sum(raw[i][0] * raw[(i + 1) % len(raw)][1] - raw[(i + 1) % len(raw)][0] * raw[i][1] for i in range(len(raw))) / 2)
        if area < 1.0:
            raise ValueError("board outline area must be at least 1 mm²")
        return self


class Terminal(BaseModel):
    id: str = Field(pattern=r"^[A-Za-z0-9_-]{1,32}$")
    position: Point
    roles: set[TerminalRole] = Field(min_length=1)
    polarity: Polarity | None = None
    side: BoardSide = BoardSide.FRONT
    shape: PadShape = PadShape.CIRCLE
    width_mm: float = Field(gt=0, le=50)
    height_mm: float = Field(gt=0, le=50)
    source_region: TerminalRegion | None = None

    @model_validator(mode="after")
    def role_polarity_contract(self) -> "Terminal":
        electrical = {TerminalRole.BATTERY, TerminalRole.CHARGE, TerminalRole.DISCHARGE}
        if self.roles & electrical and self.polarity is None:
            raise ValueError("battery, charge and discharge terminals require polarity")
        return self


class BatterySpec(BaseModel):
    count: int = Field(ge=1, le=5)
    connection: ConnectionMode
    battery_type: str = Field(default="18650", description=f"电芯类型: {', '.join(BATTERY_TYPE_VOLTAGES.keys())}")

    @model_validator(mode="after")
    def validate_type(self) -> "BatterySpec":
        if self.battery_type not in BATTERY_TYPE_VOLTAGES:
            raise ValueError(f"Unknown battery_type '{self.battery_type}'. Choose from: {list(BATTERY_TYPE_VOLTAGES.keys())}")
        return self

    @property
    def chemistry(self) -> str:
        return "LiFePO4" if self.battery_type == "LFP" else "Li-ion/LiPo"

    @property
    def cell_min_v(self) -> float:
        return BATTERY_TYPE_VOLTAGES[self.battery_type][0]

    @property
    def cell_nominal_v(self) -> float:
        return BATTERY_TYPE_VOLTAGES[self.battery_type][1]

    @property
    def cell_max_v(self) -> float:
        return BATTERY_TYPE_VOLTAGES[self.battery_type][2]

    @property
    def series_cells(self) -> int:
        return self.count if self.connection == ConnectionMode.SERIES else 1

    @property
    def parallel_cells(self) -> int:
        return self.count if self.connection == ConnectionMode.PARALLEL else 1


class ElectricalLimits(BaseModel):
    continuous_current_a: float = Field(gt=0, le=500)
    peak_current_a: float = Field(gt=0, le=1000)
    peak_duration_s: float = Field(gt=0, le=3600)
    ambient_temp_c: float = Field(ge=-40, le=100)
    max_temp_rise_c: float = Field(gt=0, le=80, default=40)
    overcurrent_trip_a: float | None = Field(default=None, gt=0, le=1000)

    @model_validator(mode="after")
    def peak_not_lower(self) -> "ElectricalLimits":
        if self.peak_current_a < self.continuous_current_a:
            raise ValueError("peak current cannot be lower than continuous current")
        if self.overcurrent_trip_a is not None and self.overcurrent_trip_a <= self.continuous_current_a:
            raise ValueError("overcurrent trip must be higher than continuous current")
        return self


class ManufacturingSpec(BaseModel):
    layers: int = Field(default=2, ge=2, le=2)
    copper_oz: float = Field(default=1.0, ge=0.5, le=4.0)
    min_clearance_mm: float = Field(default=0.2, ge=0.1, le=1.0)
    min_track_mm: float = Field(default=0.2, ge=0.1, le=2.0)


class PhotoCaptureSpec(BaseModel):
    front_calibration_id: str | None = Field(default=None, pattern=r"^[a-f0-9]{32}$")
    back_calibration_id: str | None = Field(default=None, pattern=r"^[a-f0-9]{32}$")
    back_transform: BackTransform = BackTransform.MIRROR_X
    alignment_error_mm: float | None = Field(default=None, ge=0)


class DetectedComponent(BaseModel):
    """元器件识别结果中的单个元器件。"""
    type: str = Field(description="ic|mosfet|resistor|capacitor|diode|ntc|led|other")
    silkscreen: str = Field(default="", description="丝印文字")
    package: str = Field(default="", description="封装")
    confidence: float = Field(default=0.5, ge=0, le=1)


class DesignSpec(BaseModel):
    name: str = Field(min_length=1, max_length=80)
    protection_ic: str = Field(min_length=2, max_length=80)
    battery: BatterySpec
    mos_count: int = Field(ge=1, le=20, description="板上MOS管封装个数")
    mos_mpn: str | None = Field(default=None, max_length=80, description="用户指定的MOS管型号（丝印或完整MPN）")
    outline: BoardOutline
    terminals: list[Terminal] = Field(min_length=4, max_length=30)
    manufacturing: ManufacturingSpec = Field(default_factory=ManufacturingSpec)
    photo_capture: PhotoCaptureSpec = Field(default_factory=PhotoCaptureSpec)
    detected_components: list[DetectedComponent] = Field(
        default_factory=list,
        description="VLM识别到的PCB板载元器件清单",
    )

    @model_validator(mode="after")
    def terminal_contract(self) -> "DesignSpec":
        ids = [t.id for t in self.terminals]
        if len(ids) != len(set(ids)):
            raise ValueError("terminal ids must be unique")
        for role in (TerminalRole.BATTERY, TerminalRole.CHARGE, TerminalRole.DISCHARGE):
            for polarity in Polarity:
                if not any(role in t.roles and t.polarity == polarity for t in self.terminals):
                    raise ValueError(f"missing {role.value} {polarity.value} terminal")
        for terminal in self.terminals:
            if not point_in_polygon(terminal.position, self.outline.points):
                raise ValueError(f"terminal {terminal.id} is outside the board outline")
        if any(terminal.side == BoardSide.BACK for terminal in self.terminals):
            if not self.photo_capture.back_calibration_id:
                raise ValueError("back-side terminals require a calibrated back-side photo")
            if self.photo_capture.alignment_error_mm is not None and self.photo_capture.alignment_error_mm > 0.5:
                raise ValueError("front/back photo alignment error exceeds 0.5 mm")
        return self

    @property
    def port_topology(self) -> str:
        shared = {}
        for polarity in Polarity:
            shared[polarity] = any(
                TerminalRole.CHARGE in terminal.roles
                and TerminalRole.DISCHARGE in terminal.roles
                and terminal.polarity == polarity
                for terminal in self.terminals
            )
        if all(shared.values()):
            return "common"
        if not any(shared.values()):
            return "separate"
        return "hybrid"


class ValidationRecord(BaseModel):
    overcharge_passed: bool
    overdischarge_passed: bool
    continuous_current_passed: bool
    peak_current_passed: bool
    short_circuit_recovery_passed: bool
    thermal_passed: bool
    notes: str = ""

    @property
    def passed(self) -> bool:
        return all(
            [
                self.overcharge_passed,
                self.overdischarge_passed,
                self.continuous_current_passed,
                self.peak_current_passed,
                self.short_circuit_recovery_passed,
                self.thermal_passed,
            ]
        )


def point_in_polygon(point: Point, polygon: list[Point]) -> bool:
    inside = False
    j = len(polygon) - 1
    for i, vertex in enumerate(polygon):
        previous = polygon[j]
        crosses = (vertex.y_mm > point.y_mm) != (previous.y_mm > point.y_mm)
        if crosses:
            boundary_x = (previous.x_mm - vertex.x_mm) * (point.y_mm - vertex.y_mm) / (previous.y_mm - vertex.y_mm) + vertex.x_mm
            if point.x_mm < boundary_x:
                inside = not inside
        j = i
    return inside
