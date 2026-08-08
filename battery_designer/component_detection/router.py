"""Step 3 — Component detection API endpoints.

Contains:
- ``/api/vision/detect-components`` — VLM component detection
- ``/api/ic/resolve`` — IC catalog resolution
- ``/api/cell/lookup`` — AI cell parameter lookup
- ``/api/mos/list`` — MOSFET database listing

Moved from ``pcb_recognition/router.py`` (detect-components)
and ``routers/ic_catalog.py`` (ic/resolve, cell/lookup, mos/list).
"""

from __future__ import annotations

import json
import logging

from fastapi import APIRouter, Request, Form

from ..core.errors import DesignError
from ..core.config import WORK_ROOT
from ..core.singletons import catalog as _catalog_singleton
from ..core.vlm_client import get_api_key as _get_api_key, extract_json as _extract_json
from .catalog import normalize_mpn, first_marking_line
from .mos import list_available_mpns
from .vlm_components import detect_components as _detect_components
from ..board_recognition.calibration_utils import _load_calibration

logger = logging.getLogger(__name__)

router = APIRouter()

# Use the shared singleton catalog
catalog = _catalog_singleton


# ══════════════════════════════════════════════════════════════════════
#  Component detection endpoint
# ══════════════════════════════════════════════════════════════════════

@router.post("/api/vision/detect-components")
def api_detect_components(calibration_id: str = Form(...), side: str = Form(...)):
    """Detect electronic components on the PCB using the original rectified image.

    Uses the unmodified rectified PNG (not cropped/transparent) to preserve
    full visual context for reading tiny silkscreen text on ICs, MOSFETs,
    resistors, capacitors, etc.

    Returns:
        - components: list of detected components with type/silkscreen/package/position
        - inferred_ic: auto-inferred IC model from detected IC silkscreen
        - inferred_mos: auto-inferred MOS model from detected MOSFET silkscreen
    """
    if side not in ("front", "back"):
        raise DesignError("INVALID_BOARD_SIDE", "side must be front or back")

    # Load calibration metadata and rectified PNG (original image, not transparent)
    png, w_mm, h_mm, ppm = _load_calibration(calibration_id)

    # Call VLM component detection on the original rectified image
    result = _detect_components(png, w_mm, h_mm, side)

    # Save result
    directory = WORK_ROOT / "calibrations" / calibration_id
    (directory / "components.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    return result


# ══════════════════════════════════════════════════════════════════════
#  IC catalog endpoints
# ══════════════════════════════════════════════════════════════════════

@router.get("/api/ic/resolve")
def resolve_ic(model: str):
    """Resolve IC model/MPN from user input (may be marking code or MPN)."""
    device = catalog.resolve(model)
    result = device.as_dict()
    requested_full = normalize_mpn(model)
    requested_first = normalize_mpn(first_marking_line(model))
    marking_line = normalize_mpn(device.marking.get("first_line", ""))
    is_marking = bool(marking_line) and requested_full != normalize_mpn(device.full_mpn) and (
        requested_full == marking_line or requested_first == marking_line
    )
    result["resolved_from"] = "marking" if is_marking else "mpn_or_alias"
    result["input_was_marking"] = is_marking
    return result


@router.post("/api/cell/lookup")
async def cell_lookup(request: Request):
    """Look up cell parameters by manufacturer + model using AI.

    Input: {"manufacturer": "Samsung", "model": "INR18650-25R"}
    Output: cell parameters (capacity, voltages, currents, chemistry, etc.)
    """
    body = await request.json()
    manufacturer = str(body.get("manufacturer", "")).strip()
    model = str(body.get("model", "")).strip()
    if not model:
        raise DesignError("INPUT_VALIDATION", "电芯型号不能为空")

    cell_id = f"{manufacturer}_{model}".replace(" ", "_").replace("/", "_").replace("\\", "_")
    cache_path = WORK_ROOT / "cell_cache" / f"{cell_id}.json"
    if cache_path.exists():
        return json.loads(cache_path.read_text(encoding="utf-8"))

    params = _ai_cell_lookup(manufacturer, model)
    if params is None and "/" in model:
        alt_model = model.replace("/", "-")
        logger.info("cell lookup: retry with normalized model '%s' -> '%s'", model, alt_model)
        params = _ai_cell_lookup(manufacturer, alt_model)

    if params is None:
        raise DesignError("CELL_LOOKUP_FAILED", "AI 未能获取该电芯的参数信息，请检查型号是否正确")

    params["model"] = model
    params["lookup_model_input"] = {"manufacturer": manufacturer, "model": model}

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(params, ensure_ascii=False, indent=2), encoding="utf-8")
    return params


def _ai_cell_lookup(manufacturer: str, model: str) -> dict | None:
    """Call qwen text model to retrieve lithium cell parameters."""
    try:
        import dashscope
        from dashscope import Generation
    except ImportError:
        logger.warning("dashscope SDK not installed — cell lookup unavailable")
        return None

    api_key = _get_api_key()
    if not api_key:
        logger.warning("DASHSCOPE_API_KEY not set — cell lookup unavailable")
        return None

    dashscope.api_key = api_key
    prompt = f"""你是锂电池电芯参数数据库。请提供以下电芯用于【保护板（BMS/PCM）设计】的关键参数：
厂商: {manufacturer or '未知'}
型号: {model}

只需要与保护板设计相关的参数（电压/电流/温度保护阈值、内阻、容量），不需要尺寸/重量/认证/运输/寿命等无关信息。
返回纯JSON对象（不要markdown），尽可能填满以下字段：
{{
  "manufacturer": "厂商标称",
  "model": "完整型号",
  "chemistry": "Li-ion 或 LiPo 或 LiFePO4 或 NCA/NCM/LCO 等（决定保护电压阈值）",
  "form_factor": "如 18650 / 21700 / 26650 / 软包",
  "nominal_capacity_mah": 数值,
  "min_capacity_mah": 数值,
  "nominal_voltage_v": 数值,
  "charge_cutoff_voltage_v": 数值,
  "discharge_cutoff_voltage_v": 数值,
  "standard_charge_current_a": 数值,
  "max_charge_current_a": 数值,
  "standard_discharge_current_a": 数值,
  "max_continuous_discharge_a": 数值,
  "max_pulse_discharge_a": 数值,
  "internal_resistance_mohm": 数值,
  "operating_temp_charge_c": [最低, 最高],
  "operating_temp_discharge_c": [最低, 最高],
  "notes": "简要补充说明（注明哪些字段是估计值，保持简短）"
}}

要求：
- 优先采用官方数据手册典型值；手册中不确定的字段，基于同系列/同规格电芯给出合理估计值即可
- 确实无法确定的个别字段填 null 即可，不要因为部分字段未知就返回 error
- 型号中的 "/" 与 "-" 等价（如 INR18650/26V 即 INR18650-26V），两种写法都要尝试识别
- 只要能确认该型号电芯存在（例如能从型号解读出尺寸18650和容量26V≈2600mAh），就必须返回参数对象，不要返回 error
- 仅当型号完全无法识别、明显不存在时，才返回 {{"error": "型号未找到"}}
- 只返回JSON，不要其他文字"""

    try:
        resp = Generation.call(
            model="qwen-plus",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=1024,
            result_format="message",
        )
    except Exception as exc:
        logger.error("Cell lookup API error: %s", exc)
        return None

    if resp.status_code != 200:
        logger.error("Cell lookup API status: %s", getattr(resp, 'code', '?'))
        return None

    try:
        raw = resp.output.choices[0].message.content
        if isinstance(raw, list):
            raw = "".join(p.get("text", "") for p in raw if isinstance(p, dict))
    except Exception as exc:
        logger.error("Cell lookup response parse: %s", exc)
        return None

    parsed = _extract_json(raw)
    if parsed is None or "error" in parsed:
        return None

    if not parsed.get("nominal_capacity_mah"):
        return None

    parsed["lookup_source"] = "ai"
    parsed["lookup_model_input"] = {"manufacturer": manufacturer, "model": model}
    return parsed


@router.get("/api/mos/list")
def list_mos():
    """Return available MOSFET MPNs for the IC catalog dropdown."""
    return [{"model": mpn} for mpn in list_available_mpns()]
