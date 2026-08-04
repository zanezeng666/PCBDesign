"""元器件识别模块

这个模块负责从PCB图像中识别元器件，包括IC、电阻、电容等。

主要功能：
  1. IC芯片识别 (型号/丝印)
  2. 元器件定位
  3. 元器件方向检测
  4. 元器件引脚映射

流程：
  输入: PCB图像 + 焊盘位置
  输出: 元器件列表 [{"type": "IC", "model": "DW01", "position": {...}, ...}, ...]
"""

from __future__ import annotations

import base64
import json
import logging
import os
from pathlib import Path
from typing import Literal

import cv2
import numpy as np

from .logger import get_logger, log_function, log_errors

_log = get_logger(__name__)

# 导入VLM检测功能
from .vlm_detection import detect_components as _detect_components


@log_function(level=logging.INFO, include_args=True)
def detect_components_on_pcb(
    transparent_pcb_b64: str,
    pads: list[dict],
    side: Literal["front", "back"] = "front",
    pixels_per_mm: float = 10.0,
) -> dict:
    """元器件识别

    Args:
        transparent_pcb_b64: 透明PCB图像 (base64)
        pads: 焊盘列表 (用于关联元器件引脚)
        side: 板面朝向 ("front"|"back")
        pixels_per_mm: 像素密度

    Returns:
        {
            "components": [
                {
                    "id": str,
                    "type": "IC"|"Resistor"|"Capacitor"|"...",
                    "model": str,       # 型号/丝印
                    "manufacturer": str,
                    "position": {"x_mm": ..., "y_mm": ...},
                    "rotation_deg": float,
                    "pins": [           # 引脚映射
                        {
                            "pin_number": int,
                            "pad_id": str,  # 关联的焊盘ID
                            "function": str, # 引脚功能 (VCC/GND/...)
                        },
                        ...
                    ],
                    "footprint": str,   # 封装 (SOT-23-6/SOP-8/...)
                    "confidence": float,
                },
                ...
            ],
            "component_count": int,
            "side": "front"|"back",
        }
    """
    _log.info(
        "元器件识别: 开始 (side=%s, %d 个焊盘)",
        side,
        len(pads),
    )

    # ── Step 1: VLM识别元器件 ──
    try:
        vlm_result = _detect_components(
            transparent_pcb_b64,
            side=side,
            pixels_per_mm=pixels_per_mm,
        )

        components_raw = vlm_result.get("components", [])
        _log.info("元器件识别: VLM识别到 %d 个元器件", len(components_raw))

    except Exception as e:
        _log.error("元器件识别: VLM识别失败 - %s", e)
        raise RuntimeError(f"VLM component detection failed: {e}")

    # ── Step 2: 元器件引脚映射 ──
    components_mapped = []
    for i, comp_raw in enumerate(components_raw):
        try:
            comp_mapped = _map_component_pins(comp_raw, pads)
            comp_mapped["id"] = f"comp_{i:03d}"
            components_mapped.append(comp_mapped)

        except Exception as e:
            _log.warning("元器件识别: 第 %d 个元器件引脚映射失败 - %s", i, e)
            comp_raw["id"] = f"comp_{i:03d}"
            comp_raw["pin_mapped"] = False
            components_mapped.append(comp_raw)

    _log.info(
        "元器件识别: 完成 (%d 个元器件, %d 引脚映射成功)",
        len(components_mapped),
        sum(1 for c in components_mapped if c.get("pin_mapped", True)),
    )

    return {
        "components": components_mapped,
        "component_count": len(components_mapped),
        "side": side,
    }


def _map_component_pins(component: dict, pads: list[dict]) -> dict:
    """元器件引脚映射到焊盘

    Args:
        component: 元器件数据
        pads: 焊盘列表

    Returns:
        映射后的元器件数据
    """
    comp_position = component.get("position", {})
    comp_x = comp_position.get("x_mm", 0)
    comp_y = comp_position.get("y_mm", 0)

    # 如果元器件没有引脚信息，直接返回
    if "pins" not in component:
        component["pin_mapped"] = True
        return component

    mapped_pins = []
    for pin in component.get("pins", []):
        pin_number = pin.get("pin_number", 0)
        pin_function = pin.get("function", "?")

        # 查找最近的焊盘
        # TODO: 更精确的引脚-焊盘关联算法
        nearest_pad = None
        min_dist = float("inf")

        for pad in pads:
            pad_center = pad.get("center", {})
            pad_x = pad_center.get("x_mm", 0)
            pad_y = pad_center.get("y_mm", 0)

            dist = ((pad_x - comp_x) ** 2 + (pad_y - comp_y) ** 2) ** 0.5

            if dist < min_dist:
                min_dist = dist
                nearest_pad = pad

        mapped_pin = {
            "pin_number": pin_number,
            "function": pin_function,
            "pad_id": nearest_pad.get("id", "") if nearest_pad else "",
            "distance_mm": round(min_dist, 3),
        }

        mapped_pins.append(mapped_pin)

    result = {
        **component,
        "pins": mapped_pins,
        "pin_mapped": True,
    }

    return result


def identify_ic_model(
    ic_crop_b64: str,
    known_models: list[str] | None = None,
) -> dict:
    """IC型号识别 (基于丝印)

    Args:
        ic_crop_b64: IC裁剪图像 (base64)
        known_models: 已知型号列表 (可选，用于加速匹配)

    Returns:
        {
            "model": str,       # 型号
            "manufacturer": str,
            "marking": str,     # 丝印
            "confidence": float,
            "footprint": str,   # 封装
        }
    """
    _log.info("IC型号识别: 开始")

    # TODO: 实现基于OCR + IC数据库的型号识别
    # 当前版本依赖VLM识别，后续可优化为本地OCR + 数据库匹配

    raise NotImplementedError("IC model identification not implemented yet")


# 导出公共API
__all__ = [
    "detect_components_on_pcb",
    "identify_ic_model",
]