from __future__ import annotations

import base64
import re
from functools import lru_cache

import cv2
import numpy as np

from .errors import DesignError


LABEL_CONTRACT = {
    "B+": ({"battery"}, "positive"),
    "B-": ({"battery"}, "negative"),
    "P+": ({"charge", "discharge"}, "positive"),
    "P-": ({"charge", "discharge"}, "negative"),
    "C+": ({"charge"}, "positive"),
    "C-": ({"charge"}, "negative"),
    "NTC": ({"temperature"}, None),
    "TH": ({"temperature"}, None),
    "N": ({"temperature"}, None),
    "ID": ({"identification"}, None),
}


def detect_terminal_candidates(
    rectified_png: bytes, width_mm: float, height_mm: float, side: str
) -> dict:
    image = cv2.imdecode(np.frombuffer(rectified_png, np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise DesignError("INVALID_CALIBRATION_IMAGE", "The rectified calibration image cannot be read.")
    if side not in {"front", "back"}:
        raise DesignError("INVALID_BOARD_SIDE", "side must be front or back")

    # Stage 1: recognize the finite set of terminal labels.  Pad geometry is
    # deliberately not used to decide which text exists.
    observations: list[dict] = []
    engine = _ocr_engine()
    if engine is not None:
        observations.extend(_full_ocr_observations(engine, image))
        observations.extend(_edge_text_observations(engine, image))

    # Stage 2: find metallic regions and holes, remove regions that are really
    # OCR text strokes, then associate each label with the nearest region.
    raw_pads = _find_pad_candidates(image, width_mm, height_mm)
    banks = _find_edge_pad_banks(image, width_mm, height_mm)
    raw_pads.extend(pad for bank in banks for pad in bank["pads"])
    all_pads = _deduplicate_pads(raw_pads)
    filtered_pads = _exclude_text_regions(list(all_pads), observations)

    img_h, img_w = image.shape[:2]
    candidates = _observations_to_candidates(observations, filtered_pads, width_mm, height_mm, side, img_w, img_h)

    matched_indices: set[int] = set()
    for candidate in candidates:
        if not candidate.get("region_resolved") or not candidate.get("visible_region"):
            continue
        center = candidate["visible_region"]["center"]
        cx, cy = center["x_mm"], center["y_mm"]
        for i, pad in enumerate(all_pads):
            px_mm = pad["x_px"] / img_w * width_mm
            py_mm = pad["y_px"] / img_h * height_mm
            if abs(px_mm - cx) < 1.0 and abs(py_mm - cy) < 1.0:
                matched_indices.add(i)
                break

    symmetric_pairs = _find_symmetric_pads(all_pads)
    pair_membership: dict[int, int] = {}
    for pair_id, (a, b) in enumerate(symmetric_pairs):
        pair_membership[a] = pair_id
        pair_membership[b] = pair_id

    for candidate in candidates:
        if not candidate.get("region_resolved") or not candidate.get("visible_region"):
            continue
        center = candidate["visible_region"]["center"]
        cx, cy = center["x_mm"], center["y_mm"]
        for i, pad in enumerate(all_pads):
            px_mm = pad["x_px"] / img_w * width_mm
            py_mm = pad["y_px"] / img_h * height_mm
            if abs(px_mm - cx) < 1.0 and abs(py_mm - cy) < 1.0 and i in pair_membership:
                pair_id = pair_membership[i]
                i0, i1 = symmetric_pairs[pair_id]
                other_i = i1 if i == i0 else i0
                other_pad = all_pads[other_i]
                other_center = {
                    "x_mm": round(other_pad["x_px"] / img_w * width_mm, 3),
                    "y_mm": round(other_pad["y_px"] / img_h * height_mm, 3),
                }
                candidate["symmetric_pair"] = {
                    "pair_id": pair_id,
                    "partner_center_mm": other_center,
                }
                break

    symmetric_pair_indices: set[int] = set()
    for a, b in symmetric_pairs:
        symmetric_pair_indices.add(a)
        symmetric_pair_indices.add(b)
    annotation_png = _generate_pad_annotation(image, all_pads, matched_indices, symmetric_pair_indices)
    return {
        "side": side,
        "width_mm": width_mm,
        "height_mm": height_mm,
        "ocr_available": engine is not None,
        "candidate_count": len(candidates),
        "candidates": candidates,
        "all_pad_count": len(all_pads),
        "matched_pad_count": len(matched_indices),
        "symmetric_pair_count": len(symmetric_pairs),
        "annotated_png_base64": base64.b64encode(annotation_png).decode("ascii"),
        "notice": "候选标注必须逐项人工确认；丝印识别不会自动建立内部电气连接。",
    }


@lru_cache(maxsize=1)
def _ocr_engine():
    try:
        from rapidocr import RapidOCR
    except (ImportError, OSError):
        return None
    return RapidOCR()


def _full_ocr_observations(engine, image: np.ndarray) -> list[dict]:
    observations: list[dict] = []
    height, width = image.shape[:2]
    for rotation in (0, 180):
        working = image if rotation == 0 else cv2.rotate(image, cv2.ROTATE_180)
        result = engine(working, use_det=True, use_rec=True, use_cls=False)
        if not hasattr(result, "boxes") or result.boxes is None:
            continue
        for box, raw_text, score in zip(result.boxes, result.txts, result.scores):
            label, normalization_penalty = _normalize_label(raw_text, targeted=False)
            if label is None:
                continue
            center = np.asarray(box, dtype=float).mean(axis=0)
            x, y = float(center[0]), float(center[1])
            transformed_box = np.asarray(box, dtype=float)
            if rotation == 180:
                x, y = width - x, height - y
                transformed_box = np.column_stack((width - transformed_box[:, 0], height - transformed_box[:, 1]))
            bx, by, bw, bh = cv2.boundingRect(transformed_box.astype(np.float32))
            observations.append(
                {
                    "label": label,
                    "raw_text": raw_text,
                    "x_px": x,
                    "y_px": y,
                    "bbox_px": [float(bx), float(by), float(bw), float(bh)],
                    "confidence": float(score) * normalization_penalty,
                    "method": "ocr",
                }
            )
    return observations


def _edge_text_observations(engine, image: np.ndarray) -> list[dict]:
    """Recognition-only sliding windows; independent from detected pad rows."""
    observations: list[dict] = []
    height, width = image.shape[:2]
    window_height = max(24, int(height * 0.18))
    centers = np.arange(window_height / 2, height - window_height / 2 + 1, max(12.0, height * 0.04))
    for edge, x0, x1 in (("left", 0.045, 0.16), ("right", 0.84, 0.955)):
        for center_y in centers:
            y0 = max(0, int(center_y - window_height / 2))
            y1 = min(height, y0 + window_height)
            px0, px1 = int(width * x0), int(width * x1)
            crop = image[y0:y1, px0:px1]
            if crop.size == 0:
                continue
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            gray = cv2.createCLAHE(2.0, (4, 4)).apply(gray)
            gray = cv2.resize(gray, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
            result = engine(gray, use_det=False, use_cls=False)
            if not result.txts:
                continue
            raw_text, score = result.txts[0], float(result.scores[0])
            label, penalty = _normalize_label(raw_text, targeted=True)
            if label is None:
                continue
            observations.append(
                {
                    "label": label,
                    "raw_text": raw_text,
                    "x_px": (px0 + px1) / 2,
                    "y_px": (y0 + y1) / 2,
                    "bbox_px": [float(px0), float(center_y - height * 0.06), float(px1 - px0), float(height * 0.12)],
                    "bbox_is_window": True,
                    "confidence": score * penalty,
                    "method": f"ocr-{edge}-strip",
                }
            )
    return observations


def _bank_ocr_observations(engine, image: np.ndarray, banks: list[dict]) -> list[dict]:
    observations: list[dict] = []
    height, width = image.shape[:2]
    for bank in banks:
        edge = bank["edge"]
        if not bank["pads"] or not all(pad["method"].startswith("split-") for pad in bank["pads"]):
            continue
        for pad in bank["pads"]:
            x, y = pad["x_px"], pad["y_px"]
            if edge == "right":
                crop = image[max(0, int(y - height * 0.09)) : min(height, int(y + height * 0.09)), int(width * 0.84) : int(width * 0.955)]
            elif edge == "left":
                crop = image[max(0, int(y - height * 0.09)) : min(height, int(y + height * 0.09)), int(width * 0.045) : int(width * 0.16)]
            elif edge == "top":
                crop = image[int(height * 0.045) : int(height * 0.2), max(0, int(x - width * 0.08)) : min(width, int(x + width * 0.08))]
            else:
                crop = image[int(height * 0.8) : int(height * 0.955), max(0, int(x - width * 0.08)) : min(width, int(x + width * 0.08))]
            if crop.size == 0:
                continue
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            gray = cv2.createCLAHE(2.0, (4, 4)).apply(gray)
            gray = cv2.resize(gray, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
            result = engine(gray, use_det=False, use_cls=False)
            if not result.txts:
                continue
            raw_text, score = result.txts[0], float(result.scores[0])
            label, normalization_penalty = _normalize_label(raw_text, targeted=True)
            if label is None:
                continue
            observations.append(
                {
                    "label": label,
                    "raw_text": raw_text,
                    "x_px": x,
                    "y_px": y,
                    "confidence": score * normalization_penalty,
                    "method": "ocr+edge-pad",
                    "pad": pad,
                }
            )
    return observations


def _normalize_label(raw_text: str, targeted: bool) -> tuple[str | None, float]:
    text = re.sub(r"[^A-Z0-9+\-]", "", raw_text.upper().replace("—", "-").replace("_", "-"))
    for label in ("NTC", "B+", "B-", "P+", "P-", "C+", "C-", "TH", "ID"):
        if text == label:
            return label, 1.0
    if text == "N":
        return "N", 0.75
    if text in {"1D", "LD"}:
        return "ID", 0.8
    if text in {"BL", "B-L", "B--"}:
        return "B-", 0.72
    if text in {"+8", "8+"}:
        return "B+", 0.58
    if targeted and text in {"PS", "P5"}:
        return "P-", 0.55
    if targeted and text.startswith("TH"):
        return "TH", 0.72
    if targeted and text in {"P4", "PT", "P十"}:
        return "P+", 0.62
    if targeted and text in {"PEC", "PE", "PC"}:
        return "P-", 0.48
    return None, 0.0


def _find_edge_pad_banks(image: np.ndarray, width_mm: float, height_mm: float) -> list[dict]:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # Same metal mask as _find_pad_candidates, plus adaptive for local contrast
    metal_strip = cv2.inRange(hsv, (0, 0, 40), (180, 55, 255))
    adaptive_strip = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 71, 10)
    bright = cv2.bitwise_or(metal_strip, adaptive_strip)
    height, width = bright.shape
    banks: list[dict] = []
    component_pads = _find_pad_candidates(image, width_mm, height_mm)
    for edge in ("left", "right"):
        split_pads = [pad for pad in component_pads if pad["method"] == f"split-{edge}-edge-pad"]
        if len(split_pads) >= 2:
            banks.append({"edge": edge, "pads": sorted(split_pads, key=lambda pad: pad["y_px"])})
    strip_x = max(8, int(width * 0.065))
    strip_y = max(8, int(height * 0.10))
    for edge, strip, axis in (
        ("left", bright[:, :strip_x], 1),
        ("right", bright[:, width - strip_x :], 1),
        ("top", bright[:strip_y, :], 0),
        ("bottom", bright[height - strip_y :, :], 0),
    ):
        if any(bank["edge"] == edge for bank in banks):
            continue
        projection = (strip > 0).sum(axis=axis).astype(np.float32)
        kernel = max(3, int(len(projection) * 0.018) | 1)
        smooth = np.convolve(projection, np.ones(kernel) / kernel, mode="same")
        threshold = max(3.0, 0.14 * (strip.shape[1] if axis == 1 else strip.shape[0]))
        segments = _segments_above(smooth, threshold)
        minimum_length = max(4, int(len(projection) * 0.018))
        segments = [(start, end) for start, end in segments if end - start >= minimum_length]
        if not 2 <= len(segments) <= 10:
            continue
        pads = []
        for start, end in segments:
            center = (start + end) / 2
            if center < 0.035 * len(projection) or center > 0.965 * len(projection):
                continue
            if edge == "left":
                x, y = strip_x * 0.45, center
            elif edge == "right":
                x, y = width - strip_x * 0.45, center
            elif edge == "top":
                x, y = center, strip_y * 0.45
            else:
                x, y = center, height - strip_y * 0.45
            pads.append(_pad(x, y, max(0.8, (end - start) / height * height_mm), max(0.8, (end - start) / height * height_mm), "edge-bank"))
        if len(pads) >= 2:
            banks.append({"edge": edge, "pads": pads})
    return banks


def _segments_above(values: np.ndarray, threshold: float) -> list[tuple[int, int]]:
    active = values >= threshold
    changes = np.diff(np.r_[False, active, False].astype(np.int8))
    starts, ends = np.flatnonzero(changes == 1), np.flatnonzero(changes == -1)
    return list(zip(starts.tolist(), ends.tolist()))


def _find_pad_candidates(image: np.ndarray, width_mm: float, height_mm: float) -> list[dict]:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    height, width = gray.shape
    candidates: list[dict] = []

    # Metallic surfaces have very low colour saturation regardless of brightness.
    # Adaptive threshold catches locally-bright regions against darker PCB substrate.
    metal = cv2.inRange(hsv, (0, 0, 40), (180, 55, 255))
    adaptive = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 71, 10)
    combined = cv2.bitwise_or(metal, adaptive)
    combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))

    contours, _ = cv2.findContours(combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < 0.0002 * width * height or area > 0.15 * width * height:
            continue
        x, y, w, h = cv2.boundingRect(contour)
        if w < 8 or h < 8:
            continue

        perimeter = cv2.arcLength(contour, True)
        if perimeter <= 0:
            continue
        circularity = 4 * np.pi * area / (perimeter * perimeter)
        rotated_rect = cv2.minAreaRect(contour)
        rw, rh = rotated_rect[1]
        if min(rw, rh) < 1:
            continue
        fill_ratio = area / max(rw * rh, 1)
        aspect = max(w, h) / max(min(w, h), 1)

        # Accept shapes typical of pads:
        #   rectangular:      fill >= 0.60, circularity < 0.85
        #   rounded-rect:     fill >= 0.70, circularity >= 0.40
        #   circular / hole:  circularity >= 0.55, fill >= 0.45, aspect < 2.5
        #   thin strip:       fill >= 0.50, aspect >= 2.5, circularity < 0.60
        shape_ok = (
            (fill_ratio >= 0.60 and circularity < 0.85)
            or (fill_ratio >= 0.70 and circularity >= 0.40)
            or (circularity >= 0.55 and fill_ratio >= 0.45 and aspect < 2.5)
            or (fill_ratio >= 0.50 and aspect >= 2.5 and circularity < 0.60)
        )

        center_x, center_y = x + w / 2, y + h / 2
        edge_distance = min(center_x, width - center_x, center_y, height - center_y)
        if edge_distance > 0.40 * min(width, height) and min(center_x, width - center_x) > 0.38 * width:
            continue

        component_width_mm = w / width * width_mm
        component_height_mm = h / height * height_mm

        min_pad_mm = 1.5
        if component_width_mm < min_pad_mm and component_height_mm < min_pad_mm:
            continue

        avg_val = float(np.mean(hsv[y : y + h, x : x + w, 2]))
        component_area_mm2 = component_width_mm * component_height_mm
        if avg_val > 210 and component_area_mm2 < 4.0 and fill_ratio < 0.80:
            continue

        region_type = _classify_region(gray[y : y + h, x : x + w], hsv[y : y + h, x : x + w], circularity)

        if not shape_ok:
            if fill_ratio < 0.40:
                continue
            region_type = "irregular"

        if region_type == "hole" and max(component_width_mm, component_height_mm) < 0.7:
            continue
        if region_type == "solder_pad" and component_width_mm * component_height_mm < 0.35:
            continue

        # Tall edge-pad splitting (common B+/B- double-row connectors)
        if (
            2.4 <= component_height_mm <= 5.0
            and component_width_mm >= 3.5
            and (x <= 2 or x + w >= width - 2)
        ):
            edge = "left" if x <= 2 else "right"
            for offset in (-0.25, 0.25):
                split_y = center_y + offset * h
                split_h = h * 0.48
                split_w = min(w * 0.52, width * 2.8 / width_mm)
                split_x = max(center_x, width - split_w) if edge == "right" else 0.0
                bbox = [split_x, split_y - split_h / 2, split_w, split_h]
                candidates.append(
                    _pad(
                        split_x + split_w / 2,
                        split_y,
                        min(2.8, component_width_mm * 0.55),
                        min(1.8, component_height_mm * 0.48),
                        f"split-{edge}-edge-pad",
                        bbox_px=bbox,
                        polygon_px=_bbox_polygon(bbox),
                        visual_class="metallic",
                    )
                )
            continue

        candidates.append(
            _pad(
                center_x,
                center_y,
                max(0.5, component_width_mm),
                max(0.5, component_height_mm),
                "multi-mask",
                bbox_px=[float(x), float(y), float(w), float(h)],
                polygon_px=cv2.approxPolyDP(contour, max(1.0, 0.01 * perimeter), True).reshape(-1, 2).astype(float).tolist(),
                region_type=region_type,
                visual_class="metallic",
            )
        )
    return candidates


def _pad(
    x_px: float,
    y_px: float,
    width_mm: float,
    height_mm: float,
    method: str,
    *,
    bbox_px: list[float] | None = None,
    polygon_px: list[list[float]] | None = None,
    region_type: str = "solder_pad",
    visual_class: str = "estimated",
) -> dict:
    return {
        "x_px": float(x_px),
        "y_px": float(y_px),
        "width_mm": width_mm,
        "height_mm": height_mm,
        "method": method,
        "bbox_px": bbox_px,
        "polygon_px": polygon_px,
        "region_type": region_type,
        "visual_class": visual_class,
    }


def _bbox_polygon(bbox: list[float]) -> list[list[float]]:
    x, y, width, height = bbox
    return [[x, y], [x + width, y], [x + width, y + height], [x, y + height]]


def _classify_region(gray_patch: np.ndarray, hsv_patch: np.ndarray, circularity: float) -> str:
    avg_val = float(np.mean(hsv_patch[:, :, 2]))
    if avg_val < 55 and circularity > 0.35:
        return "hole"
    return "solder_pad"


def _deduplicate_pads(pads: list[dict]) -> list[dict]:
    result: list[dict] = []
    for pad in sorted(pads, key=lambda item: (item["bbox_px"] is not None, item["width_mm"] * item["height_mm"]), reverse=True):
        if any(np.hypot(pad["x_px"] - other["x_px"], pad["y_px"] - other["y_px"]) < 18 for other in result):
            continue
        result.append(pad)
    return result


def _exclude_text_regions(regions: list[dict], observations: list[dict]) -> list[dict]:
    text_boxes = [
        observation.get("bbox_px")
        for observation in observations
        if observation.get("bbox_px") and not observation.get("bbox_is_window")
    ]
    result = []
    for region in regions:
        bbox = region.get("bbox_px")
        if bbox is None:
            continue
        if any(_intersection_ratio(bbox, text_box) >= 0.45 for text_box in text_boxes):
            continue
        px = region.get("x_px")
        py = region.get("y_px")
        if px is not None and py is not None:
            if any(_point_in_bbox(px, py, text_box) for text_box in text_boxes):
                continue
        result.append(region)
    return result


def _intersection_ratio(first: list[float], second: list[float]) -> float:
    ax, ay, aw, ah = first
    bx, by, bw, bh = second
    overlap_w = max(0.0, min(ax + aw, bx + bw) - max(ax, bx))
    overlap_h = max(0.0, min(ay + ah, by + bh) - max(ay, by))
    return overlap_w * overlap_h / max(1.0, min(aw * ah, bw * bh))


def _point_in_bbox(px: float, py: float, bbox: list[float]) -> bool:
    bx, by, bw, bh = bbox
    return bx <= px <= bx + bw and by <= py <= by + bh


def _observations_to_candidates(
    observations: list[dict], pads: list[dict], width_mm: float, height_mm: float, side: str, image_width: int, image_height: int
) -> list[dict]:
    if not observations:
        return []
    candidates: list[dict] = []
    for observation in observations:
        pad = _nearest_pad(observation, pads, image_width, image_height)
        x_px = pad["x_px"] if pad else observation["x_px"]
        y_px = pad["y_px"] if pad else observation["y_px"]
        x_mm = min(width_mm, max(0.0, x_px / image_width * width_mm))
        y_mm = min(height_mm, max(0.0, y_px / image_height * height_mm))
        roles, polarity = LABEL_CONTRACT[observation["label"]]
        confidence = min(0.99, observation["confidence"] * (1.0 if pad else 0.78))
        region = _visible_region(pad, width_mm, height_mm, image_width, image_height) if pad else None
        text_region = _visible_text_region(observation, width_mm, height_mm, image_width, image_height)
        distance_mm = None if pad is None else _region_distance_px(observation, pad) * width_mm / image_width
        candidate = {
            "id": _candidate_id(observation["label"]),
            "label": observation["label"],
            "recognized_text": observation["raw_text"],
            "visible_position": {"x_mm": round(x_mm, 3), "y_mm": round(y_mm, 3)},
            "visible_region": region,
            "text_region": text_region,
            "roles": sorted(roles),
            "polarity": polarity,
            "side": side,
            "shape": region["shape"] if region else "circle",
            "width_mm": round(min(12.0, pad["width_mm"]), 3) if pad else None,
            "height_mm": round(min(12.0, pad["height_mm"]), 3) if pad else None,
            "confidence": round(confidence, 3),
            "method": f"{observation['method']}→nearest-{pad['region_type']}" if pad else observation["method"] + "→no-region",
            "region_resolved": pad is not None,
            "match_distance_mm": None if distance_mm is None else round(float(distance_mm), 3),
            "requires_confirmation": True,
        }
        existing = next((item for item in candidates if item["label"] == candidate["label"] and _candidate_distance(item, candidate) < 1.5), None)
        if existing is None:
            candidates.append(candidate)
        elif candidate["confidence"] > existing["confidence"]:
            candidates[candidates.index(existing)] = candidate
    return sorted(candidates, key=lambda item: (item["visible_position"]["x_mm"], item["visible_position"]["y_mm"], item["label"]))


def _nearest_pad(observation: dict, pads: list[dict], image_width: float, image_height: float) -> dict | None:
    pads = [pad for pad in pads if pad.get("bbox_px") and pad.get("region_type") in {"solder_pad", "hole"}]
    if not pads:
        return None
    ranked = sorted(pads, key=lambda pad: _region_distance_px(observation, pad))
    distance = _region_distance_px(observation, ranked[0])
    return ranked[0] if distance <= 0.22 * max(image_width, image_height) else None


def _region_distance_px(observation: dict, region: dict) -> float:
    text_box = observation.get("bbox_px") or [observation["x_px"], observation["y_px"], 1.0, 1.0]
    region_box = region["bbox_px"]
    tx, ty, tw, th = text_box
    rx, ry, rw, rh = region_box
    dx = max(rx - (tx + tw), tx - (rx + rw), 0.0)
    dy = max(ry - (ty + th), ty - (ry + rh), 0.0)
    return float(np.hypot(dx, dy))


def _visible_region(region: dict, width_mm: float, height_mm: float, image_width: int, image_height: int) -> dict:
    x, y, w, h = region["bbox_px"]
    polygon = region.get("polygon_px") or _bbox_polygon(region["bbox_px"])
    width_value = w / image_width * width_mm
    height_value = h / image_height * height_mm
    if region["region_type"] == "hole":
        shape = "circle"
    elif abs(width_value - height_value) <= 0.25:
        shape = "circle"
    elif len(polygon) == 4:
        shape = "rect"
    else:
        shape = "oval"
    return {
        "type": region["region_type"],
        "visual_class": region["visual_class"],
        "shape": shape,
        "center": {"x_mm": round(region["x_px"] / image_width * width_mm, 3), "y_mm": round(region["y_px"] / image_height * height_mm, 3)},
        "bbox": {"x_mm": round(x / image_width * width_mm, 3), "y_mm": round(y / image_height * height_mm, 3), "width_mm": round(width_value, 3), "height_mm": round(height_value, 3)},
        "polygon": [{"x_mm": round(point[0] / image_width * width_mm, 3), "y_mm": round(point[1] / image_height * height_mm, 3)} for point in polygon],
        "source": region["method"],
    }


def _visible_text_region(observation: dict, width_mm: float, height_mm: float, image_width: int, image_height: int) -> dict:
    x, y, w, h = observation.get("bbox_px") or [observation["x_px"], observation["y_px"], 1.0, 1.0]
    return {"x_mm": round(x / image_width * width_mm, 3), "y_mm": round(y / image_height * height_mm, 3), "width_mm": round(w / image_width * width_mm, 3), "height_mm": round(h / image_height * height_mm, 3)}


def _draw_dashed_line(img: np.ndarray, pt1: tuple[float, float], pt2: tuple[float, float], color: tuple[int, int, int], thickness: int, dash: int = 6, gap: int = 4) -> None:
    x1, y1 = pt1
    x2, y2 = pt2
    seg_len = float(np.hypot(x2 - x1, y2 - y1))
    if seg_len < 1:
        return
    dx = (x2 - x1) / seg_len
    dy = (y2 - y1) / seg_len
    step = dash + gap
    pos = 0.0
    while pos < seg_len:
        s = min(pos, seg_len)
        e = min(pos + dash, seg_len)
        cv2.line(img, (int(x1 + dx * s), int(y1 + dy * s)), (int(x1 + dx * e), int(y1 + dy * e)), color, thickness)
        pos += step


def _draw_dashed_polygon(img: np.ndarray, points: np.ndarray, color: tuple[int, int, int], thickness: int) -> None:
    pts = points.tolist()
    n = len(pts)
    for i in range(n):
        p1 = (float(pts[i][0]), float(pts[i][1]))
        p2 = (float(pts[(i + 1) % n][0]), float(pts[(i + 1) % n][1]))
        _draw_dashed_line(img, p1, p2, color, thickness)


def _draw_dashed_rect(img: np.ndarray, bbox: tuple[int, int, int, int], color: tuple[int, int, int], thickness: int) -> None:
    x, y, w, h = bbox
    corners = [(x, y), (x + w, y), (x + w, y + h), (x, y + h)]
    for i in range(4):
        _draw_dashed_line(img, corners[i], corners[(i + 1) % 4], color, thickness)


def _generate_pad_annotation(image: np.ndarray, all_pads: list[dict], matched_indices: set[int], pair_indices: set[int]) -> bytes:
    annotated = image.copy()
    for i, pad in enumerate(all_pads):
        bbox = pad.get("bbox_px")
        if bbox is None:
            continue
        bx, by, bw, bh = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
        polygon = pad.get("polygon_px")
        is_matched = i in matched_indices
        is_paired = i in pair_indices
        color_bgr = (0, 0, 255) if is_matched else (0, 255, 255)
        thickness = max(2, min(image.shape[0], image.shape[1]) // 250)
        if polygon:
            pts = np.array([[int(p[0]), int(p[1])] for p in polygon], np.int32)
            if is_matched:
                cv2.polylines(annotated, [pts], True, color_bgr, thickness)
            else:
                _draw_dashed_polygon(annotated, pts, color_bgr, thickness)
        else:
            rect = (bx, by, bw, bh)
            if is_matched:
                cv2.rectangle(annotated, (bx, by), (bx + bw, by + bh), color_bgr, thickness)
            else:
                _draw_dashed_rect(annotated, rect, color_bgr, thickness)
        region_type = pad.get("region_type", "?")
        dim_label = f"{pad.get('width_mm', 0):.1f}x{pad.get('height_mm', 0):.1f}mm"
        if is_matched:
            label = region_type
        else:
            label = f"({region_type} {dim_label})"
        cv2.putText(annotated, label, (bx + 2, max(by - 4, 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 0) if is_matched else (0, 220, 255), 1)
    _, encoded = cv2.imencode(".png", annotated)
    return encoded.tobytes()


def _candidate_id(label: str) -> str:
    return label.replace("+", "_POS").replace("-", "_NEG")


def _candidate_distance(first: dict, second: dict) -> float:
    a, b = first["visible_position"], second["visible_position"]
    return float(np.hypot(a["x_mm"] - b["x_mm"], a["y_mm"] - b["y_mm"]))


def _find_symmetric_pads(all_pads: list[dict]) -> list[tuple[int, int]]:
    pairs: list[tuple[int, int]] = []
    used: set[int] = set()
    n = len(all_pads)
    for i in range(n):
        if i in used:
            continue
        w_i = all_pads[i].get("width_mm", 0)
        h_i = all_pads[i].get("height_mm", 0)
        if w_i <= 0 or h_i <= 0:
            continue
        group = [i]
        for j in range(i + 1, n):
            if j in used:
                continue
            w_j = all_pads[j].get("width_mm", 0)
            h_j = all_pads[j].get("height_mm", 0)
            if w_j <= 0 or h_j <= 0:
                continue
            w_ratio = max(w_i, w_j) / max(min(w_i, w_j), 1e-6)
            h_ratio = max(h_i, h_j) / max(min(h_i, h_j), 1e-6)
            if w_ratio <= 1.35 and h_ratio <= 1.35:
                group.append(j)
        if len(group) == 2:
            pairs.append((group[0], group[1]))
            used.update(group)
        elif len(group) > 2:
            used.update(group)
    return pairs
