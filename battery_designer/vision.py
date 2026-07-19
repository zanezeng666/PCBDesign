from __future__ import annotations

import base64
from dataclasses import dataclass

import cv2
import numpy as np

from .errors import DesignError


@dataclass
class CalibrationResult:
    width_mm: float
    height_mm: float
    pixels_per_mm: float
    outline: list[dict[str, float]]
    confidence: float
    rectified_png: bytes
    preview_png: bytes
    marker_ids: list[int]
    method: str = "aruco"
    source_quad_px: list[list[float]] | None = None
    perspective_method: str | None = None

    def response(self) -> dict:
        return {
            "width_mm": self.width_mm,
            "height_mm": self.height_mm,
            "pixels_per_mm": self.pixels_per_mm,
            "outline": self.outline,
            "confidence": self.confidence,
            "marker_ids": self.marker_ids,
            "method": self.method,
            "source_quad_px": self.source_quad_px,
            "perspective_method": self.perspective_method,
            "rectified_png_base64": base64.b64encode(self.rectified_png).decode("ascii"),
            "preview_png_base64": base64.b64encode(self.preview_png).decode("ascii"),
        }


def calibrate_photo(image_bytes: bytes, marker_size_mm: float) -> CalibrationResult:
    if not 2 <= marker_size_mm <= 100:
        raise DesignError("INVALID_MARKER_SIZE", "marker_size_mm must be between 2 and 100 mm")
    image = cv2.imdecode(np.frombuffer(image_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise DesignError("INVALID_IMAGE", "The uploaded file is not a readable image.")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    if hasattr(cv2.aruco, "ArucoDetector"):
        detector = cv2.aruco.ArucoDetector(dictionary, cv2.aruco.DetectorParameters())
        corners, ids, _ = detector.detectMarkers(gray)
    else:  # OpenCV 4 compatibility.
        corners, ids, _ = cv2.aruco.detectMarkers(gray, dictionary)
    if ids is None or len(ids) != 4:
        raise DesignError("ARUCO_COUNT", "Exactly four ArUco markers are required.", {"detected": 0 if ids is None else len(ids)})

    centers = np.array([corner[0].mean(axis=0) for corner in corners], dtype=np.float32)
    ordered_centers = _order_quad(centers)
    marker_sides = []
    for corner in corners:
        points = corner[0]
        marker_sides.extend(np.linalg.norm(points[(i + 1) % 4] - points[i]) for i in range(4))
    pixels_per_mm = float(np.median(marker_sides) / marker_size_mm)
    if pixels_per_mm < 1:
        raise DesignError("CALIBRATION_TOO_SMALL", "Markers are too small in the image for reliable calibration.")

    top = np.linalg.norm(ordered_centers[1] - ordered_centers[0])
    bottom = np.linalg.norm(ordered_centers[2] - ordered_centers[3])
    left = np.linalg.norm(ordered_centers[3] - ordered_centers[0])
    right = np.linalg.norm(ordered_centers[2] - ordered_centers[1])
    width_px = max(200, int(round((top + bottom) / 2)))
    height_px = max(200, int(round((left + right) / 2)))
    target = np.array([[0, 0], [width_px - 1, 0], [width_px - 1, height_px - 1], [0, height_px - 1]], dtype=np.float32)
    transform = cv2.getPerspectiveTransform(ordered_centers, target)
    rectified = cv2.warpPerspective(image, transform, (width_px, height_px))

    outline_px, contour_confidence = _extract_outline(rectified)
    outline = [{"x_mm": round(float(x / pixels_per_mm), 3), "y_mm": round(float(y / pixels_per_mm), 3)} for x, y in outline_px]

    preview = rectified.copy()
    cv2.polylines(preview, [outline_px.astype(np.int32)], True, (0, 0, 255), max(2, width_px // 400))
    ok_rectified, encoded_rectified = cv2.imencode(".png", rectified)
    ok_preview, encoded_preview = cv2.imencode(".png", preview)
    if not ok_rectified or not ok_preview:
        raise DesignError("IMAGE_ENCODING_FAILED", "Failed to encode calibration previews.")
    side_variation = float(np.std(marker_sides) / max(np.mean(marker_sides), 1))
    confidence = max(0.0, min(1.0, contour_confidence * (1.0 - min(side_variation, 0.5))))
    return CalibrationResult(
        width_mm=round(width_px / pixels_per_mm, 3),
        height_mm=round(height_px / pixels_per_mm, 3),
        pixels_per_mm=pixels_per_mm,
        outline=outline,
        confidence=confidence,
        rectified_png=encoded_rectified.tobytes(),
        preview_png=encoded_preview.tobytes(),
        marker_ids=sorted(int(value) for value in ids.ravel()),
    )


def _detect_board_by_color(hsv: np.ndarray, image_shape: tuple, image_area: float, target_aspect: float) -> list[tuple[float, np.ndarray, float, float]]:
    """Detect PCB board contour using multi-color HSV masks (green, blue, black) with progressive saturation sweep."""
    candidates: list[tuple[float, np.ndarray, float, float]] = []
    h, w = image_shape[:2]

    # Multi-color PCB detection: green/blue/black all supported
    color_ranges = [
        # Name, lower_bound, upper_bound
        ("green",  (25, 40, 20),   (115, 255, 255)),
        ("green",  (30, 30, 15),   (105, 255, 255)),
        ("blue",   (90, 40, 25),   (150, 255, 255)),
        ("blue",   (95, 30, 20),   (145, 255, 255)),
        ("black",  (0, 0, 5),      (180, 60, 70)),
        ("black",  (0, 0, 3),      (180, 80, 80)),
    ]

    kernel_size = max(7, int(round(min(h, w) / 140)) | 1)
    kernel = np.ones((kernel_size, kernel_size), np.uint8)

    for color_name, lower, upper in color_ranges:
        # Progressive saturation sweep: start strict (high sat threshold), then relax (lower sat)
        # This ensures the algorithm captures the board shape across varied lighting
        if color_name in ("green", "blue"):
            # Sweep lower saturation from high (strict) to low (relaxed), like original 140→80
            sweep_values = (140, 120, 100, 80)
        else:
            # Black boards: sweep upper S/V bound from low (strict) to higher (relaxed)
            sweep_values = (0.55, 0.65, 0.75, 0.85, 0.95, 1.0)
        for sweep_val in sweep_values:
            if color_name in ("green", "blue"):
                lower_adj = (lower[0], sweep_val, lower[2])
                upper_adj = upper
            else:
                # For black, scale the upper S/V bounds proportionally
                lower_adj = lower
                upper_adj = (upper[0], int(upper[1] * sweep_val), int(upper[2] * sweep_val))
            mask = cv2.inRange(hsv, lower_adj, upper_adj)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                area = cv2.contourArea(contour)
                if area < 0.002 * image_area:
                    continue
                rect = cv2.minAreaRect(contour)
                a, b = rect[1]
                if min(a, b) < 10:
                    continue
                aspect = max(a, b) / min(a, b)
                if aspect < 2 or max(a, b) < 0.25 * max(h, w):
                    continue
                aspect_error = abs(np.log(aspect / target_aspect))
                fill = area / max(a * b, 1)
                # Prefer green over blue/black slightly (greens are most reliable)
                color_penalty = 0.15 if color_name == "blue" else (0.30 if color_name == "black" else 0.0)
                score = float(2.5 * aspect_error - min(area / image_area, 0.3) - 0.15 * fill + color_penalty)
                candidates.append((score, contour, aspect, fill))

    return candidates


def _detect_board_by_otsu(v_channel: np.ndarray, image_shape: tuple, image_area: float, target_aspect: float) -> list[tuple[float, np.ndarray, float, float]]:
    """Fallback: Otsu brightness separation on V channel to distinguish board from background.

    When color-based detection fails (non-green/blue boards, unusual lighting), Otsu
    adaptively thresholds the V channel to separate the PCB from its background
    (desk, paper, etc.). Both dark-on-light and light-on-dark orientations are tried.
    """
    candidates: list[tuple[float, np.ndarray, float, float]] = []
    h, w = image_shape[:2]

    otsu_thresh, _ = cv2.threshold(v_channel, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel_size = max(7, int(round(min(h, w) / 120)) | 1)
    kernel = np.ones((kernel_size, kernel_size), np.uint8)

    # Try both polarities: board could be darker or brighter than background
    for invert, label in ((False, "otsu_bright"), (True, "otsu_dark")):
        if invert:
            binary = (v_channel < otsu_thresh).astype(np.uint8) * 255
        else:
            binary = (v_channel > otsu_thresh).astype(np.uint8) * 255

        # Remove small noise, close gaps
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8), iterations=1)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=3)

        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 0.015 * image_area or area > 0.92 * image_area:
                continue
            hull = cv2.convexHull(contour)
            hull_area = cv2.contourArea(hull)
            if hull_area < 0.02 * image_area:
                continue
            rect = cv2.minAreaRect(hull)
            a, b = rect[1]
            if min(a, b) < 15:
                continue
            aspect = max(a, b) / min(a, b)
            if aspect < 1.3 or aspect > 12:
                continue
            if max(a, b) < 0.22 * max(h, w):
                continue
            aspect_error = abs(np.log(aspect / target_aspect))
            fill = hull_area / max(a * b, 1)
            # Otsu is less reliable than color detection → higher base penalty
            score = float(2.5 * aspect_error - min(hull_area / image_area, 0.25) - 0.15 * fill + 0.45)
            candidates.append((score, hull, aspect, fill))

    return candidates


def _detect_board_by_edges(gray: np.ndarray, image_shape: tuple, image_area: float, target_aspect: float) -> list[tuple[float, np.ndarray, float, float]]:
    """Final fallback: Canny edge detection when both color and Otsu approaches fail."""
    candidates: list[tuple[float, np.ndarray, float, float]] = []
    h, w = image_shape[:2]

    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    for low_thresh in (30, 40, 50, 60):
        edges = cv2.Canny(blurred, low_thresh, low_thresh * 3)
        kernel = np.ones((7, 7), np.uint8)
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=3)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 0.008 * image_area:
                continue
            hull = cv2.convexHull(contour)
            hull_area = cv2.contourArea(hull)
            if hull_area < 0.01 * image_area:
                continue
            rect = cv2.minAreaRect(hull)
            a, b = rect[1]
            if min(a, b) < 15:
                continue
            aspect = max(a, b) / min(a, b)
            if aspect < 1.5 or aspect > 10:
                continue
            if max(a, b) < 0.2 * max(h, w):
                continue
            aspect_error = abs(np.log(aspect / target_aspect))
            fill = hull_area / max(a * b, 1)
            # Edge detection is least reliable → highest base penalty
            score = float(3.0 * aspect_error - min(hull_area / image_area, 0.3) - 0.1 * fill + 0.55)
            candidates.append((score, hull, aspect, fill))
    return candidates


def calibrate_known_size(image_bytes: bytes, width_mm: float, height_mm: float) -> CalibrationResult:
    """Rectify a rectangular PCB using its confirmed size and visible board color.

    Detection pipeline (3 stages, progressive fallback):
    1. Multi-color HSV masks (green, blue, black) — most reliable
    2. Otsu V-channel brightness separation — handles unusual lighting
    3. Canny edge detection — last resort
    """
    if not 2 <= width_mm <= 500 or not 2 <= height_mm <= 500:
        raise DesignError("INVALID_BOARD_SIZE", "width_mm and height_mm must be between 2 and 500 mm")
    image = cv2.imdecode(np.frombuffer(image_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise DesignError("INVALID_IMAGE", "The uploaded file is not a readable image.")

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    v_channel = hsv[:, :, 2]
    image_area = image.shape[0] * image.shape[1]
    target_aspect = max(width_mm, height_mm) / min(width_mm, height_mm)

    # Stage 1: multi-color detection (green, blue, black)
    candidates = _detect_board_by_color(hsv, image.shape, image_area, target_aspect)
    detection_method = "color"

    # Stage 2: Otsu brightness fallback
    if not candidates:
        candidates = _detect_board_by_otsu(v_channel, image.shape, image_area, target_aspect)
        detection_method = "otsu"

    # Stage 3: edge-based fallback (last resort)
    if not candidates:
        candidates = _detect_board_by_edges(gray, image.shape, image_area, target_aspect)
        detection_method = "edges"

    if not candidates:
        raise DesignError(
            "RECTANGULAR_BOARD_NOT_FOUND",
            "No reliable rectangular PCB was found. Use an ArUco capture or provide a clearer top-down photo.",
        )

    _, contour, source_aspect, fill = min(candidates, key=lambda item: item[0])
    source, perspective_method = _contour_quad(contour)
    pixels_per_mm = 50.0
    output_width = max(200, int(round(width_mm * pixels_per_mm)))
    output_height = max(100, int(round(height_mm * pixels_per_mm)))
    target = np.array([[0, 0], [output_width - 1, 0], [output_width - 1, output_height - 1], [0, output_height - 1]], dtype=np.float32)
    transform = cv2.getPerspectiveTransform(source, target)
    rectified = cv2.warpPerspective(image, transform, (output_width, output_height))

    outline_px = np.array([[0, 0], [output_width - 1, 0], [output_width - 1, output_height - 1], [0, output_height - 1]], dtype=np.int32)
    preview = rectified.copy()
    cv2.polylines(preview, [outline_px], True, (0, 0, 255), max(2, output_width // 500))
    ok_rectified, encoded_rectified = cv2.imencode(".png", rectified)
    ok_preview, encoded_preview = cv2.imencode(".png", preview)
    if not ok_rectified or not ok_preview:
        raise DesignError("IMAGE_ENCODING_FAILED", "Failed to encode calibration previews.")
    aspect_quality = max(0.0, 1.0 - min(abs(float(np.log(source_aspect / target_aspect))), 0.7))
    # Confidence reflects both shape fit and detection method reliability
    base_confidence = max(0.20, min(0.88, 0.55 * aspect_quality + 0.30 * min(fill, 1.0)))
    method_penalty = {"color": 0.0, "otsu": 0.15, "edges": 0.25}.get(detection_method, 0.25)
    confidence = max(0.18, base_confidence - method_penalty)
    method_str = f"known_size_{detection_method}"
    return CalibrationResult(
        width_mm=round(width_mm, 3),
        height_mm=round(height_mm, 3),
        pixels_per_mm=pixels_per_mm,
        outline=[
            {"x_mm": 0.0, "y_mm": 0.0},
            {"x_mm": round(width_mm, 3), "y_mm": 0.0},
            {"x_mm": round(width_mm, 3), "y_mm": round(height_mm, 3)},
            {"x_mm": 0.0, "y_mm": round(height_mm, 3)},
        ],
        confidence=confidence,
        rectified_png=encoded_rectified.tobytes(),
        preview_png=encoded_preview.tobytes(),
        marker_ids=[],
        method=method_str,
        source_quad_px=[[round(float(x), 2), round(float(y), 2)] for x, y in source],
        perspective_method=perspective_method,
    )


def transform_back_point(point: dict[str, float], source_width_mm: float, source_height_mm: float, target_width_mm: float, target_height_mm: float, transform: str) -> dict[str, float]:
    x = point["x_mm"] * target_width_mm / source_width_mm
    y = point["y_mm"] * target_height_mm / source_height_mm
    if transform in {"mirror_x", "rotate_180"}:
        x = target_width_mm - x
    if transform in {"mirror_y", "rotate_180"}:
        y = target_height_mm - y
    return {"x_mm": x, "y_mm": y}


def outline_alignment_error(front: dict, back: dict, transform: str) -> float:
    front_points = front["outline"]
    back_points = [
        transform_back_point(point, back["width_mm"], back["height_mm"], front["width_mm"], front["height_mm"], transform)
        for point in back["outline"]
    ]

    def directed(a: list[dict[str, float]], b: list[dict[str, float]]) -> float:
        distances = []
        for point in a:
            distances.append(min(((point["x_mm"] - other["x_mm"]) ** 2 + (point["y_mm"] - other["y_mm"]) ** 2) ** 0.5 for other in b))
        return sum(distances) / len(distances)

    return (directed(front_points, back_points) + directed(back_points, front_points)) / 2


def _order_quad(points: np.ndarray) -> np.ndarray:
    ordered = np.zeros((4, 2), dtype=np.float32)
    sums = points.sum(axis=1)
    differences = np.diff(points, axis=1).ravel()
    ordered[0] = points[np.argmin(sums)]
    ordered[2] = points[np.argmax(sums)]
    ordered[1] = points[np.argmin(differences)]
    ordered[3] = points[np.argmax(differences)]
    return ordered


def _contour_quad(contour: np.ndarray) -> tuple[np.ndarray, str]:
    """Prefer the photographed PCB's four visible corners over a rotated box.

    A min-area rectangle only fixes rotation and uniform scale.  A four-corner
    homography also removes the trapezoid distortion introduced by an oblique
    phone camera.  Rounded or partly hidden boards safely fall back to the
    min-area rectangle and are still flagged for manual confirmation by the UI.
    """
    hull = cv2.convexHull(contour)
    perimeter = cv2.arcLength(hull, True)
    # Stage 1: Try DP approximation at various epsilon levels
    for epsilon in (0.004, 0.006, 0.008, 0.01, 0.015, 0.02, 0.03, 0.04, 0.05, 0.06):
        polygon = cv2.approxPolyDP(hull, epsilon * perimeter, True)
        if len(polygon) != 4 or not cv2.isContourConvex(polygon):
            continue
        polygon = polygon.reshape(4, 2).astype(np.float32)
        if cv2.contourArea(polygon) < 0.75 * cv2.contourArea(hull):
            continue
        return _order_quad(polygon), "detected_board_corners"

    # Stage 2: Extract best 4 extreme points from convex hull
    hull_pts = hull.reshape(-1, 2).astype(np.float32)
    hull_area = cv2.contourArea(hull)
    # Find extreme points: top-left, top-right, bottom-right, bottom-left
    sums = hull_pts.sum(axis=1)
    diffs = np.diff(hull_pts, axis=1).ravel()
    tl = hull_pts[np.argmin(sums)]
    br = hull_pts[np.argmax(sums)]
    tr = hull_pts[np.argmin(diffs)]
    bl = hull_pts[np.argmax(diffs)]
    quad = np.array([tl, tr, br, bl], dtype=np.float32)

    # Validate: area should be close to hull area
    quad_area = cv2.contourArea(quad)
    if quad_area > 0.65 * hull_area:
        return _order_quad(quad), "extreme_points_quad"

    # Stage 3: Use minAreaRect on hull, refine corners from hull vertices for best fit
    rect_pts = cv2.boxPoints(cv2.minAreaRect(hull)).astype(np.float32)
    # Snap each rect corner to the nearest point on the convex hull
    snap_threshold = max(8.0, perimeter * 0.04)
    for i in range(4):
        pt = rect_pts[i].reshape(1, 2)
        # Find closest hull point
        distances = np.linalg.norm(hull_pts - pt, axis=1)
        closest_idx = np.argmin(distances)
        if distances[closest_idx] < snap_threshold:
            rect_pts[i] = hull_pts[closest_idx]
    return _order_quad(rect_pts), "rect_with_hull_snap"


def _extract_outline(image: np.ndarray) -> tuple[np.ndarray, float]:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 40, 120)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8), iterations=2)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    image_area = image.shape[0] * image.shape[1]
    center = (image.shape[1] / 2, image.shape[0] / 2)
    candidates = [c for c in contours if 0.03 * image_area < cv2.contourArea(c) < 0.95 * image_area]
    if not candidates:
        raise DesignError("OUTLINE_NOT_FOUND", "No reliable PCB outline was found; adjust lighting or edit the outline manually.")
    candidates.sort(key=lambda c: (cv2.pointPolygonTest(c, center, False) >= 0, cv2.contourArea(c)), reverse=True)
    contour = candidates[0]
    perimeter = cv2.arcLength(contour, True)
    simplified = cv2.approxPolyDP(contour, max(1.0, 0.003 * perimeter), True).reshape(-1, 2)
    confidence = min(1.0, cv2.contourArea(contour) / (0.25 * image_area))
    return simplified, confidence
