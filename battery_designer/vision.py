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


def calibrate_known_size(image_bytes: bytes, width_mm: float, height_mm: float) -> CalibrationResult:
    """Rectify a rectangular PCB using its confirmed size and visible board color."""
    if not 2 <= width_mm <= 500 or not 2 <= height_mm <= 500:
        raise DesignError("INVALID_BOARD_SIZE", "width_mm and height_mm must be between 2 and 500 mm")
    image = cv2.imdecode(np.frombuffer(image_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise DesignError("INVALID_IMAGE", "The uploaded file is not a readable image.")

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    image_area = image.shape[0] * image.shape[1]
    target_aspect = max(width_mm, height_mm) / min(width_mm, height_mm)
    candidates: list[tuple[float, np.ndarray, float, float]] = []
    for saturation in (140, 120, 100, 80):
        mask = cv2.inRange(hsv, (25, saturation, 20), (115, 255, 255))
        kernel_size = max(7, int(round(min(image.shape[:2]) / 140)) | 1)
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 0.005 * image_area:
                continue
            rect = cv2.minAreaRect(contour)
            a, b = rect[1]
            if min(a, b) < 10:
                continue
            aspect = max(a, b) / min(a, b)
            if aspect < 2 or max(a, b) < 0.25 * max(image.shape[:2]):
                continue
            aspect_error = abs(np.log(aspect / target_aspect))
            fill = area / max(a * b, 1)
            score = float(2.5 * aspect_error - min(area / image_area, 0.3) - 0.15 * fill)
            candidates.append((score, contour, aspect, fill))
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
    confidence = max(0.25, min(0.85, 0.55 * aspect_quality + 0.30 * min(fill, 1.0)))
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
        method="known_size_auto",
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
    for epsilon in (0.004, 0.006, 0.008, 0.01, 0.015, 0.02, 0.03, 0.04, 0.05, 0.06):
        polygon = cv2.approxPolyDP(hull, epsilon * perimeter, True)
        if len(polygon) != 4 or not cv2.isContourConvex(polygon):
            continue
        polygon = polygon.reshape(4, 2).astype(np.float32)
        if cv2.contourArea(polygon) < 0.75 * cv2.contourArea(hull):
            continue
        return _order_quad(polygon), "detected_board_corners"
    return _order_quad(cv2.boxPoints(cv2.minAreaRect(contour)).astype(np.float32)), "minimum_area_rectangle_fallback"


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
