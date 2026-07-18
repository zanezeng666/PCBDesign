from __future__ import annotations

import cv2
import numpy as np

from battery_designer.vision import calibrate_known_size, calibrate_photo, outline_alignment_error, transform_back_point


def test_four_marker_photo_is_rectified_and_outlined():
    image = np.full((600, 800, 3), 255, np.uint8)
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    for marker_id, (x, y) in enumerate([(30, 30), (690, 30), (690, 490), (30, 490)]):
        marker = cv2.aruco.generateImageMarker(dictionary, marker_id, 80)
        image[y : y + 80, x : x + 80] = cv2.cvtColor(marker, cv2.COLOR_GRAY2BGR)
    cv2.rectangle(image, (180, 150), (620, 450), (30, 90, 30), -1)
    cv2.rectangle(image, (180, 150), (620, 450), (0, 0, 0), 5)
    ok, encoded = cv2.imencode(".png", image)
    assert ok
    result = calibrate_photo(encoded.tobytes(), 10)
    assert result.width_mm > 60
    assert result.height_mm > 40
    assert len(result.outline) >= 4
    assert result.confidence > 0.5
    assert result.marker_ids == [0, 1, 2, 3]


def test_back_point_is_mirrored_into_front_coordinates():
    assert transform_back_point({"x_mm": 10, "y_mm": 4}, 40, 20, 40, 20, "mirror_x") == {"x_mm": 30, "y_mm": 4}


def test_mirrored_outlines_align():
    front = {"width_mm": 40, "height_mm": 20, "outline": [{"x_mm": 2, "y_mm": 2}, {"x_mm": 38, "y_mm": 2}, {"x_mm": 38, "y_mm": 18}, {"x_mm": 2, "y_mm": 18}]}
    back = {"width_mm": 40, "height_mm": 20, "outline": [{"x_mm": 38, "y_mm": 2}, {"x_mm": 2, "y_mm": 2}, {"x_mm": 2, "y_mm": 18}, {"x_mm": 38, "y_mm": 18}]}
    assert outline_alignment_error(front, back, "mirror_x") == 0


def test_known_size_photo_is_rectified_without_markers():
    image = np.full((600, 1000, 3), 45, np.uint8)
    cv2.rectangle(image, (100, 220), (900, 380), (40, 150, 40), -1)
    ok, encoded = cv2.imencode(".png", image)
    assert ok
    result = calibrate_known_size(encoded.tobytes(), 41, 7)
    assert result.width_mm == 41
    assert result.height_mm == 7
    assert result.method == "known_size_auto"
    assert result.marker_ids == []


def test_known_size_uses_four_corner_perspective_for_trapezoid():
    image = np.full((500, 900, 3), 35, np.uint8)
    board = np.array([[90, 180], [820, 130], [790, 340], [120, 380]], np.int32)
    cv2.fillConvexPoly(image, board, (40, 150, 40))
    ok, encoded = cv2.imencode(".png", image)
    assert ok
    result = calibrate_known_size(encoded.tobytes(), 41, 7)
    assert result.perspective_method == "detected_board_corners"
    assert len(result.source_quad_px or []) == 4
