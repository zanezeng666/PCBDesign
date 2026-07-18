from battery_designer.terminal_detection import _normalize_label, _region_distance_px, _visible_region


def test_exact_terminal_silkscreen_labels_are_supported():
    for label in ("B+", "B-", "P+", "P-", "C+", "C-", "NTC", "N", "TH", "ID"):
        normalized, confidence = _normalize_label(label, targeted=False)
        assert normalized == label
        assert confidence > 0


def test_targeted_ocr_confusions_remain_low_confidence_candidates():
    assert _normalize_label("1D", targeted=True) == ("ID", 0.8)
    label, confidence = _normalize_label("PEC", targeted=True)
    assert label == "P-"
    assert confidence < 0.5


def test_text_to_pad_distance_uses_region_edges_not_centers():
    observation = {"x_px": 20, "y_px": 20, "bbox_px": [10, 10, 20, 10]}
    touching_pad = {"bbox_px": [30, 12, 40, 30]}
    assert _region_distance_px(observation, touching_pad) == 0


def test_visible_pad_region_keeps_bbox_and_polygon():
    region = {
        "x_px": 50,
        "y_px": 25,
        "bbox_px": [40, 15, 20, 20],
        "polygon_px": [[40, 15], [60, 15], [60, 35], [40, 35]],
        "region_type": "solder_pad",
        "visual_class": "metallic",
        "method": "bright-component",
    }
    visible = _visible_region(region, 10, 5, 100, 50)
    assert visible["center"] == {"x_mm": 5.0, "y_mm": 2.5}
    assert visible["bbox"]["width_mm"] == 2.0
    assert len(visible["polygon"]) == 4
