from __future__ import annotations

import math

import cv2
import numpy as np


def _estimate_corner_radius(poly_pts, x1, y1, x2, y2):
    """Estimate corner radius from polygon with symmetry constraint.

    PCB pads are symmetric rounded rectangles — all 4 corners should have
    the same radius. For each bbox corner, the closest polygon point on the
    arc is at distance r*(sqrt(2)-1) from the corner.

    Symmetry repair: use median of per-corner radii to reject outliers
    caused by physical damage (e.g. chipped copper on one corner).
    """
    import math
    w = x2 - x1
    h = y2 - y1
    short = min(w, h)
    if short < 2 or len(poly_pts) < 4:
        return max(1, int(short * 0.1))

    corners = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
    radii = []
    search_r = short * 0.45  # only consider points near the corner

    for cx, cy in corners:
        min_d = float('inf')
        for px, py in poly_pts:
            if abs(px - cx) > search_r or abs(py - cy) > search_r:
                continue
            d = math.hypot(px - cx, py - cy)
            if d < min_d:
                min_d = d
        if min_d < float('inf') and min_d > 0:
            r = min_d / (math.sqrt(2) - 1)
            radii.append(r)

    if not radii:
        return max(1, int(short * 0.1))

    # Symmetry constraint: PCB pads have identical radii at all 4 corners.
    # Use median to reject outliers from damaged corners (e.g. chipped copper).
    radii_sorted = sorted(radii)
    n = len(radii_sorted)
    if n >= 3:
        median_r = radii_sorted[n // 2] if n % 2 == 1 else (
            radii_sorted[n // 2 - 1] + radii_sorted[n // 2]) / 2
        # Filter out corners deviating >40% from median (likely damaged)
        good = [r for r in radii if 0.6 * median_r <= r <= 1.4 * median_r]
        if good:
            r = sum(good) / len(good)
        else:
            r = median_r
    else:
        r = sum(radii) / len(radii)

    return max(1, min(int(round(r)), short // 2))


def _draw_rounded_rect(img, x1, y1, x2, y2, radius, color, thickness):
    """Draw a rounded rectangle with the given corner radius."""
    w = x2 - x1
    h = y2 - y1
    r = max(1, min(radius, min(w, h) // 2))
    try:
        cv2.roundRect(img, (x1, y1), (w, h), (r, r), color, thickness)
    except AttributeError:
        # Fallback for OpenCV < 4.6
        if thickness < 0:
            cv2.rectangle(img, (x1 + r, y1), (x2 - r, y2), color, -1)
            cv2.rectangle(img, (x1, y1 + r), (x2, y2 - r), color, -1)
            cv2.circle(img, (x1 + r, y1 + r), r, color, -1)
            cv2.circle(img, (x2 - r, y1 + r), r, color, -1)
            cv2.circle(img, (x1 + r, y2 - r), r, color, -1)
            cv2.circle(img, (x2 - r, y2 - r), r, color, -1)
        else:
            cv2.rectangle(img, (x1 + r, y1), (x2 - r, y2), color, thickness)
            cv2.rectangle(img, (x1, y1 + r), (x2, y2 - r), color, thickness)
            cv2.ellipse(img, (x1 + r, y1 + r), (r, r), 0, 180, 270, color, thickness)
            cv2.ellipse(img, (x2 - r, y1 + r), (r, r), 0, 270, 360, color, thickness)
            cv2.ellipse(img, (x2 - r, y2 - r), (r, r), 0, 0, 90, color, thickness)
            cv2.ellipse(img, (x1 + r, y2 - r), (r, r), 0, 90, 180, color, thickness)
