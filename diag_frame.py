"""Diagnostic: save every intermediate step of detect_black_frame."""
import cv2, numpy as np
from battery_designer.vision import detect_black_frame

for side in ("front", "back"):
    name = f"input/{side}.jpg"
    buf = open(name, "rb").read()
    nparr = np.frombuffer(buf, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    h, w = img.shape[:2]
    print(f"\n=== {name}: {w}x{h} ===")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.imwrite(f"diag_{side}_gray.png", gray)

    _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    cv2.imwrite(f"diag_{side}_otsu.png", otsu)

    adaptive = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                      cv2.THRESH_BINARY_INV, 101, 20)
    cv2.imwrite(f"diag_{side}_adaptive.png", adaptive)

    combined = cv2.bitwise_or(otsu, adaptive)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel, 2)
    cv2.imwrite(f"diag_{side}_combined.png", combined)

    contours, hierarchy = cv2.findContours(combined, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    print(f"  Total contours: {len(contours)}")

    # Draw all valid contours
    viz = img.copy()
    for idx, cnt in enumerate(contours):
        area = cv2.contourArea(cnt)
        if area < w * h * 0.005:
            continue
        rect = cv2.minAreaRect(cnt)
        rw, rh = rect[1]
        if min(rw, rh) < 30:
            continue
        box = cv2.boxPoints(rect)
        box_area_pts = cv2.contourArea(box.reshape(-1, 1, 2).astype(np.int32))
        rect_score = min(box_area_pts / (rw * rh), (rw * rh) / max(box_area_pts, 1))
        if rect_score < 0.6:
            continue
        aspect = max(rw, rh) / max(min(rw, rh), 1)
        if aspect < 0.5 or aspect > 3.0:
            continue

        has_child = (hierarchy is not None and len(hierarchy[0]) > idx and hierarchy[0][idx][2] >= 0)
        color = (0, 255, 0) if has_child else (0, 0, 255)
        thickness = 3 if has_child else 1
        cv2.drawContours(viz, [box.astype(np.int32)], -1, color, thickness)
        print(f"  contour[{idx}]: area={area:.0f} ({area/(w*h)*100:.1f}%), "
              f"rect={rw:.0f}x{rh:.0f} (@{rect[2]:.1f}°), "
              f"aspect={aspect:.3f}, rect_score={rect_score:.3f}, "
              f"has_child={has_child}")

    cv2.imwrite(f"diag_{side}_all_frames.png", viz)

    # Now call the actual function
    for label, tasp in [("no_aspect", None), ("aspect_2.0", 2.0)]:
        r = detect_black_frame(buf, tasp)
        print(f"  detect_black_frame({label}): found={r['found']}, aspect={r.get('aspect_ratio')}, "
              f"size={r.get('avg_width_px')}x{r.get('avg_height_px')}px")

print("\nDone. Check diag_*.png files.")
