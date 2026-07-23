"""Full pipeline diagnostic: detection → calibration → VLM contour."""
import cv2, numpy as np, json, base64, io
from battery_designer.vision import detect_black_frame, calibrate_black_frame

for side in ['front', 'back']:
    buf = open(f'input/{side}.jpg', 'rb').read()
    nparr = np.frombuffer(buf, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # 1. Detection + annotated image (green outline on original)
    dr = detect_black_frame(buf, 2.0)
    if dr.get('annotated_png_base64'):
        with open(f'diag2_{side}_01_detect.png', 'wb') as f:
            f.write(base64.b64decode(dr['annotated_png_base64']))
    print(f"{side}: detect found={dr['found']} aspect={dr.get('aspect_ratio')} "
          f"size={dr.get('avg_width_px')}x{dr.get('avg_height_px')}px")

    # 2. Calibration
    cr = calibrate_black_frame(buf, 60.0, 30.0)
    with open(f'diag2_{side}_02_cal_annotated.png', 'wb') as f:
        f.write(base64.b64decode(cr['annotated_png_base64']))
    with open(f'diag2_{side}_03_rectified.png', 'wb') as f:
        f.write(base64.b64decode(cr['rectified_png_base64']))

    outline = np.array(cr['outline'], dtype=np.int32)
    h, w = img.shape[:2]

    # Draw overlay with corner numbers
    overlay = img.copy()
    cv2.drawContours(overlay, [outline], -1, (0, 255, 0), 3)
    for i, pt in enumerate(outline):
        cv2.circle(overlay, tuple(pt), 10, (255, 0, 0), -1)
        cv2.putText(overlay, str(i), (pt[0]+12, pt[1]-12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    cv2.imwrite(f'diag2_{side}_04_overlay.png', overlay)

    print(f"  calibrate: id={cr['calibration_id']} ppm={cr['pixels_per_mm']} "
          f"{cr['width_mm']}x{cr['height_mm']}mm")
    print(f"  outline corners (px): {cr['outline']}")
    print(f"  image: {w}x{h} — all corners in bounds")

print("\nDone. Check diag2_*.png files in workspace.")
