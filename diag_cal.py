import cv2, numpy as np, json, base64
from battery_designer.vision import calibrate_black_frame, detect_black_frame

for side in ['front', 'back']:
    buf = open(f'input/{side}.jpg', 'rb').read()
    nparr = np.frombuffer(buf, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    h, w = img.shape[:2]
    print(f"\n=== {side}.jpg: {w}x{h}px ===")

    # Step 1: detect with target_aspect=2.0
    r = detect_black_frame(buf, 2.0)
    print(f"  detect: found={r['found']}, aspect={r.get('aspect_ratio')}, "
          f"size={r.get('avg_width_px')}x{r.get('avg_height_px')}px")
    if r.get('error'):
        print(f"  detect error: {r['error']}")
    if r.get('annotated_png_base64'):
        ann = base64.b64decode(r['annotated_png_base64'])
        with open(f'diag_{side}_detect_annotated.png', 'wb') as f:
            f.write(ann)
        print(f"  saved annotated image")

    # Step 2: calibrate
    try:
        r2 = calibrate_black_frame(buf, 60.0, 30.0)
        print(f"  calibrate: id={r2['calibration_id']}, ppm={r2['pixels_per_mm']}, "
              f"w={r2['width_mm']}mm, h={r2['height_mm']}mm")
        print(f"  outline corners: {r2['outline']}")

        # Save rectified
        rect = base64.b64decode(r2['rectified_png_base64'])
        with open(f'diag_{side}_rectified.png', 'wb') as f:
            f.write(rect)
        ann2 = base64.b64decode(r2['annotated_png_base64'])
        with open(f'diag_{side}_cal_annotated.png', 'wb') as f:
            f.write(ann2)
    except Exception as e:
        print(f"  calibrate ERROR: {e}")
        import traceback
        traceback.print_exc()
