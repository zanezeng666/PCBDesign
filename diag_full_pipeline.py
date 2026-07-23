"""Full pipeline diagnostic: save every intermediate image for inspection."""
from __future__ import annotations
import sys, json, base64, cv2, numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from battery_designer.vision import detect_black_frame, calibrate_black_frame, extract_pcb

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "diag_full"
OUT.mkdir(exist_ok=True)

for side, frame_w, frame_h in [("front", 60, 30), ("back", 60, 30)]:
    img_path = ROOT / "input" / f"{side}.jpg"
    if not img_path.exists():
        print(f"SKIP: {img_path} not found")
        continue

    print(f"\n{'='*60}")
    print(f"Diagnosing: {side}.jpg ({frame_w}x{frame_h}mm)")
    print(f"{'='*60}")

    img_bytes = img_path.read_bytes()

    # ── Step 0: raw image info ──
    nparr = np.frombuffer(img_bytes, np.uint8)
    raw = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    rh, rw = raw.shape[:2]
    print(f"  Raw image: {rw}x{rh} px")

    # ── Step 1: Detect black frame ──
    framed = detect_black_frame(img_bytes, target_aspect=frame_w/frame_h)
    print(f"  Frame found: {framed.get('found')}")
    if framed.get("found"):
        outline = framed["outline"]
        print(f"  Frame outline: {outline}")
        print(f"  Detected aspect: {framed['aspect_ratio']}")
        print(f"  Expected aspect: {frame_w/frame_h:.4f}")
        print(f"  Avg width px: {framed['avg_width_px']}")
        print(f"  Avg height px: {framed['avg_height_px']}")

        # Save annotated frame detection
        anno = raw.copy()
        pts = np.array(outline, np.int32).reshape(-1, 1, 2)
        cv2.drawContours(anno, [pts], -1, (0, 255, 0), 5)
        for i, pt in enumerate(outline):
            cv2.circle(anno, tuple(pt), 15, (255, 0, 0), -1)
            cv2.putText(anno, f"{i}", (pt[0]+20, pt[1]+20),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
        cv2.imwrite(str(OUT / f"diag_{side}_01_frame_detected.jpg"), anno)
        print(f"  -> Saved diag_{side}_01_frame_detected.jpg")
    else:
        print(f"  ERROR: {framed.get('error')}")
        continue

    # ── Step 2: Calibrate (perspective rectification) ──
    try:
        cal = calibrate_black_frame(img_bytes, frame_w, frame_h)
    except Exception as e:
        print(f"  ERROR in calibrate: {e}")
        continue

    cal_id = cal["calibration_id"]
    print(f"  Calibration ID: {cal_id}")
    print(f"  pixels_per_mm: {cal['pixels_per_mm']}")

    # Save rectified image
    rect_b64 = cal["rectified_png_base64"]
    rect_buf = base64.b64decode(rect_b64)
    (OUT / f"diag_{side}_02_rectified.png").write_bytes(rect_buf)
    print(f"  -> Saved diag_{side}_02_rectified.png")

    # Save annotated original
    anno_b64 = cal["annotated_png_base64"]
    anno_buf = base64.b64decode(anno_b64)
    (OUT / f"diag_{side}_03_annotated.png").write_bytes(anno_buf)
    print(f"  -> Saved diag_{side}_03_annotated.png")

    # ── Step 3: Load rectified and check ──
    r_np = np.frombuffer(rect_buf, np.uint8)
    rect_img = cv2.imdecode(r_np, cv2.IMREAD_COLOR)
    rect_h, rect_w = rect_img.shape[:2]
    print(f"  Rectified: {rect_w}x{rect_h} px")

    # Check board visibility: look for green/blue board vs white paper
    hsv = cv2.cvtColor(rect_img, cv2.COLOR_BGR2HSV)
    # White paper: low saturation, high value
    white_mask = cv2.inRange(hsv, (0, 0, 180), (180, 40, 255))
    # Dark board: low value
    dark_mask = cv2.inRange(hsv, (0, 0, 0), (180, 255, 80))
    white_pct = np.sum(white_mask > 0) / (rect_w * rect_h) * 100
    dark_pct = np.sum(dark_mask > 0) / (rect_w * rect_h) * 100
    print(f"  White paper region: {white_pct:.1f}%")
    print(f"  Dark/PCB region: {dark_pct:.1f}%")
    print(f"  Total (non-marginal): {100 - white_pct - dark_pct:.1f}%")

    if white_pct < 5 and dark_pct > 60:
        print(f"  WARNING: Rectified image is mostly dark - PCB may be CROPPED!")

    # Check if board is near edges (possible cropping indicator)
    gray = cv2.cvtColor(rect_img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 200)
    # Check edge density along borders
    top_edge = np.mean(edges[0:5, :]) / 255 * 100
    bot_edge = np.mean(edges[-5:, :]) / 255 * 100
    left_edge = np.mean(edges[:, 0:5]) / 255 * 100
    right_edge = np.mean(edges[:, -5:]) / 255 * 100
    print(f"  Edge density: top={top_edge:.1f}% bot={bot_edge:.1f}% left={left_edge:.1f}% right={right_edge:.1f}%")

    # ── Step 4: Extract PCB ──
    try:
        pcb = extract_pcb(rect_buf, frame_w, frame_h, cal["pixels_per_mm"])
        vc = pcb["vertex_count"]
        print(f"  PCB outline vertices: {vc}")
        print(f"  Grooves: {pcb['groove_count']}")

        # Save debug steps
        for ds in pcb.get("debug_steps", []):
            step_name = ds["step"]
            label = ds["label"]
            if "image_base64" in ds:
                buf = base64.b64decode(ds["image_base64"])
                (OUT / f"diag_{side}_{step_name}_{label}.png").write_bytes(buf)
                print(f"  -> Saved diag_{side}_{step_name}_{label}.png")

        # Save pcb outline data
        pcb_data = {
            "outline": pcb["outline"],
            "grooves": pcb["grooves"],
            "vertex_count": vc,
            "groove_count": pcb["groove_count"],
        }
        (OUT / f"diag_{side}_pcb_data.json").write_text(json.dumps(pcb_data, indent=2))
        print(f"  -> Saved diag_{side}_pcb_data.json")

        # Draw PCB outline overlay on rectified
        overlay = rect_img.copy()
        if pcb["outline"]:
            h_r, w_r = rect_img.shape[:2]
            pts = np.array([[
                int(p["x_mm"] / frame_w * w_r),
                int(p["y_mm"] / frame_h * h_r)
            ] for p in pcb["outline"]], np.int32)
            cv2.polylines(overlay, [pts], True, (0, 255, 0), 3)
            for i, pt in enumerate(pts):
                cv2.circle(overlay, tuple(pt), 8, (255, 0, 0), -1)
                cv2.putText(overlay, str(i), (pt[0]+10, pt[1]+10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        cv2.imwrite(str(OUT / f"diag_{side}_04_pcb_outline_overlay.png"), overlay)
        print(f"  -> Saved diag_{side}_04_pcb_outline_overlay.png")

    except Exception as e:
        print(f"  ERROR in extract_pcb: {e}")
        import traceback
        traceback.print_exc()

print(f"\n{'='*60}")
print(f"All diagnostic images saved to: {OUT}")
print(f"{'='*60}")
