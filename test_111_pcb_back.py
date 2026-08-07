"""测试 111 PCB back 目录的图片识别 + 正反面交叉校验"""
import sys
import json
import base64
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from battery_designer.pcb_recognition.pipeline import PCBRecognitionPipeline
from battery_designer.pcb_recognition.cross_validator import CrossValidator

INPUT_DIR = ROOT / "input" / "111 PCB back"
DEBUG_DIR = ROOT / "test_output" / "111_pcb_back_debug"
DEBUG_DIR.mkdir(parents=True, exist_ok=True)

# 黑框尺寸 60mm × 30mm
FRAME_W = 60.0
FRAME_H = 30.0

pipeline = PCBRecognitionPipeline()

# ── Step 1: 分别处理正反面图片 ──
results = {}  # "front"/"back" -> pipeline result

for img_file in sorted(INPUT_DIR.glob("*.jpg")):
    label = img_file.stem  # "front" 或 "back"
    print(f"\n{'='*60}")
    print(f"处理: {img_file.name}")
    print(f"{'='*60}")

    image_bytes = img_file.read_bytes()
    debug_path = DEBUG_DIR / label
    debug_path.mkdir(parents=True, exist_ok=True)

    try:
        result = pipeline.run(
            image_bytes,
            frame_width_mm=FRAME_W,
            frame_height_mm=FRAME_H,
            debug_dir=str(debug_path),
        )
        results[label] = result

        print(f"\n类型: {result.get('type', 'N/A')}")
        print(f"像素密度: {result.get('pixels_per_mm', 'N/A')} px/mm")

        outline = result.get("outline", [])
        print(f"轮廓顶点数: {len(outline)}")
        if outline:
            xk = "x_mm" if "x_mm" in outline[0] else "x"
            yk = "y_mm" if "y_mm" in outline[0] else "y"
            xs = [p[xk] for p in outline]
            ys = [p[yk] for p in outline]
            print(f"PCB尺寸: {max(xs)-min(xs):.2f}mm × {max(ys)-min(ys):.2f}mm")

        # 保存结果JSON（不含b64字段）
        result_save = {k: v for k, v in result.items()
                       if k not in ("transparent_pcb_b64", "rectified_png_b64")}
        out_json = DEBUG_DIR / f"{label}_result.json"
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(result_save, f, indent=2, ensure_ascii=False, default=str)

        # 保存单面透明PNG（未经校验）
        if result.get("transparent_pcb_b64"):
            png_data = base64.b64decode(result["transparent_pcb_b64"])
            png_path = DEBUG_DIR / f"{label}_transparent.png"
            with open(png_path, "wb") as f:
                f.write(png_data)

        # 保存校正图
        if result.get("rectified_png_b64"):
            rect_data = base64.b64decode(result["rectified_png_b64"])
            rect_path = DEBUG_DIR / f"{label}_rectified.png"
            with open(rect_path, "wb") as f:
                f.write(rect_data)

    except Exception as e:
        import traceback
        print(f"[ERROR] {e}")
        traceback.print_exc()

# ── Step 2: 正反面交叉校验 ──
if "front" in results and "back" in results:
    print(f"\n{'#'*60}")
    print(f"正反面交叉校验")
    print(f"{'#'*60}")

    try:
        cross_result = CrossValidator.validate(
            results["front"],
            results["back"],
            width_mm=FRAME_W,
            height_mm=FRAME_H,
        )

        consensus = cross_result["outline"]
        print(f"\n共识轮廓顶点数: {len(consensus)}")
        if consensus:
            xk = "x_mm" if "x_mm" in consensus[0] else "x"
            yk = "y_mm" if "y_mm" in consensus[0] else "y"
            xs = [p[xk] for p in consensus]
            ys = [p[yk] for p in consensus]
            print(f"共识PCB尺寸: {max(xs)-min(xs):.2f}mm × {max(ys)-min(ys):.2f}mm")
            for i, p in enumerate(consensus):
                print(f"  [{i}] x={p[xk]:.3f}mm  y={p[yk]:.3f}mm")

        print(f"\n正面面积:   {cross_result['front_area_mm2']} mm2")
        print(f"背面面积:   {cross_result['back_area_mm2']} mm2")
        print(f"共识面积:   {cross_result['consensus_area_mm2']} mm2")

        # 保存交叉校验结果JSON
        cross_save = {k: v for k, v in cross_result.items()
                      if k not in ("transparent_pcb_b64", "transparent_pcb_back_b64",
                                   "diff_image_b64")}
        cross_json = DEBUG_DIR / "cross_validate_result.json"
        with open(cross_json, "w", encoding="utf-8") as f:
            json.dump(cross_save, f, indent=2, ensure_ascii=False, default=str)
        print(f"\n结果已保存: {cross_json}")

        # 保存校验后正面透明PNG
        if cross_result.get("transparent_pcb_b64"):
            png_data = base64.b64decode(cross_result["transparent_pcb_b64"])
            png_path = DEBUG_DIR / "cross_front_transparent.png"
            with open(png_path, "wb") as f:
                f.write(png_data)
            print(f"校验后正面透明PNG: {png_path}")

        # 保存校验后背面透明PNG
        if cross_result.get("transparent_pcb_back_b64"):
            png_data = base64.b64decode(cross_result["transparent_pcb_back_b64"])
            png_path = DEBUG_DIR / "cross_back_transparent.png"
            with open(png_path, "wb") as f:
                f.write(png_data)
            print(f"校验后背面透明PNG: {png_path}")

        # 保存 diff 可视化图
        if cross_result.get("diff_image_b64"):
            diff_data = base64.b64decode(cross_result["diff_image_b64"])
            diff_path = DEBUG_DIR / "cross_diff.png"
            with open(diff_path, "wb") as f:
                f.write(diff_data)
            print(f"diff可视化图: {diff_path}")

    except Exception as e:
        import traceback
        print(f"[交叉校验 ERROR] {e}")
        traceback.print_exc()
else:
    print("\n缺少正反面图片，无法执行交叉校验")
