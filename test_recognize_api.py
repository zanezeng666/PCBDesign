"""测试 /api/vision/recognize-pcb 统一识别接口"""
import json
import os
import sys

from battery_designer.app import app
from fastapi.testclient import TestClient

client = TestClient(app)

front_path = "input/111 PCB back/front.jpg"
back_path = "input/111 PCB back/back.jpg"

if not os.path.exists(front_path) or not os.path.exists(back_path):
    print("SKIP: input files not found")
    sys.exit(0)

with open(front_path, "rb") as f, open(back_path, "rb") as b:
    resp = client.post(
        "/api/vision/recognize-pcb",
        files={
            "front_image": ("front.jpg", f, "image/jpeg"),
            "back_image": ("back.jpg", b, "image/jpeg"),
        },
        data={"frame_w_mm": "60.0", "frame_h_mm": "30.0"},
    )

print(f"Status: {resp.status_code}")
data = resp.json()
print(f"success: {data['success']}")
print(f"frame_w_mm: {data['frame_w_mm']}")
print(f"frame_h_mm: {data['frame_h_mm']}")
print(f"pcb_width_mm: {data['pcb_width_mm']}")
print(f"pcb_height_mm: {data['pcb_height_mm']}")
print(f"outline_vertex_count: {data['outline_vertex_count']}")
print(f"front.calibration_success: {data['front']['calibration_success']}")
print(f"back.calibration_success: {data['back']['calibration_success']}")
print(f"front.outline points: {len(data['front']['outline'])}")
print(f"back.outline points: {len(data['back']['outline'])}")
print(f"consensus.ok: {data['consensus']['ok']}")
print(f"consensus.message: {data['consensus']['message']}")
print(f"consensus.deviation_pct: {data['consensus']['deviation_pct']}")
print(f"front.overlay_b64 length: {len(data['front'].get('overlay_b64', ''))}")
print(f"back.overlay_b64 length: {len(data['back'].get('overlay_b64', ''))}")
print(f"front.transparent_b64 length: {len(data['front'].get('transparent_pcb_b64', ''))}")
print(f"back.transparent_b64 length: {len(data['back'].get('transparent_pcb_b64', ''))}")

if data["front"].get("error"):
    print(f"front.error: {data['front']['error']}")
if data["back"].get("error"):
    print(f"back.error: {data['back']['error']}")

# 保存 overlay 图片到 test_output 用于目视检查
os.makedirs("test_output/recognize", exist_ok=True)
for side in ("front", "back"):
    b64 = data[side].get("overlay_b64", "")
    if b64:
        import base64

        with open(f"test_output/recognize/{side}_overlay.png", "wb") as f:
            f.write(base64.b64decode(b64))
        print(f"Saved: test_output/recognize/{side}_overlay.png")

    tb64 = data[side].get("transparent_pcb_b64", "")
    if tb64:
        import base64

        with open(f"test_output/recognize/{side}_transparent.png", "wb") as f:
            f.write(base64.b64decode(tb64))
        print(f"Saved: test_output/recognize/{side}_transparent.png")

cb64 = data["consensus"].get("transparent_pcb_b64", "")
if cb64:
    import base64

    with open("test_output/recognize/consensus_transparent.png", "wb") as f:
        f.write(base64.b64decode(cb64))
    print("Saved: test_output/recognize/consensus_transparent.png")

# 保存完整 JSON
with open("test_output/recognize/result.json", "w", encoding="utf-8") as f:
    # 截断 base64 以方便阅读
    summary = {k: v for k, v in data.items()
               if k not in ("front", "back", "consensus")}
    summary["front_outline_pts"] = len(data["front"]["outline"])
    summary["back_outline_pts"] = len(data["back"]["outline"])
    json.dump(summary, f, ensure_ascii=False, indent=2)
print("Saved: test_output/recognize/result.json")
