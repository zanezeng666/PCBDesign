"""测试PCB轮廓识别"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import cv2
import numpy as np
from battery_designer.vision import extract_pcb

img_path = r"E:\worksapce\PCBDesign\input\111 PCB back\front.jpg"
img = cv2.imread(img_path)
h, w = img.shape[:2]

_, buf = cv2.imencode('.png', img)
img_bytes = buf.tobytes()

result = extract_pcb(img_bytes, 100.0, 60.0, 50.0)

import base64
transparent_bytes = base64.b64decode(result['transparent_pcb_b64'])
nparr = np.frombuffer(transparent_bytes, np.uint8)
transparent_img = cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)
cv2.imwrite(r"E:\worksapce\PCBDesign\test_output.png", transparent_img)
print(f"已保存: E:\\worksapce\\PCBDesign\\test_output.png")
print(f"轮廓顶点: {len(result['outline'])}")