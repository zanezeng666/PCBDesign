"""检查输出图片的详细信息"""

from pathlib import Path
from PIL import Image

output_dir = Path("output/333_pcb_corrected")
files = list(output_dir.glob("*.png"))

print("输出图片信息:")
print("-" * 70)

for f in sorted(files):
    img = Image.open(f)
    exif = img.getexif()
    orientation = exif.get(274, 1)

    print(f"{f.name}:")
    print(f"  尺寸: {img.size[0]}x{img.size[1]}")
    print(f"  格式: {img.mode}")
    print(f"  EXIF方向: {orientation}")
    print()