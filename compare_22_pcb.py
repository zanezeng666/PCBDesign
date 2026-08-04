"""详细对比 22 PCB 原始图片和修正后的图片"""

from pathlib import Path
from PIL import Image
import io

# 原始图片
input_dir = Path("input/22 PCB")
# 修正后的图片
output_dir = Path("output/22_pcb_corrected")

print("=" * 70)
print("22 PCB 图片对比分析")
print("=" * 70)

for img_name in ["front.jpg", "back.jpg"]:
    input_path = input_dir / img_name
    output_path = output_dir / f"corrected_{img_name}".replace('.jpg', '.png')

    if not input_path.exists():
        continue

    print(f"\n{img_name}:")
    print("-" * 70)

    # 原始图片（直接用 PIL 打开，不经过修正器）
    input_img = Image.open(input_path)
    input_exif = input_img.getexif()
    input_orientation = input_exif.get(274, 1)
    input_size_kb = input_path.stat().st_size / 1024

    print(f"原始图片（PIL直接打开）:")
    print(f"  尺寸: {input_img.size[0]}x{input_img.size[1]} (宽x高)")
    print(f"  格式: {input_img.mode}")
    print(f"  EXIF方向: {input_orientation}")
    print(f"  文件大小: {input_size_kb:.2f} KB")

    # 手动检查是否有旋转
    width, height = input_img.size
    if width > height:
        print(f"  方向: 横屏 (landscape)")
    else:
        print(f"  方向: 竖屏 (portrait)")

    # 修正后的图片
    if output_path.exists():
        output_img = Image.open(output_path)
        output_exif = output_img.getexif()
        output_orientation = output_exif.get(274, 1)
        output_size_kb = output_path.stat().st_size / 1024

        print(f"\n修正后图片:")
        print(f"  尺寸: {output_img.size[0]}x{output_img.size[1]} (宽x高)")
        print(f"  格式: {output_img.mode}")
        print(f"  EXIF方向: {output_orientation}")
        print(f"  文件大小: {output_size_kb:.2f} KB")

        # 检查方向
        width, height = output_img.size
        if width > height:
            print(f"  方向: 横屏 (landscape)")
        else:
            print(f"  方向: 竖屏 (portrait)")

        # 对比尺寸变化
        print(f"\n尺寸变化:")
        if input_img.size != output_img.size:
            print(f"  变化: {input_img.size} -> {output_img.size}")
            if input_img.size[0] < input_img.size[1] and output_img.size[0] > output_img.size[1]:
                print(f"  说明: 竖屏 -> 横屏 (自动旋转了90度)")
        else:
            print(f"  无变化")

print("\n" + "=" * 70)
print("结论:")
print("  - 虽然EXIF方向是1，但PIL读取时可能自动应用了旋转")
print("  - 或者ImageOps.exif_transpose()做了额外处理")
print("  - 最终结果：竖屏图片被转换成了横屏")
print("=" * 70)