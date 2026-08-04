#!/usr/bin/env python
"""测试方向检测优化功能"""
from pathlib import Path
from PIL import Image
from io import BytesIO

from battery_designer.pcb_recognition.orientation_detector import OrientationDetector


def test_orientation_optimization(image_path: str):
    """测试方向检测优化

    验证：
    1. 检测图片方向（横屏和竖屏都需要）
    2. 横屏也检测是否需要旋转180°
    3. 竖屏检测应该旋转90°还是270°
    """
    print("="*60)
    print(f"测试图片: {image_path}")
    print("="*60)

    # 加载图片
    print("\n[Step 1] 加载图片")
    image_bytes = Path(image_path).read_bytes()
    image = Image.open(BytesIO(image_bytes))
    
    width, height = image.size
    is_landscape = width > height

    print(f"  原始尺寸: {width}x{height} ({'横屏' if is_landscape else '竖屏'})")

    # Step 2: 方向检测
    print("\n[Step 2] 方向检测")

    detector = OrientationDetector()

    print(f"  VLM 可用: {detector.vlm_available}")

    result_orient = detector.detect_orientation(image)

    if is_landscape:
        print(f"  图片类型: 横屏 (检查是否需要旋转180°)")
    else:
        print(f"  图片类型: 竖屏 (检查应该旋转90°还是270°)")

    print(f"  检测结果:")
    print(f"    - 方向: {result_orient['orientation']}°")
    print(f"    - 方法: {result_orient['method']}")
    print(f"    - 置信度: {result_orient['confidence']:.2f}")
    print(f"    - 需要旋转: {result_orient['needs_rotation']}")

    print("\n" + "="*60)


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        # 使用默认测试图片
        test_images = [
            "input/22 PCB/front.jpg",
            "input/111 PCB back/back.jpg",
        ]

        for img_path in test_images:
            if Path(img_path).exists():
                test_orientation_optimization(img_path)
                print()
    else:
        test_orientation_optimization(sys.argv[1])