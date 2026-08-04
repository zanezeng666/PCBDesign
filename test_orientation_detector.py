"""测试方向检测和强制横屏功能"""

import sys
import logging
from pathlib import Path

# 添加项目根目录到Python路径
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from battery_designer import get_logger, configure_logging
from battery_designer.pcb_recognition import OrientationDetector
from PIL import Image

# 配置日志
configure_logging(level=logging.INFO, console=True)
logger = get_logger(__name__)


def test_orientation_detector():
    """测试方向检测器"""
    logger.info("=" * 70)
    logger.info("测试方向检测和强制横屏功能")
    logger.info("=" * 70)

    # 测试图片路径
    test_images = [
        ROOT / "input" / "22 PCB" / "front.jpg",
        ROOT / "input" / "22 PCB" / "back.jpg",
    ]

    # 创建检测器
    detector = OrientationDetector()

    # 输出目录
    output_dir = ROOT / "output" / "orientation_corrected"
    output_dir.mkdir(parents=True, exist_ok=True)

    for img_path in test_images:
        if not img_path.exists():
            logger.warning("图片不存在: %s", img_path)
            continue

        logger.info("\n处理图片: %s", img_path.name)
        logger.info("-" * 70)

        # 读取原始图片
        img = Image.open(img_path)
        original_size = img.size
        original_size_kb = img_path.stat().st_size / 1024

        logger.info("原始图片:")
        logger.info("  - 尺寸: %dx%d", original_size[0], original_size[1])
        logger.info("  - 方向: %s", "横屏" if original_size[0] > original_size[1] else "竖屏")
        logger.info("  - 文件大小: %.2f KB", original_size_kb)

        # ── Step 1: 检测方向 ──
        result = detector.detect_orientation(img)

        logger.info("\n检测结果:")
        logger.info("  - 建议旋转: %d°", result["orientation"])
        logger.info("  - 置信度: %.2f", result["confidence"])
        logger.info("  - 检测方法: %s", result["method"])
        logger.info("  - 是否需要旋转: %s", result["needs_rotation"])

        # ── Step 2: 旋转到正确方向 ──
        if result["needs_rotation"]:
            rotated = detector.rotate_to_landscape(img, result["orientation"])

            logger.info("\n旋转结果:")
            logger.info("  - 新尺寸: %dx%d", rotated.size[0], rotated.size[1])
            logger.info("  - 方向: %s", "横屏" if rotated.size[0] > rotated.size[1] else "竖屏")

            # 保存结果
            output_path = output_dir / f"corrected_{img_path.name}"
            output_path = output_path.with_suffix('.png')
            rotated.save(output_path, format='PNG')

            logger.info("  - 已保存到: %s", output_path)

            # 验证输出
            output_size_kb = output_path.stat().st_size / 1024
            logger.info("  - 输出大小: %.2f KB", output_size_kb)
        else:
            logger.info("\n无需旋转，图片方向正确")

            # 保存原图
            output_path = output_dir / f"corrected_{img_path.name}"
            output_path = output_path.with_suffix('.png')
            img.save(output_path, format='PNG')
            logger.info("已保存到: %s", output_path)

        logger.info("[OK] %s 处理完成", img_path.name)

    # ── 汇总结果 ──
    logger.info("\n" + "=" * 70)
    logger.info("测试结果汇总")
    logger.info("=" * 70)

    output_files = list(output_dir.glob("*.png"))
    for f in sorted(output_files):
        img = Image.open(f)
        size_kb = f.stat().st_size / 1024
        direction = "横屏" if img.size[0] > img.size[1] else "竖屏"
        logger.info(
            "  - %s: %dx%d (%s, %.2f KB)",
            f.name,
            img.size[0],
            img.size[1],
            direction,
            size_kb,
        )

    logger.info("\n测试完成！")
    logger.info("请查看输出目录: %s", output_dir)


def test_ocr_availability():
    """测试 OCR 功能是否可用"""
    logger.info("\n" + "=" * 70)
    logger.info("测试 OCR 功能")
    logger.info("=" * 70)

    detector = OrientationDetector()

    if detector.ocr_available:
        logger.info("OCR 功能可用: pytesseract 已安装")

        # 测试简单 OCR
        test_img = Image.new('RGB', (100, 50), color='white')
        from PIL import ImageDraw
        draw = ImageDraw.Draw(test_img)
        draw.text((10, 10), "TEST", fill='black')

        import pytesseract
        text = pytesseract.image_to_string(test_img)
        logger.info("OCR 测试结果: '%s'", text.strip())

    else:
        logger.warning("OCR 功能不可用: pytesseract 未安装")
        logger.info("将使用启发式方法判断方向")
        logger.info("安装方法: pip install pytesseract")
        logger.info("注意：还需要安装 Tesseract-OCR 软件")


if __name__ == "__main__":
    # 测试 OCR
    test_ocr_availability()

    # 测试方向检测
    test_orientation_detector()