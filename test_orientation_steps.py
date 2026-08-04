"""测试 Pipeline 的方向检测步骤"""

import sys
import logging
from pathlib import Path

# 添加项目根目录到Python路径
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from battery_designer import get_logger, configure_logging
from battery_designer.pcb_recognition import OrientationDetector
from PIL import Image
from io import BytesIO

# 配置日志
configure_logging(level=logging.INFO, console=True)
logger = get_logger(__name__)


def test_orientation_step():
    """测试方向检测步骤"""
    logger.info("=" * 70)
    logger.info("测试 Pipeline 方向检测步骤")
    logger.info("=" * 70)

    # 测试图片
    test_images = [
        ROOT / "input" / "22 PCB" / "front.jpg",
        ROOT / "input" / "333 PCB" / "front.jpg",
    ]

    # 创建处理器
    orientation_detector = OrientationDetector()

    # 输出目录
    output_dir = ROOT / "output" / "pipeline_steps"
    output_dir.mkdir(parents=True, exist_ok=True)

    for img_path in test_images:
        if not img_path.exists():
            logger.warning("图片不存在: %s", img_path)
            continue

        logger.info("\n处理图片: %s", img_path.name)
        logger.info("-" * 70)

        try:
            # 读取原始图片
            image_bytes = img_path.read_bytes()
            img = Image.open(img_path)

            logger.info("原始图片: %dx%d", img.size[0], img.size[1])

            # ── Step 1: 方向检测 ──
            logger.info("\nStep 1/1: 方向检测")
            orientation_result = orientation_detector.detect_orientation(img)

            logger.info("  结果: orientation=%d°, method=%s, confidence=%.2f",
                orientation_result["orientation"],
                orientation_result["method"],
                orientation_result["confidence"]
            )
            logger.info("  是否需要旋转: %s", orientation_result["needs_rotation"])

            # 如果需要旋转
            if orientation_result["needs_rotation"]:
                rotated_img = orientation_detector.rotate_to_landscape(
                    img,
                    orientation_result["orientation"]
                )

                # 确保是 PIL.Image
                if hasattr(rotated_img, 'shape'):
                    # numpy array → PIL.Image
                    rotated_img = Image.fromarray(rotated_img[:, :, [2, 1, 0]])

                logger.info("  旋转后尺寸: %dx%d", rotated_img.size[0], rotated_img.size[1])
            else:
                rotated_img = img

            # 保存结果
            output_path = output_dir / f"processed_{img_path.name}"
            output_path = output_path.with_suffix('.png')
            rotated_img.save(output_path, format='PNG')

            logger.info("\n已保存: %s", output_path)
            logger.info("[OK] %s 处理完成", img_path.name)

        except Exception as e:
            logger.error("[FAIL] %s 处理失败: %s", img_path, e)
            import traceback
            traceback.print_exc()

    # ── 汇总 ──
    logger.info("\n" + "=" * 70)
    logger.info("测试完成！")
    logger.info("输出目录: %s", output_dir)

    output_files = list(output_dir.glob("*.png"))
    logger.info("生成文件数: %d", len(output_files))

    for f in sorted(output_files):
        img = Image.open(f)
        size_kb = f.stat().st_size / 1024
        direction = "横屏" if img.size[0] > img.size[1] else "竖屏"
        logger.info("  - %s: %dx%d (%s, %.2f KB)",
            f.name, img.size[0], img.size[1], direction, size_kb
        )


if __name__ == "__main__":
    test_orientation_step()