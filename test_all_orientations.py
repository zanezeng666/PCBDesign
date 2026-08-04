"""综合测试：方向检测 + 强制横屏

测试所有 PCB 图片文件夹。
"""

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


def test_all_pcb_images():
    """测试所有 PCB 图片"""
    logger.info("=" * 70)
    logger.info("综合测试：方向检测 + 强制横屏")
    logger.info("=" * 70)

    # 所有测试文件夹
    test_folders = [
        ROOT / "input" / "22 PCB",
        ROOT / "input" / "333 PCB",
        ROOT / "input",
    ]

    # 创建处理器
    orientation_detector = OrientationDetector()

    # 输出目录
    output_base = ROOT / "output" / "all_corrected"
    output_base.mkdir(parents=True, exist_ok=True)

    results = []

    for folder in test_folders:
        if not folder.exists():
            continue

        # 获取所有 jpg 图片
        images = list(folder.glob("*.jpg"))

        if not images:
            continue

        logger.info("\n处理文件夹: %s", folder.name)
        logger.info("=" * 70)

        for img_path in images:
            logger.info("\n图片: %s", img_path.name)
            logger.info("-" * 70)

            try:
                # 读取原始图片
                image_bytes = img_path.read_bytes()
                img = Image.open(img_path)

                original_size = img.size
                original_direction = "横屏" if img.size[0] > img.size[1] else "竖屏"

                logger.info("原始图片:")
                logger.info("  尺寸: %dx%d (%s)", original_size[0], original_size[1], original_direction)
                logger.info("  大小: %.2f KB", len(image_bytes) / 1024)

                # ── Step 1: 方向检测 ──
                orientation_result = orientation_detector.detect_orientation(img)

                logger.info("\n方向检测:")
                logger.info("  建议: %d°, 置信度=%.2f, 方法=%s",
                    orientation_result["orientation"],
                    orientation_result["confidence"],
                    orientation_result["method"]
                )

                # ── Step 2: 强制横屏 ──
                if orientation_result["needs_rotation"]:
                    final_img = orientation_detector.rotate_to_landscape(
                        img,
                        orientation_result["orientation"]
                    )

                    logger.info("\n旋转结果:")
                    logger.info("  新尺寸: %dx%d", final_img.size[0], final_img.size[1])
                else:
                    final_img = img
                    logger.info("\n无需旋转")

                # ── Step 3: 保存结果 ──
                output_dir = output_base / folder.name
                output_dir.mkdir(parents=True, exist_ok=True)

                output_path = output_dir / f"final_{img_path.name}"
                output_path = output_path.with_suffix('.png')
                final_img.save(output_path, format='PNG')

                logger.info("\n已保存: %s", output_path)
                logger.info("  输出大小: %.2f KB", output_path.stat().st_size / 1024)

                results.append({
                    "folder": folder.name,
                    "image": img_path.name,
                    "original_size": original_size,
                    "original_direction": original_direction,
                    "final_size": final_img.size,
                    "final_direction": "横屏" if final_img.size[0] > final_img.size[1] else "竖屏",
                    "orientation": orientation_result["orientation"],
                    "method": orientation_result["method"],
                })

                logger.info("[OK] %s/%s 处理完成", folder.name, img_path.name)

            except Exception as e:
                logger.error("[FAIL] %s 处理失败: %s", img_path, e)
                import traceback
                traceback.print_exc()

    # ── 汇总结果 ──
    logger.info("\n" + "=" * 70)
    logger.info("测试结果汇总")
    logger.info("=" * 70)

    for r in results:
        logger.info(
            "%s/%s: %s (%dx%d) → %s (%dx%d) [旋转:%d°, %s]",
            r["folder"],
            r["image"],
            r["original_direction"],
            r["original_size"][0],
            r["original_size"][1],
            r["final_direction"],
            r["final_size"][0],
            r["final_size"][1],
            r["orientation"],
            r["method"],
        )

    logger.info("\n总计: %d 张图片处理完成", len(results))
    logger.info("输出目录: %s", output_base)

    return len(results) > 0


if __name__ == "__main__":
    success = test_all_pcb_images()
    sys.exit(0 if success else 1)