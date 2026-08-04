"""测试集成方向检测后的 Pipeline"""

import sys
import logging
from pathlib import Path

# 添加项目根目录到Python路径
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from battery_designer import get_logger, configure_logging
from battery_designer.pcb_recognition import PCBRecognitionPipeline

# 配置日志
configure_logging(level=logging.INFO, console=True)
logger = get_logger(__name__)


def test_pipeline_with_orientation():
    """测试集成方向检测后的完整 Pipeline"""
    logger.info("=" * 70)
    logger.info("测试集成方向检测后的 Pipeline")
    logger.info("=" * 70)

    # 测试图片
    test_images = [
        ROOT / "input" / "22 PCB" / "front.jpg",
        ROOT / "input" / "333 PCB" / "front.jpg",
    ]

    # 创建 Pipeline
    pipeline = PCBRecognitionPipeline()

    # 黑色方框尺寸（假设为100x100mm）
    frame_width_mm = 100.0
    frame_height_mm = 100.0

    results = []

    for img_path in test_images:
        if not img_path.exists():
            logger.warning("图片不存在: %s", img_path)
            continue

        logger.info("\n测试图片: %s", img_path)
        logger.info("-" * 70)

        try:
            # 读取图片
            image_bytes = img_path.read_bytes()
            logger.info("图片大小: %.2f KB", len(image_bytes) / 1024)

            # 运行 Pipeline
            result = pipeline.run(
                image_bytes,
                frame_width_mm,
                frame_height_mm,
                enable_groove_detection=False,
            )

            # 输出结果
            logger.info("\nPipeline 结果:")
            logger.info("  Calibration ID: %s", result["calibration_id"])
            logger.info("  像素密度: %.2f px/mm", result["pixels_per_mm"])

            # 检查方向检测步骤
            if "orientation_detection" in result["steps"]:
                orientation_step = result["steps"]["orientation_detection"]

                if "error" in orientation_step:
                    logger.warning("  方向检测错误: %s", orientation_step["error"])
                else:
                    logger.info("  方向检测:")
                    logger.info("    - orientation=%d°", orientation_step["orientation"])
                    logger.info("    - method=%s", orientation_step["method"])
                    logger.info("    - confidence=%.2f", orientation_step["confidence"])
                    logger.info("    - needs_rotation=%s", orientation_step["needs_rotation"])

            logger.info("[OK] Pipeline 运行成功")

            results.append({
                "image": img_path.name,
                "success": True,
                "calibration_id": result["calibration_id"],
            })

        except Exception as e:
            logger.error("[FAIL] Pipeline 运行失败: %s", e)
            import traceback
            traceback.print_exc()

            results.append({
                "image": img_path.name,
                "success": False,
                "error": str(e),
            })

    # ── 汇总结果 ──
    logger.info("\n" + "=" * 70)
    logger.info("测试结果汇总")
    logger.info("=" * 70)

    for r in results:
        if r["success"]:
            logger.info("[OK] %s - ID=%s", r["image"], r["calibration_id"])
        else:
            logger.error("[FAIL] %s - %s", r["image"], r.get("error", "Unknown error"))

    passed = sum(1 for r in results if r["success"])
    total = len(results)

    logger.info("\n总计: %d/%d 测试通过", passed, total)

    return passed == total


if __name__ == "__main__":
    success = test_pipeline_with_orientation()
    sys.exit(0 if success else 1)