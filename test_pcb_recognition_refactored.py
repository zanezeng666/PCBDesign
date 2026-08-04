"""测试重构后的PCB轮廓识别模块

验证每个步骤的类是否正常工作。
"""

import sys
from pathlib import Path

# 添加项目根目录到Python路径
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import logging
from battery_designer import (
    get_logger,
    configure_logging,
)

# 配置日志
configure_logging(level=logging.INFO, console=True)
logger = get_logger(__name__)


def test_step_classes():
    """测试各个步骤类是否可以正常导入和初始化"""
    logger.info("=" * 60)
    logger.info("测试 1: 步骤类导入和初始化")
    logger.info("=" * 60)

    try:
        from battery_designer.pcb_recognition import (
            BlackFrameDetector,
            PerspectiveCalibrator,
            HSVPCBExtractor,
            PaperModelBuilder,
            GrooveValidator,
            TransparentPNGGenerator,
            PCBRecognitionPipeline,
        )

        logger.info("[OK] 所有步骤类导入成功")

        # 初始化各个步骤
        frame_detector = BlackFrameDetector()
        logger.info("[OK] BlackFrameDetector 初始化成功")

        calibrator = PerspectiveCalibrator()
        logger.info("[OK] PerspectiveCalibrator 初始化成功")

        pcb_extractor = HSVPCBExtractor()
        logger.info("[OK] HSVPCBExtractor 初始化成功")

        paper_builder = PaperModelBuilder()
        logger.info("[OK] PaperModelBuilder 初始化成功")

        groove_validator = GrooveValidator()
        logger.info("[OK] GrooveValidator 初始化成功")

        png_generator = TransparentPNGGenerator()
        logger.info("[OK] TransparentPNGGenerator 初始化成功")

        pipeline = PCBRecognitionPipeline()
        logger.info("[OK] PCBRecognitionPipeline 初始化成功")

        return True

    except Exception as e:
        logger.error("[FAIL] 步骤类测试失败: %s", e)
        import traceback
        traceback.print_exc()
        return False


def test_pipeline_api():
    """测试Pipeline API"""
    logger.info("\n" + "=" * 60)
    logger.info("测试 2: Pipeline API")
    logger.info("=" * 60)

    try:
        from battery_designer.pcb_recognition import PCBRecognitionPipeline
        import inspect

        pipeline = PCBRecognitionPipeline()

        # 检查run方法签名
        sig = inspect.signature(pipeline.run)
        params = list(sig.parameters.keys())

        logger.info("Pipeline.run 参数: %s", params)

        # 验证必需参数
        required_params = ["image_bytes", "frame_width_mm", "frame_height_mm"]
        for param in required_params:
            if param not in params:
                logger.error("[FAIL] 缺少必需参数: %s", param)
                return False

        logger.info("[OK] Pipeline API 检查通过")
        return True

    except Exception as e:
        logger.error("[FAIL] Pipeline API 测试失败: %s", e)
        import traceback
        traceback.print_exc()
        return False


def test_file_structure():
    """测试文件结构"""
    logger.info("\n" + "=" * 60)
    logger.info("测试 3: 文件结构")
    logger.info("=" * 60)

    try:
        expected_files = [
            "battery_designer/pcb_recognition/__init__.py",
            "battery_designer/pcb_recognition/black_frame_detector.py",
            "battery_designer/pcb_recognition/perspective_calibrator.py",
            "battery_designer/pcb_recognition/hsv_pcb_extractor.py",
            "battery_designer/pcb_recognition/paper_model_builder.py",
            "battery_designer/pcb_recognition/groove_validator.py",
            "battery_designer/pcb_recognition/transparent_png_generator.py",
            "battery_designer/pcb_recognition/pipeline.py",
        ]

        all_exist = True
        for file_path in expected_files:
            full_path = ROOT / file_path
            if full_path.exists():
                logger.info("  [OK] %s", file_path)
            else:
                logger.error("  [MISSING] %s", file_path)
                all_exist = False

        if all_exist:
            logger.info("[OK] 所有文件存在")
            return True
        else:
            logger.error("[FAIL] 部分文件缺失")
            return False

    except Exception as e:
        logger.error("[FAIL] 文件结构检查失败: %s", e)
        return False


def test_api_compatibility():
    """测试API兼容性"""
    logger.info("\n" + "=" * 60)
    logger.info("测试 4: API 兼容性")
    logger.info("=" * 60)

    try:
        # 测试原有的API是否仍然可用
        from battery_designer import detect_pcb_outline
        import inspect

        sig = inspect.signature(detect_pcb_outline)
        params = list(sig.parameters.keys())

        logger.info("detect_pcb_outline 参数: %s", params)

        # 验证基本参数
        required_params = ["image_bytes", "frame_width_mm", "frame_height_mm"]
        for param in required_params:
            if param not in params:
                logger.error("[FAIL] 缺少必需参数: %s", param)
                return False

        logger.info("[OK] API 兼容性检查通过")
        return True

    except Exception as e:
        logger.error("[FAIL] API 兼容性测试失败: %s", e)
        import traceback
        traceback.print_exc()
        return False


def main():
    """运行所有测试"""
    logger.info("开始 PCB 轮廓识别模块重构测试...")
    logger.info("")

    tests = [
        ("步骤类导入和初始化", test_step_classes),
        ("Pipeline API", test_pipeline_api),
        ("文件结构", test_file_structure),
        ("API 兼容性", test_api_compatibility),
    ]

    results = []
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            logger.error("测试 '%s' 异常: %s", name, e)
            results.append((name, False))

    # ── 汇总结果 ──
    logger.info("\n" + "=" * 60)
    logger.info("测试结果汇总")
    logger.info("=" * 60)

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for name, result in results:
        status = "[PASS]" if result else "[FAIL]"
        logger.info("%s - %s", status, name)

    logger.info("")
    logger.info("总计: %d/%d 测试通过", passed, total)

    if passed == total:
        logger.info("[SUCCESS] 所有测试通过！重构成功。")
        return 0
    else:
        logger.error("[WARNING] 部分测试失败，请检查实现。")
        return 1


if __name__ == "__main__":
    sys.exit(main())