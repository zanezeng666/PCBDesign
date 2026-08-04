"""识别流程模块测试脚本

测试三个独立模块的基本功能：
  1. PCB轮廓识别 (已成熟)
  2. 焊盘识别 (重点优化)
  3. 元器件识别
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
    detect_pcb_outline,
    detect_pads,
    detect_components_on_pcb,
)

# 配置日志
configure_logging(level=logging.INFO, console=True)
logger = get_logger(__name__)


def test_module_imports():
    """测试模块导入"""
    logger.info("=" * 60)
    logger.info("测试 1: 模块导入")
    logger.info("=" * 60)

    try:
        from battery_designer import (
            detect_pcb_outline,
            refine_outline,
            detect_pads,
            verify_pad_alignment,
            detect_components_on_pcb,
            identify_ic_model,
        )
        logger.info("[OK] 所有模块导入成功")
        return True
    except Exception as e:
        logger.error("✗ 模块导入失败: %s", e)
        return False


def test_pcb_contour_api():
    """测试PCB轮廓识别API"""
    logger.info("\n" + "=" * 60)
    logger.info("测试 2: PCB轮廓识别API")
    logger.info("=" * 60)

    try:
        from battery_designer.pcb_contour import detect_pcb_outline, refine_outline

        # 检查函数签名
        import inspect

        sig = inspect.signature(detect_pcb_outline)
        params = list(sig.parameters.keys())

        logger.info("detect_pcb_outline 参数: %s", params)
        logger.info("[OK] PCB轮廓识别API检查通过")
        return True

    except Exception as e:
        logger.error("✗ PCB轮廓识别API测试失败: %s", e)
        return False


def test_pad_detection_api():
    """测试焊盘识别API"""
    logger.info("\n" + "=" * 60)
    logger.info("测试 3: 焊盘识别API")
    logger.info("=" * 60)

    try:
        from battery_designer.pad_detection import detect_pads, verify_pad_alignment

        # 检查函数签名
        import inspect

        sig = inspect.signature(detect_pads)
        params = list(sig.parameters.keys())

        logger.info("detect_pads 参数: %s", params)
        logger.info("[OK] 焊盘识别API检查通过")
        return True

    except Exception as e:
        logger.error("✗ 焊盘识别API测试失败: %s", e)
        return False


def test_component_detection_api():
    """测试元器件识别API"""
    logger.info("\n" + "=" * 60)
    logger.info("测试 4: 元器件识别API")
    logger.info("=" * 60)

    try:
        from battery_designer.component_detection import (
            detect_components_on_pcb,
            identify_ic_model,
        )

        # 检查函数签名
        import inspect

        sig = inspect.signature(detect_components_on_pcb)
        params = list(sig.parameters.keys())

        logger.info("detect_components_on_pcb 参数: %s", params)
        logger.info("[OK] 元器件识别API检查通过")
        return True

    except Exception as e:
        logger.error("✗ 元器件识别API测试失败: %s", e)
        return False


def test_module_structure():
    """测试模块结构"""
    logger.info("\n" + "=" * 60)
    logger.info("测试 5: 模块文件结构")
    logger.info("=" * 60)

    try:
        module_files = [
            "battery_designer/pcb_contour.py",
            "battery_designer/pad_detection.py",
            "battery_designer/component_detection.py",
            "battery_designer/vision.py",
            "battery_designer/vlm_detection.py",
        ]

        all_exist = True
        for file_path in module_files:
            full_path = ROOT / file_path
            if full_path.exists():
                logger.info("  [OK] %s", file_path)
            else:
                logger.error("  [MISSING] %s (不存在)", file_path)
                all_exist = False

        if all_exist:
            logger.info("[OK] 所有模块文件存在")
            return True
        else:
            logger.error("[FAIL] 部分模块文件缺失")
            return False

    except Exception as e:
        logger.error("✗ 模块结构检查失败: %s", e)
        return False


def main():
    """运行所有测试"""
    logger.info("开始识别流程模块测试...")
    logger.info("")

    tests = [
        ("模块导入", test_module_imports),
        ("PCB轮廓识别API", test_pcb_contour_api),
        ("焊盘识别API", test_pad_detection_api),
        ("元器件识别API", test_component_detection_api),
        ("模块文件结构", test_module_structure),
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
        logger.info("[SUCCESS] 所有测试通过！模块结构正常。")
        return 0
    else:
        logger.error("[WARNING] 部分测试失败，请检查模块实现。")
        return 1


if __name__ == "__main__":
    sys.exit(main())