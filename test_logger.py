#!/usr/bin/env python3
"""测试日志系统"""

import logging
import sys
from pathlib import Path

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent))

from battery_designer import configure_logging, get_logger, set_log_level
from battery_designer.logger import log_function, log_errors


def test_basic_logging():
    """测试基本日志功能"""
    print("\n=== 测试基本日志功能 ===")
    
    # 配置日志（仅控制台）
    configure_logging(level=logging.DEBUG, console=True)
    
    logger = get_logger(__name__)
    
    logger.debug("这是 DEBUG 信息")
    logger.info("这是 INFO 信息")
    logger.warning("这是 WARNING 信息")
    logger.error("这是 ERROR 信息")
    logger.critical("这是 CRITICAL 信息")


def test_file_logging():
    """测试文件日志"""
    print("\n=== 测试文件日志 ===")
    
    # 配置日志（控制台 + 文件）
    configure_logging(
        level=logging.INFO,
        log_dir="logs",
        console=True
    )
    
    logger = get_logger("test.file")
    
    logger.info("这条信息会同时输出到控制台和文件")
    logger.debug("这条 DEBUG 信息不会显示在控制台，但会记录到文件")
    
    print("请检查 logs/ 目录下的日志文件")


def test_log_decorator():
    """测试日志装饰器"""
    print("\n=== 测试日志装饰器 ===")
    
    configure_logging(level=logging.DEBUG, console=True)
    logger = get_logger(__name__)
    
    @log_function(level=logging.INFO, log_time=True)
    def process_data(data):
        """处理数据的示例函数"""
        import time
        time.sleep(0.1)
        return data.upper()
    
    result = process_data("hello world")
    print(f"返回值: {result}")


def test_error_decorator():
    """测试异常捕获装饰器"""
    print("\n=== 测试异常捕获装饰器 ===")
    
    configure_logging(level=logging.INFO, console=True)
    
    @log_errors("处理失败", reraise=False)
    def risky_function(should_fail=True):
        """可能失败的函数"""
        if should_fail:
            raise ValueError("这是一个测试错误")
        return "成功"
    
    # 测试失败情况
    result = risky_function(should_fail=True)
    print(f"失败时返回: {result}")
    
    # 测试成功情况
    result = risky_function(should_fail=False)
    print(f"成功时返回: {result}")


def test_dynamic_level():
    """测试动态调整日志级别"""
    print("\n=== 测试动态调整日志级别 ===")
    
    configure_logging(level=logging.WARNING, console=True)
    logger = get_logger("test.dynamic")
    
    logger.info("这条 INFO 信息不应该显示")
    logger.warning("这是 WARNING 信息")
    
    # 动态调整为 DEBUG
    print(">>> 调整为 DEBUG 级别")
    set_log_level(logging.DEBUG)
    
    logger.info("现在可以看到 INFO 信息了")
    logger.debug("甚至可以看到 DEBUG 信息")


def main():
    """运行所有测试"""
    print("=" * 60)
    print("日志系统测试")
    print("=" * 60)
    
    try:
        test_basic_logging()
        test_file_logging()
        test_log_decorator()
        test_error_decorator()
        test_dynamic_level()
        
        print("\n" + "=" * 60)
        print("所有测试完成！")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n测试失败: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())