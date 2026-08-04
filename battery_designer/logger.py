"""
统一日志配置模块

提供项目全局的日志配置和管理功能。

特性:
- 统一的日志格式
- 支持控制台和文件输出
- 可配置的日志级别
- 按模块自动命名 logger

使用方法:
    from battery_designer.logger import get_logger
    
    logger = get_logger(__name__)
    logger.info("操作成功")
    logger.error("发生错误", exc_info=True)
"""

import logging
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional


# 日志格式定义
LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# 默认日志级别
DEFAULT_LOG_LEVEL = logging.INFO

# 全局 logger 配置标志
_configured = False


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """
    获取指定名称的 logger 实例
    
    Args:
        name: logger 名称，通常使用 __name__
              如果为 None，返回根 logger
    
    Returns:
        配置好的 Logger 实例
    
    Example:
        logger = get_logger(__name__)
        logger.info("开始处理")
    """
    global _configured
    
    # 首次调用时进行基本配置
    if not _configured:
        configure_logging()
        _configured = True
    
    return logging.getLogger(name)


def configure_logging(
    level: int = None,
    log_file: Optional[str] = None,
    log_dir: Optional[str] = None,
    format_string: str = LOG_FORMAT,
    date_format: str = DATE_FORMAT,
    console: bool = True,
    file_level: Optional[int] = None
) -> None:
    """
    配置全局日志系统
    
    Args:
        level: 控制台日志级别 (默认: INFO)
        log_file: 日志文件路径 (优先级高于 log_dir)
        log_dir: 日志目录，自动生成带日期的日志文件名
        format_string: 日志格式字符串
        date_format: 日期格式字符串
        console: 是否输出到控制台
        file_level: 文件日志级别 (默认与 level 相同)
    
    Example:
        # 基本配置 (仅控制台)
        configure_logging(level=logging.DEBUG)
        
        # 输出到文件
        configure_logging(log_file="app.log")
        
        # 自动生成日志文件
        configure_logging(log_dir="logs")
    """
    global _configured
    
    # 设置默认级别
    if level is None:
        level = DEFAULT_LOG_LEVEL
    if file_level is None:
        file_level = level
    
    # 获取根 logger
    root_logger = logging.getLogger()
    root_logger.setLevel(min(level, file_level))
    
    # 清除已有的 handlers
    root_logger.handlers.clear()
    
    # 创建 formatter
    formatter = logging.Formatter(format_string, datefmt=date_format)
    
    # 控制台 handler
    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)
    
    # 文件 handler
    if log_file or log_dir:
        if log_file:
            log_path = Path(log_file)
        else:
            # 自动生成日志文件名
            log_path = Path(log_dir) / f"battery_designer_{datetime.now().strftime('%Y%m%d')}.log"
        
        # 确保日志目录存在
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_path, encoding='utf-8')
        file_handler.setLevel(file_level)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
    
    _configured = True


def set_log_level(level: int) -> None:
    """
    动态调整日志级别
    
    Args:
        level: logging.DEBUG, INFO, WARNING, ERROR, CRITICAL
    """
    logging.getLogger().setLevel(level)
    for handler in logging.getLogger().handlers:
        if isinstance(handler, logging.StreamHandler):
            handler.setLevel(level)


# 环境变量控制
def configure_from_env() -> None:
    """
    从环境变量读取日志配置
    
    支持的环境变量:
        LOG_LEVEL: 日志级别 (DEBUG, INFO, WARNING, ERROR)
        LOG_FILE: 日志文件路径
        LOG_DIR: 日志目录
    """
    import os
    
    # 日志级别
    level_name = os.getenv("LOG_LEVEL", "INFO").upper()
    level_map = {
        "DEBUG": logging.DEBUG,
        "INFO": logging.INFO,
        "WARNING": logging.WARNING,
        "ERROR": logging.ERROR,
        "CRITICAL": logging.CRITICAL,
    }
    level = level_map.get(level_name, logging.INFO)
    
    # 日志文件
    log_file = os.getenv("LOG_FILE")
    log_dir = os.getenv("LOG_DIR")
    
    configure_logging(level=level, log_file=log_file, log_dir=log_dir)


# 便捷函数：创建模块专用 logger
def create_module_logger(module_name: str) -> logging.Logger:
    """
    为模块创建专用的 logger
    
    Args:
        module_name: 模块名称 (通常传 __name__)
    
    Returns:
        Logger 实例
    """
    return get_logger(module_name)


# 用于兼容现有 print 语句的适配器
class PrintAdapter:
    """
    将 print 语句转换为日志输出的适配器
    
    用于逐步迁移现有的 print 语句到 logging
    """
    
    def __init__(self, logger: logging.Logger):
        self.logger = logger
    
    def info(self, *args, **kwargs):
        """替换 print() -> logger.info()"""
        msg = " ".join(str(arg) for arg in args)
        self.logger.info(msg)
    
    def debug(self, *args, **kwargs):
        """替换 print() -> logger.debug()"""
        msg = " ".join(str(arg) for arg in args)
        self.logger.debug(msg)
    
    def warning(self, *args, **kwargs):
        """替换 print() -> logger.warning()"""
        msg = " ".join(str(arg) for arg in args)
        self.logger.warning(msg)
    
    def error(self, *args, **kwargs):
        """替换 print() -> logger.error()"""
        msg = " ".join(str(arg) for arg in args)
        self.logger.error(msg)


# 日志装饰器
import functools
import time
from typing import Callable, Optional


def log_function(
    level: int = logging.INFO,
    include_args: bool = False,
    include_result: bool = False,
    log_time: bool = True,
    logger: Optional[logging.Logger] = None
) -> Callable:
    """
    函数日志装饰器，自动记录函数调用、执行时间和异常
    
    Args:
        level: 日志级别
        include_args: 是否记录函数参数
        include_result: 是否记录返回值
        log_time: 是否记录执行时间
        logger: 使用的 logger，默认使用调用模块的 logger
    
    Example:
        @log_function(level=logging.INFO, log_time=True)
        def process_data(data):
            return data.upper()
        
        # 输出: INFO: Calling process_data() - 0.002s
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # 获取 logger
            nonlocal logger
            if logger is None:
                logger = logging.getLogger(func.__module__)
            
            # 构建日志消息
            msg_parts = [f"Calling {func.__name__}()"]
            
            if include_args:
                args_repr = [repr(a) for a in args]
                kwargs_repr = [f"{k}={v!r}" for k, v in kwargs.items()]
                signature = ", ".join(args_repr + kwargs_repr)
                msg_parts.append(f"with args: {signature}")
            
            # 记录开始
            logger.log(level, " -> ".join(msg_parts))
            
            # 执行函数
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                
                # 记录成功
                if log_time:
                    elapsed = time.time() - start_time
                    msg_parts.append(f"- {elapsed:.3f}s")
                
                if include_result:
                    msg_parts.append(f"=> {result!r}")
                
                logger.log(level, " -> ".join(msg_parts))
                return result
                
            except Exception as e:
                # 记录异常
                elapsed = time.time() - start_time
                logger.exception(
                    f"Error in {func.__name__}() after {elapsed:.3f}s: {e}"
                )
                raise
        
        return wrapper
    return decorator


def log_errors(
    message: str = "Function failed",
    reraise: bool = True,
    logger: Optional[logging.Logger] = None
) -> Callable:
    """
    异常捕获装饰器，自动记录异常
    
    Args:
        message: 错误消息前缀
        reraise: 是否重新抛出异常
        logger: 使用的 logger
    
    Example:
        @log_errors("Failed to process data")
        def process_data(data):
            # ... 可能抛出异常的代码 ...
            pass
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            nonlocal logger
            if logger is None:
                logger = logging.getLogger(func.__module__)
            
            try:
                return func(*args, **kwargs)
            except Exception as e:
                logger.exception(f"{message}: {e}")
                if reraise:
                    raise
                return None
        
        return wrapper
    return decorator


if __name__ == "__main__":
    # 测试日志配置
    configure_logging(level=logging.DEBUG, log_dir="logs")
    
    logger = get_logger(__name__)
    logger.debug("调试信息")
    logger.info("普通信息")
    logger.warning("警告信息")
    logger.error("错误信息")
    
    print("\n日志配置测试完成")