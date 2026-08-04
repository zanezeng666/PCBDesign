"""
Engine 模块日志配置

提供 engine 包的统一日志管理。
"""

import logging
from battery_designer.logger import get_logger as _get_logger

# 获取 engine 模块的根 logger
logger = _get_logger("engine")


def get_logger(name: str = None) -> logging.Logger:
    """
    获取 engine 子模块的 logger
    
    Args:
        name: 子模块名称，通常使用 __name__
    
    Returns:
        Logger 实例
    
    Example:
        logger = get_logger(__name__)  # 返回 engine.schematic
    """
    if name is None:
        return logger
    
    # 确保以 engine. 开头
    if not name.startswith("engine."):
        name = f"engine.{name}"
    
    return _get_logger(name)