"""Parametric battery protection board design service."""

__version__ = "0.1.0"

# 导出日志功能
from .logger import get_logger, configure_logging, set_log_level

# 导出识别流程模块
from .pcb_contour import detect_pcb_outline
from .pcb_recognition import PCBRecognitionPipeline
from .pad_detection import detect_pads, verify_pad_alignment
from .component_detection import detect_components_on_pcb, identify_ic_model

__all__ = [
    # 日志系统
    "get_logger",
    "configure_logging",
    "set_log_level",
    # PCB轮廓识别 (已成熟，保持稳定)
    "detect_pcb_outline",
    "PCBRecognitionPipeline",
    # 焊盘识别 (重点优化模块)
    "detect_pads",
    "verify_pad_alignment",
    # 元器件识别
    "detect_components_on_pcb",
    "identify_ic_model",
    # 版本
    "__version__",
]
