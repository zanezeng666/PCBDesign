"""Parametric battery protection board design service."""

__version__ = "0.1.0"

# Core infrastructure
from .core.logger import get_logger, configure_logging, set_log_level

# Step 1 — PCB board recognition
from .board_recognition import PCBRecognitionPipeline

# Step 2 — Pad / terminal detection
from .pad_detection import detect_with_vlm, detect_all_vlm, verify_pad_crop

# Step 3 — Component detection
from .component_detection import detect_components

# Step 4 — Design generation
from .design_generation import DesignGenerator, KicadPipeline

__all__ = [
    "get_logger",
    "configure_logging",
    "set_log_level",
    # Step 1
    "PCBRecognitionPipeline",
    # Step 2
    "detect_with_vlm",
    "detect_all_vlm",
    "verify_pad_crop",
    # Step 3
    "detect_components",
    # Step 4
    "DesignGenerator",
    "KicadPipeline",
    "__version__",
]
