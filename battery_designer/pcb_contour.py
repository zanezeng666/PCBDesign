"""PCB轮廓识别模块 (已成熟，保持稳定)

这个模块已经过充分测试，保持稳定，不再修改。

流程:
  1. EXIF方向修正
  2. 黑色方框检测
  3. 透视校正
  4. HSV PCB提取
  5. 纸色模型构建
  6. 透明PNG生成

注意：孔洞检测已移除，应该在焊盘识别阶段处理。
"""

from __future__ import annotations

from .pcb_recognition import PCBRecognitionPipeline

# 导出公共API
__all__ = [
    "detect_pcb_outline",
    "PCBRecognitionPipeline",
]


def detect_pcb_outline(
    image_bytes: bytes,
    frame_width_mm: float,
    frame_height_mm: float,
) -> dict:
    """PCB轮廓识别 - 第一步（已成熟）

    从图片上传到识别透明PCB轮廓的完整流程。

    Args:
        image_bytes: 原始图片数据 (JPEG/PNG)
        frame_width_mm: 黑色方框宽度 (mm)
        frame_height_mm: 黑色方框高度 (mm)

    Returns:
        {
            "calibration_id": str,       # 校准ID
            "pixels_per_mm": float,      # 像素密度
            "width_mm": float,           # PCB宽度
            "height_mm": float,          # PCB高度
            "outline": [dict],           # PCB轮廓顶点 (mm)
            "transparent_pcb_b64": str,  # 透明PNG (base64)
            "rectified_png_b64": str,    # 校正后图片 (base64)
        }
    """
    # 使用新的Pipeline
    pipeline = PCBRecognitionPipeline()
    result = pipeline.run(
        image_bytes,
        frame_width_mm,
        frame_height_mm,
    )

    return result