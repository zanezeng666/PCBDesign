"""PCB轮廓识别流程编排

功能：将各个步骤类串联成完整的识别流程。

流程：
  1. 方向检测+强制横屏 (OrientationDetector)
  2. 黑色方框检测 (BlackFrameDetector) → 提供标定信息
  3. 透视校正 (PerspectiveCalibrator) → 提供校正后图片
  4. PCB轮廓提取 (HSVPCBExtractor) - 绿色环边界检测：
     4.1 HSV色彩空间检测绿色像素 (H∈[35,100], S>25, V>15)
     4.2 取最大连通域（排除噪声）
     4.3 形态学闭运算填补缝隙，形成完整闭环
     4.4 填充外轮廓内部 → PCB mask
     4.5 形态学清理 + 最大连通域 + 轮廓光滑
  5. 纸色模型构建 (PaperModelBuilder)
  6. 透明PNG生成 (TransparentPNGGenerator)

核心思路：PCB边缘一定是绿色阻焊层，形成连续的绿色环（覆盖率97-99.7%），
         直接通过绿色环即可确定PCB边界，无需复杂的纸张/阴影分离。

注：凹槽检测已移至 vision.py 中通过纸色验证+mask回写实现，
    确保透明PNG保留PCB的凹槽/缺口几何信息。
"""

from __future__ import annotations

import base64
import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from .orientation_detector import OrientationDetector
from .black_frame_detector import BlackFrameDetector
from .perspective_calibrator import PerspectiveCalibrator
from .hsv_pcb_extractor import HSVPCBExtractor
from .paper_model_builder import PaperModelBuilder
from .transparent_png_generator import TransparentPNGGenerator

if TYPE_CHECKING:
    from .cross_validator import CrossValidator

from ..logger import get_logger

_log = get_logger(__name__)


class PCBRecognitionPipeline:
    """PCB轮廓识别流程编排器

    串联各个步骤，提供完整的识别流程。
    """

    def __init__(self):
        """初始化流程编排器"""
        # 初始化各个步骤
        self.orientation_detector = OrientationDetector()
        self.frame_detector = BlackFrameDetector()
        self.calibrator = PerspectiveCalibrator()
        self.pcb_extractor = HSVPCBExtractor()
        self.paper_builder = PaperModelBuilder()
        self.png_generator = TransparentPNGGenerator()

    def run(
        self,
        image_bytes: bytes,
        frame_width_mm: float,
        frame_height_mm: float,
        debug_dir=None,
    ) -> dict[str, Any]:
        """执行完整的PCB轮廓识别流程

        Args:
            image_bytes: 原始图片字节
            frame_width_mm: 黑色方框宽度 (mm)
            frame_height_mm: 黑色方框高度 (mm)

            Returns:
            {
                "calibration_id": str,       # 校准ID
                "pixels_per_mm": float,      # 像素密度
                "outline": [dict],           # PCB轮廓顶点 (mm)
                "grooves": [dict],           # 凹槽列表 (空列表，保留兼容)
                "transparent_pcb_b64": str,  # 透明PNG (base64)
                "rectified_png_b64": str,    # 校正后图片 (base64)
                "steps": {                   # 各步骤结果
                    "orientation_detection": dict,
                    "frame_detection": dict,
                    "calibration": dict,
                    "pcb_extraction": dict,
                    "paper_model": dict,
                    "png_generation": dict,
                }
            }
        """
        _log.info("=" * 60)
        _log.info("PCB轮廓识别流程: 开始")
        _log.info("=" * 60)

        steps_result = {}

        # ── Step 1: 方向检测 + 强制横屏 ──
        _log.info("Step 1/6: 方向检测")
        try:
            # 从字节加载图片
            from io import BytesIO
            from PIL import Image

            img = Image.open(BytesIO(image_bytes))

            # 检测方向
            orientation_result = self.orientation_detector.detect_orientation(img)
            steps_result["orientation_detection"] = orientation_result

            _log.info(
                "[OK] 方向检测: orientation=%d°, method=%s, confidence=%.2f",
                orientation_result["orientation"],
                orientation_result["method"],
                orientation_result["confidence"],
            )

            # 如果需要旋转，执行旋转
            if orientation_result["needs_rotation"]:
                rotated_img = self.orientation_detector.rotate_to_landscape(
                    img,
                    orientation_result["orientation"],
                )

                # 确保是 PIL.Image 类型
                if isinstance(rotated_img, np.ndarray):
                    from PIL import Image as PILImage
                    rotated_img = PILImage.fromarray(rotated_img[:, :, [2, 1, 0]])  # BGR → RGB

                # 转换回字节
                output_buffer = BytesIO()
                rotated_img.save(output_buffer, format='PNG')
                image_bytes = output_buffer.getvalue()

                _log.info("[OK] 强制横屏: 旋转 %d°", orientation_result["orientation"])

        except Exception as e:
            _log.warning("[WARN] 方向检测失败: %s (继续流程)", e)
            steps_result["orientation_detection"] = {"error": str(e)}

        # ── Step 2: 黑色方框检测 ──
        _log.info("Step 2/6: 黑色方框检测")
        try:
            result_frame = self.frame_detector.detect(
                image_bytes,
                frame_width_mm,
                frame_height_mm,
            )
            steps_result["frame_detection"] = result_frame
            corners = result_frame["corners"]
            pixels_per_mm = result_frame["pixels_per_mm"]
            frame_border_mm = result_frame.get("frame_border_mm", 1.5)
            _log.info("[OK] 方框检测: 密度=%.2f px/mm, 线宽=%.2fmm",
                      pixels_per_mm, frame_border_mm)

        except Exception as e:
            _log.error("[FAIL] 方框检测失败: %s", e)
            raise

        # ── Step 3: 透视校正 ──
        _log.info("Step 3/6: 透视校正")
        try:
            result_calib = self.calibrator.calibrate(
                image_bytes,
                corners,
                pixels_per_mm,
                frame_width_mm,
                frame_height_mm,
            )
            steps_result["calibration"] = result_calib
            calibration_id = result_calib["calibration_id"]
            rectified_bytes = result_calib["rectified_bytes"]
            _log.info("[OK] 透视校正: ID=%s", calibration_id)

        except Exception as e:
            _log.error("[FAIL] 透视校正失败: %s", e)
            raise

        # ── Step 4: PCB提取（绿色环边界检测） ──
        _log.info("Step 4/6: PCB提取（绿色环边界检测）")
        try:
            result_extract = self.pcb_extractor.extract(
                rectified_bytes,
                pixels_per_mm=pixels_per_mm,
                frame_border_mm=frame_border_mm,
                debug_dir=debug_dir,
            )
            steps_result["pcb_extraction"] = result_extract
            pcb_contour = result_extract["pcb_contour"]
            _log.info("[OK] PCB提取: 面积占比=%.1f%%", result_extract["pcb_area_ratio"] * 100)

        except Exception as e:
            _log.error("[FAIL] PCB提取失败: %s", e)
            raise

        # ── Step 5: 纸色模型构建 ──
        _log.info("Step 5/6: 纸色模型构建")
        try:
            result_paper = self.paper_builder.build(
                rectified_bytes,
                pcb_contour,
                pixels_per_mm,
            )
            steps_result["paper_model"] = result_paper
            refined_contour = result_paper["refined_contour"]
            _log.info("[OK] 纸色模型: 类型=%s", result_paper["paper_model"]["type"])

        except Exception as e:
            _log.error("[FAIL] 纸色模型构建失败: %s", e)
            raise

        # ── Step 6: 透明PNG生成 ──
        _log.info("Step 6/6: 透明PNG生成")
        try:
            result_png = self.png_generator.generate(
                rectified_bytes,
                refined_contour,
                calibration_id,
            )
            steps_result["png_generation"] = result_png
            transparent_b64 = result_png["transparent_b64"]
            _log.info("[OK] 透明PNG生成: 完成")

        except Exception as e:
            _log.error("[FAIL] 透明PNG生成失败: %s", e)
            raise

        # ── Step 7: 构建最终结果 ──
        # 转换轮廓为mm坐标
        outline_mm = self._contour_to_mm(refined_contour, pixels_per_mm)

        # 编码校正后图片
        rectified_b64 = base64.b64encode(rectified_bytes).decode('utf-8')

        _log.info("=" * 60)
        _log.info("PCB轮廓识别流程: 完成 (ID=%s)", calibration_id)
        _log.info("=" * 60)

        return {
            "calibration_id": calibration_id,
            "pixels_per_mm": pixels_per_mm,
            "pcb_color": steps_result["pcb_extraction"]["color_name"],
            "paper_color": steps_result["paper_model"]["paper_color"],
            "width_mm": frame_width_mm,
            "height_mm": frame_height_mm,
            "outline": outline_mm,
            "grooves": [],  # 凹槽检测已移至vision.py，这里保留空列表以兼容
            "transparent_pcb_b64": transparent_b64,
            "rectified_png_b64": rectified_b64,
            "steps": steps_result,
        }

    def _contour_to_mm(
        self,
        contour: np.ndarray,
        pixels_per_mm: float,
    ) -> list[dict[str, float]]:
        """将轮廓坐标从pixel转换为mm

        Args:
            contour: 轮廓点集 (numpy数组)
            pixels_per_mm: 像素密度

        Returns:
            轮廓顶点列表 [{"x_mm": ..., "y_mm": ...}, ...]
        """
        outline_mm = []

        for point in contour:
            if hasattr(point, "__len__") and len(point) >= 1 and hasattr(point[0], "__len__"):
                # OpenCV contour format: (N, 1, 2) → point shape is (1, 2)
                x_px, y_px = point[0]
            else:
                # Flat format: (N, 2)
                x_px, y_px = point[0], point[1]
            x_mm = round(x_px / pixels_per_mm, 3)
            y_mm = round(y_px / pixels_per_mm, 3)
            outline_mm.append({"x_mm": x_mm, "y_mm": y_mm})

        return outline_mm

    @staticmethod
    def cross_validate_front_back(
        front_result: dict[str, Any],
        back_result: dict[str, Any],
        width_mm: float | None = None,
        height_mm: float | None = None,
    ) -> dict[str, Any]:
        """正反面交叉校验（Step 7）

        用正反面轮廓交集消除单面检测的毛刺/阴影。

        Args:
            front_result: 正面 pipeline.run() 结果
            back_result: 背面 pipeline.run() 结果
            width_mm: PCB 宽度 (mm)，若未提供则从 result 提取或使用默认值
            height_mm: PCB 高度 (mm)，若未提供则从 result 提取或使用默认值

        Returns:
            {
                "outline": 共识轮廓 (mm),
                "front_area_mm2": 正面面积 (mm²),
                "back_area_mm2": 背面面积 (mm²),
                "consensus_area_mm2": 共识面积 (mm²),
                "diff_image_b64": diff 可视化图 (base64),
                "transparent_pcb_b64": 从共识轮廓生成的透明 PNG,
            }
        """
        # 延迟导入避免循环依赖
        from .cross_validator import CrossValidator
        
        return CrossValidator.validate(front_result, back_result, width_mm, height_mm)


# ── 测试代码 ──
if __name__ == "__main__":
    import sys
    from pathlib import Path

    if len(sys.argv) < 2:
        print("使用方法: python -m battery_designer.pcb_recognition.pipeline <image_path>")
        sys.exit(1)

    image_path = Path(sys.argv[1])
    if not image_path.exists():
        print(f"错误: 图片不存在 {image_path}")
        sys.exit(1)

    pipeline = PCBRecognitionPipeline()
    print("PCB轮廓识别流程初始化完成\n")

    # 读取图片字节
    image_bytes = image_path.read_bytes()

    # 执行完整流程（使用默认的框架尺寸）
    result = pipeline.run(
        image_bytes=image_bytes,
        frame_width_mm=100.0,  # 默认框架宽度
        frame_height_mm=100.0,  # 默认框架高度
    )

    # 输出结果摘要
    print("\n" + "="*60)
    print("识别结果摘要:")
    print("="*60)
    print(f"图片路径: {image_path}")
    print(f"总耗时: {result['total_time_seconds']:.2f}s")

    # Step 1: EXIF 修正
    exif = result['steps_result'].get('exif_correction', {})
    print(f"\n[Step 1] EXIF 修正: corrected={exif.get('corrected', False)}")

    # Step 2: 方向检测
    orient = result['steps_result'].get('orientation_detection', {})
    if orient.get('skipped'):
        print(f"[Step 2] 方向检测: 跳过（EXIF修正后已是横屏）")
    else:
        print(f"[Step 2] 方向检测: method={orient.get('method')}, rotation={orient.get('orientation')}°, confidence={orient.get('confidence'):.2f}")

    print("="*60)