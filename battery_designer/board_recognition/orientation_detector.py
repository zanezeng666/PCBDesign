"""图片方向检测器

自动判断 PCB 图片的正确方向，确保文字正向显示。

方法：
1. OCR 文字方向检测：检测 PCB 上的文字，判断是否正向
2. IC 芯片方向检测：检测 IC 的缺口标记（备选）
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageOps
from PIL.Image import Transpose

from ..core.logger import get_logger

_log = get_logger(__name__)


class OrientationDetector:
    """图片方向检测器

    自动判断 PCB 图片的正确方向。
    """

    def __init__(self):
        """初始化方向检测器"""
        self.ocr_available = self._check_ocr()
        self.vlm_available = self._check_vlm()

    def _check_ocr(self) -> bool:
        """检查 OCR 库是否可用"""
        try:
            import pytesseract
            return True
        except ImportError:
            _log.info("pytesseract 未安装，将尝试 VLM 方法")
            return False

    def _check_vlm(self) -> bool:
        """检查 VLM API 是否可用"""
        try:
            from ..core.vlm_client import get_api_key as _get_api_key
            return bool(_get_api_key())
        except Exception:
            return False

    def detect_orientation(self, image: Image.Image | np.ndarray) -> dict[str, Any]:
        """检测图片的正确方向

        Args:
            image: PIL Image 或 numpy 数组

        Returns:
            {
                "orientation": int,          # 建议的旋转角度 (0, 90, 180, 270)
                "confidence": float,         # 置信度 (0-1)
                "method": str,               # 检测方法
                "is_landscape": bool,        # 是否为横屏
                "needs_rotation": bool,      # 是否需要旋转
            }
        """
        # 转换为 PIL Image
        if isinstance(image, np.ndarray):
            # numpy: (H, W, C) → PIL: (W, H)
            if len(image.shape) == 3 and image.shape[2] >= 3:
                # BGR → RGB
                if image.shape[2] == 3:
                    image = image[:, :, [2, 1, 0]]
                elif image.shape[2] == 4:
                    image = image[:, :, [2, 1, 0, 3]]
            image = Image.fromarray(image)

        width, height = image.size
        is_landscape = width > height

        _log.info(
            "方向检测: 图片尺寸 %dx%d (%s)",
            width, height,
            "横屏" if is_landscape else "竖屏"
        )

        # 如果已经是横屏，检查是否需要旋转 180°
        if is_landscape:
            return self._check_landscape_orientation(image)
        
        # 如果是竖屏，需要旋转成横屏
        return self._check_portrait_orientation(image)

    def _check_landscape_orientation(self, image: Image.Image) -> dict[str, Any]:
        """检查横屏图片是否需要旋转 180°

        Args:
            image: 横屏图片

        Returns:
            方向检测结果
        """
        # 尝试 VLM 检测（最准确）
        if self.vlm_available:
            return self._detect_by_vlm(image)

        # VLM 不可用，无法判断
        _log.warning("VLM 不可用，无法判断横屏方向，默认不旋转")
        return {
            "orientation": 0,
            "confidence": 0.0,
            "method": "vlm_unavailable",
            "is_landscape": True,
            "needs_rotation": False,
        }

    def _check_portrait_orientation(self, image: Image.Image) -> dict[str, Any]:
        """检查竖屏图片应该顺时针还是逆时针旋转

        Args:
            image: 竖屏图片

        Returns:
            方向检测结果
        """
        # 尝试 VLM 检测（最准确）
        if self.vlm_available:
            return self._detect_by_vlm(image)

        # VLM 不可用，无法判断
        _log.warning("VLM 不可用，无法判断竖屏方向，默认不旋转")
        return {
            "orientation": 0,
            "confidence": 0.0,
            "method": "vlm_unavailable",
            "is_landscape": False,
            "needs_rotation": False,
        }

    def _detect_by_ocr(
        self,
        image: Image.Image,
        orientations: list[int],
    ) -> dict[str, Any]:
        """使用 OCR 检测最佳方向

        Args:
            image: 原始图片
            orientations: 候选旋转角度列表

        Returns:
            方向检测结果
        """
        import pytesseract

        best_orientation = 0
        best_confidence = 0.0
        best_method = "ocr"

        _log.info("使用 OCR 检测方向，候选角度: %s", orientations)

        for angle in orientations:
            # 旋转图片
            if angle == 0:
                rotated = image
            elif angle == 90:
                rotated = image.transpose(Transpose.ROTATE_90)
            elif angle == 180:
                rotated = image.transpose(Transpose.ROTATE_180)
            elif angle == 270:
                rotated = image.transpose(Transpose.ROTATE_270)
            else:
                continue

            # OCR 识别
            try:
                # 获取 OCR 数据（包含置信度）
                ocr_data = pytesseract.image_to_data(
                    rotated,
                    output_type=pytesseract.Output.DICT,
                    config='--psm 6'  # 假设为统一文本块
                )

                # 计算平均置信度
                confidences = [
                    int(conf) for conf in ocr_data['conf']
                    if conf != '-1' and str(conf).isdigit()
                ]

                if confidences:
                    avg_confidence = sum(confidences) / len(confidences) / 100.0
                else:
                    avg_confidence = 0.0

                # 统计识别到的文字数量
                text_count = len([t for t in ocr_data['text'] if t.strip()])

                _log.debug(
                    "角度 %d°: 置信度=%.2f, 文字数=%d",
                    angle, avg_confidence, text_count
                )

                # 选择置信度最高的方向
                if avg_confidence > best_confidence:
                    best_confidence = avg_confidence
                    best_orientation = angle

            except Exception as e:
                _log.warning("OCR 检测失败 (角度=%d): %s", angle, e)
                continue

        width, height = image.size
        is_landscape = width > height

        return {
            "orientation": best_orientation,
            "confidence": best_confidence,
            "method": "ocr",
            "is_landscape": is_landscape,
            "needs_rotation": best_orientation != 0,
        }

    def _detect_by_heuristic(self, image: Image.Image) -> dict[str, Any]:
        """启发式方法检测方向（当 OCR 不可用时）

        策略：
        1. 假设 PCB 拍摄时文字是正向的（最常见）
        2. 顺时针旋转 90° (因为手机竖拍通常需要顺时针旋转)

        Args:
            image: 竖屏图片

        Returns:
            方向检测结果
        """
        _log.info("使用启发式方法：默认顺时针旋转 90°")

        width, height = image.size

        return {
            "orientation": 90,  # 顺时针旋转
            "confidence": 0.7,  # 中等置信度
            "method": "heuristic",
            "is_landscape": False,
            "needs_rotation": True,
        }

    def _detect_by_vlm(self, image: Image.Image) -> dict[str, Any]:
        """使用 VLM 检测图片方向

        Args:
            image: 输入图片

        Returns:
            方向检测结果
        """
        import base64
        import json
        import re
        from io import BytesIO

        from dashscope import MultiModalConversation

        from ..core.vlm_client import get_api_key as _get_api_key, vlm_call as _vlm_call_with_retry

        _log.info("使用 VLM 检测图片方向")

        # 将图片编码为 base64
        buffer = BytesIO()
        image.save(buffer, format='PNG')
        image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        image_url = f"data:image/png;base64,{image_base64}"

        # 构造提示词
        prompt = """请分析这张 PCB 电路板图片的方向。

判断标准：
- PCB 上的文字应该正向显示（可读）
- 如果文字倒置或侧向，需要旋转

请回答：
1. 图片是否正向？（是/否）
2. 如果不是正向，需要顺时针旋转多少度？（0/90/180/270）
3. 置信度（高/中/低）

请用以下 JSON 格式回复：
{
  "is_correct": true/false,
  "rotation_needed": 0/90/180/270,
  "confidence": "high"/"medium"/"low"
}"""

        try:
            # 构造消息
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"image": image_url},
                        {"text": prompt},
                    ],
                }
            ]

            # 调用 VLM API
            response = _vlm_call_with_retry(
                model="qwen3.7-plus",
                messages=messages,
                temperature=0.1,
                max_tokens=200,
                max_retries=2,
            )

            if not response:
                raise RuntimeError("VLM API 调用失败")

            # 提取响应文本
            response_text = response.output.choices[0].message.content[0]["text"]
            _log.debug("VLM 响应: %s", response_text)

            # 解析 JSON 响应
            json_match = re.search(r'\{[^}]+\}', response_text, re.DOTALL)
            if json_match:
                result_json = json.loads(json_match.group())
                rotation = result_json.get("rotation_needed", 0)
                confidence_str = result_json.get("confidence", "medium")

                # 转换置信度
                confidence_map = {"high": 0.9, "medium": 0.7, "low": 0.5}
                confidence = confidence_map.get(confidence_str, 0.7)

                width, height = image.size
                is_landscape = width > height

                _log.info(
                    "VLM 检测结果: rotation=%d°, confidence=%.2f",
                    rotation, confidence
                )

                return {
                    "orientation": rotation,
                    "confidence": confidence,
                    "method": "vlm",
                    "is_landscape": is_landscape,
                    "needs_rotation": rotation != 0,
                }

        except Exception as e:
            _log.warning("VLM 检测失败: %s (默认不处理)", e)

        # VLM 失败，默认不处理
        width, height = image.size
        is_landscape = width > height

        return {
            "orientation": 0,
            "confidence": 0.0,
            "method": "vlm_failed",
            "is_landscape": is_landscape,
            "needs_rotation": False,
        }

    def rotate_to_landscape(
        self,
        image: Image.Image | np.ndarray,
        orientation: int,
    ) -> Image.Image | np.ndarray:
        """将图片旋转到正确的横屏方向

        Args:
            image: 输入图片
            orientation: 旋转角度 (0, 90, 180, 270)

        Returns:
            旋转后的图片
        """
        is_numpy = isinstance(image, np.ndarray)

        # 转换为 PIL Image
        if is_numpy:
            if len(image.shape) == 3 and image.shape[2] >= 3:
                if image.shape[2] == 3:
                    image = image[:, :, [2, 1, 0]]  # BGR → RGB
                elif image.shape[2] == 4:
                    image = image[:, :, [2, 1, 0, 3]]  # BGRA → RGBA
            image = Image.fromarray(image)

        # 旋转
        if orientation == 0:
            rotated = image
        elif orientation == 90:
            rotated = image.transpose(Transpose.ROTATE_90)
        elif orientation == 180:
            rotated = image.transpose(Transpose.ROTATE_180)
        elif orientation == 270:
            rotated = image.transpose(Transpose.ROTATE_270)
        else:
            rotated = image

        _log.info("图片已旋转 %d°，新尺寸: %dx%d",
            orientation, rotated.size[0], rotated.size[1]
        )

        # 转换回 numpy（如果需要）
        if is_numpy:
            rotated_array = np.array(rotated)
            if len(rotated_array.shape) == 3 and rotated_array.shape[2] >= 3:
                rotated_array = rotated_array[:, :, [2, 1, 0]]  # RGB → BGR
            return rotated_array

        return rotated


# ── 测试代码 ──
if __name__ == "__main__":
    import sys
    import logging
    from battery_designer import configure_logging

    configure_logging(level=logging.INFO, console=True)

    detector = OrientationDetector()

    # 测试图片
    test_images = [
        Path("input/22 PCB/front.jpg"),
        Path("input/22 PCB/back.jpg"),
    ]

    for img_path in test_images:
        if not img_path.exists():
            print(f"图片不存在: {img_path}")
            continue

        print(f"\n测试: {img_path}")
        print("-" * 60)

        # 读取图片
        img = Image.open(img_path)
        print(f"原始尺寸: {img.size[0]}x{img.size[1]}")

        # 检测方向
        result = detector.detect_orientation(img)
        print(f"检测结果: {result}")

        # 旋转到正确方向
        if result["needs_rotation"]:
            rotated = detector.rotate_to_landscape(img, result["orientation"])
            print(f"旋转后尺寸: {rotated.size[0]}x{rotated.size[1]}")

            # 保存结果
            output_path = Path("output") / f"rotated_{img_path.name}"
            output_path.parent.mkdir(exist_ok=True)
            rotated.save(output_path)
            print(f"已保存到: {output_path}")