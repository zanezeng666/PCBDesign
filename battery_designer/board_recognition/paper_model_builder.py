"""纸色模型构建

功能：基于PCB轮廓构建纸色模型，用于验证和修正轮廓。

输入：PCB轮廓多边形 + 原图
输出：优化后的PCB轮廓

核心思想：
    PCB轮廓遵循严格的几何规律：
    1. 主体是矩形基板
    2. 槽/切口都是规则的矩形
    3. 内角可以是直角或圆角
    4. 边缘必须是直线
    
    轮廓 = 矩形基板 - 矩形槽/切口
"""

from __future__ import annotations

import logging

import cv2
import numpy as np

from ..core.logger import get_logger
from .pcb_rectangular_contour_refiner import PCBRectangularContourRefiner

_log = get_logger(__name__)


class PaperModelBuilder:
    """纸色模型构建器

    基于PCB几何规律性（矩形基板 - 矩形槽）优化轮廓。
    """

    def __init__(
        self,
        use_rectangular_refinement: bool = True,
        groove_depth_ratio: float = 0.01,  # 降低阈值，更容易检测到凹槽
    ):
        """初始化构建器

        Args:
            use_rectangular_refinement: 是否使用矩形轮廓精修
            groove_depth_ratio: 凹槽深度阈值（相对于最小边长）
        """
        self.use_rectangular_refinement = use_rectangular_refinement
        
        # 初始化矩形轮廓精修器
        if use_rectangular_refinement:
            self.rectangular_refiner = PCBRectangularContourRefiner(
                groove_depth_ratio=groove_depth_ratio,
            )
        else:
            self.rectangular_refiner = None

    def build(
        self,
        image_bytes: bytes,
        pcb_contour: np.ndarray,
        pixels_per_mm: float,
    ) -> dict:
        """构建纸色模型并优化轮廓

        Args:
            image_bytes: 校正后图片字节
            pcb_contour: HSV提取的PCB轮廓
            pixels_per_mm: 像素密度

        Returns:
            {
                "refined_contour": np.ndarray,  # 优化后的轮廓
                "paper_model": dict,            # 纸色模型参数
                "paper_color": str,             # 纸张颜色
                "is_rectangular": bool,         # 是否为矩形
            }
        """
        _log.info("纸色模型构建: 开始")

        # ── Step 1: 解码图片 ──
        try:
            nparr = np.frombuffer(image_bytes, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            if img is None:
                raise ValueError("Failed to decode image")

            h, w = img.shape[:2]

        except Exception as e:
            _log.error("纸色模型构建: 图片解码失败 - %s", e)
            raise ValueError(f"Failed to decode image: {e}")

        # ── Step 2: 提取纸张颜色 ──
        paper_color = self._detect_paper_color(img, pcb_contour)

        # ── Step 3: 直接使用矩形精修器 ──
        # 不再进行二次轮廓近似，直接让矩形精修器处理
        if self.use_rectangular_refinement and self.rectangular_refiner:
            _log.info("使用矩形轮廓精修器...")
            
            # 从轮廓生成mask
            pcb_mask = np.zeros((h, w), dtype=np.uint8)
            cv2.drawContours(pcb_mask, [pcb_contour], -1, 255, -1)
            
            # 调用精修器
            refined_mask, refined_contour = self.rectangular_refiner.refine(pcb_mask)
            
            # 检查精修结果
            if len(refined_contour) <= 4:
                # 无凹槽：用approxPolyDP简化为完美矩形4角点
                peri = cv2.arcLength(refined_contour, True)
                approx = cv2.approxPolyDP(refined_contour, 0.02 * peri, True)
                
                if len(approx) == 4:
                    refined_contour = approx.reshape(-1, 2)
                    paper_model = {
                        "type": "rectangular",
                        "vertex_count": 4,
                    }
                else:
                    paper_model = {
                        "type": "rectangular_refined",
                        "vertex_count": len(refined_contour),
                    }
            else:
                # 有凹槽：直接使用精修器的正交多边形，不做approxPolyDP简化
                # （approxPolyDP的epsilon=0.02*peri约94px，会把17-22px的浅凹槽当噪声平滑掉）
                paper_model = {
                    "type": "rectangular_refined",
                    "vertex_count": len(refined_contour),
                }
        else:
            # 不使用精修器，保持原轮廓
            refined_contour = pcb_contour
            
            paper_model = {
                "type": "irregular",
                "vertex_count": len(pcb_contour),
            }

        _log.info(
            "纸色模型构建: 完成 (类型=%s, 纸色=%s, 顶点数=%d)",
            paper_model["type"],
            paper_color,
            len(refined_contour),
        )

        return {
            "refined_contour": refined_contour,
            "paper_model": paper_model,
            "paper_color": paper_color,
            "is_rectangular": paper_model["type"] == "rectangular",
        }

    def _detect_paper_color(self, img: np.ndarray, pcb_contour: np.ndarray) -> str:
        """检测纸张颜色

        在黑色方框内、PCB外部采样，判断纸张颜色类型。

        Args:
            img: 图片 (BGR)
            pcb_contour: PCB轮廓

        Returns:
            纸张颜色: "white" / "cream" / "gray"
        """
        h, w = img.shape[:2]

        # 创建全图mask
        mask = np.ones((h, w), dtype=np.uint8) * 255

        # 排除PCB区域
        cv2.drawContours(mask, [pcb_contour], -1, 0, -1)

        # 排除边缘（黑色方框）
        margin = int(min(w, h) * 0.05)
        mask[:margin, :] = 0  # 上边
        mask[-margin:, :] = 0  # 下边
        mask[:, :margin] = 0  # 左边
        mask[:, -margin:] = 0  # 右边

        # 计算纸张区域的平均颜色
        mean_color = cv2.mean(img, mask)[:3]  # BGR
        b, g, r = mean_color

        # 转换为亮度
        brightness = (b + g + r) / 3

        # 判断颜色类型
        if brightness > 200:
            paper_color = "white"
        elif brightness > 150:
            paper_color = "cream"
        else:
            paper_color = "gray"

        _log.debug(
            "纸张颜色检测: RGB=(%.0f, %.0f, %.0f), 亮度=%.0f, 结果=%s",
            r, g, b, brightness, paper_color
        )

        return paper_color

    def _is_approx_rectangular(self, contour: np.ndarray, threshold: float = 0.1) -> bool:
        """判断轮廓是否近似为矩形

        Args:
            contour: 轮廓点集
            threshold: 凸度阈值

        Returns:
            是否近似为矩形
        """
        # 计算凸包
        hull = cv2.convexHull(contour)

        # 计算轮廓面积与凸包面积的比值
        contour_area = cv2.contourArea(contour)
        hull_area = cv2.contourArea(hull)

        if hull_area == 0:
            return False

        ratio = contour_area / hull_area

        return ratio > (1 - threshold)


# ── 测试代码 ──
if __name__ == "__main__":
    builder = PaperModelBuilder()
    print("纸色模型构建器初始化完成")