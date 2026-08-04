# PCB识别流程拆分说明

## 概述

原来的识别流程耦合度较高，现拆分为三个独立模块，便于单独测试、优化和维护。

---

## 模块划分

### 1. PCB轮廓识别 (`pcb_recognition/`) - **已成熟，保持稳定**

**职责**: 从原始图片中提取透明的PCB轮廓

**架构**: 子文件夹结构，每个步骤独立成类

```
battery_designer/pcb_recognition/
├── orientation_detector.py    # Step 1: 方向检测
├── black_frame_detector.py    # Step 2: 黑色方框检测
├── perspective_calibrator.py  # Step 3: 透视校正
├── hsv_pcb_extractor.py       # Step 4: HSV PCB提取
├── paper_model_builder.py     # Step 5: 纸色模型构建
├── groove_validator.py        # Step 6: 凹槽验证
├── transparent_png_generator.py # Step 7: 透明PNG生成
├── pipeline.py                # 流程编排器
└── __init__.py                # 导出API
```

**流程**:
```
原始图片 → 方向检测 → 方框检测 → 透视校正 → HSV提取 → 纸色模型 → 凹槽验证 → 透明PNG
```

**每个步骤的特点**:
- ✅ 独立的类实现
- ✅ 明确的输入输出
- ✅ 可独立测试
- ✅ 便于问题定位

**API**:
```python
from battery_designer import detect_pcb_outline

result = detect_pcb_outline(
    image_bytes=image_bytes,
    frame_width_mm=85.0,
    frame_height_mm=60.0,
    enable_groove_detection=True,  # 可选：是否启用凹槽检测
)

# 返回:
# - calibration_id: 校准ID
# - outline: PCB轮廓顶点 (mm)
# - grooves: 边缘凹槽
# - transparent_pcb_b64: 透明PNG (base64)
# - steps: 各步骤详细结果 (便于调试)
```

**状态**: ✅ 已充分测试，不再修改

**重要变更**:
- ❌ **移除了孔洞检测** - 孔洞/槽形焊盘应该在焊盘识别阶段处理

---

### 2. 焊盘识别 (`pad_detection.py`) - **重点优化模块**

**职责**: 从透明PCB图像中识别焊盘/触点

**流程**:
```
透明PCB → VLM候选检测 → 几何精修 → 类型分类 → 尺寸测量 → 焊盘列表
```

**API**:
```python
from battery_designer import detect_pads

result = detect_pads(
    transparent_pcb_b64=transparent_pcb_b64,
    outline_points_mm=outline_points,
    side="front",
    pixels_per_mm=10.0,
    refine_iterations=3,
)

# 返回:
# - pads: 焊盘列表
#   - label: 标签 (P+/P-/B-/C+/...)
#   - type: 类型 (rect/round/oval/slot)
#   - center: 中心坐标 (mm)
#   - bbox: 边界框 (mm)
#   - radius_mm: 圆形焊盘半径
#   - corner_radius_mm: 矩形焊盘圆角
```

**状态**: ⚠️ 需要重点优化

**优化方向**:
1. **焊盘候选检测精度** - 减少漏检和误检
2. **焊盘类型分类准确性** - 区分矩形/圆形/椭圆/槽形
3. **焊盘位置精修算法** - 提高定位精度
4. **焊盘尺寸测量** - 减小测量误差
5. **焊盘标签识别** - 提高OCR准确率

---

### 3. 元器件识别 (`component_detection.py`) - **独立模块**

**职责**: 从PCB图像中识别元器件（IC、电阻、电容等）

**流程**:
```
透明PCB + 焊盘位置 → VLM元器件检测 → 引脚映射 → 元器件列表
```

**API**:
```python
from battery_designer import detect_components_on_pcb

result = detect_components_on_pcb(
    transparent_pcb_b64=transparent_pcb_b64,
    pads=pads,
    side="front",
    pixels_per_mm=10.0,
)

# 返回:
# - components: 元器件列表
#   - type: 类型 (IC/Resistor/Capacitor/...)
#   - model: 型号
#   - position: 位置 (mm)
#   - rotation_deg: 旋转角度
#   - pins: 引脚映射
#   - footprint: 封装
```

**状态**: ✅ 独立模块，后续可优化

---

## 使用示例

### 完整识别流程

```python
from battery_designer import (
    detect_pcb_outline,
    detect_pads,
    detect_components_on_pcb,
)

# ── Step 1: PCB轮廓识别 (已成熟) ──
pcb_result = detect_pcb_outline(
    image_bytes=image_bytes,
    frame_width_mm=85.0,
    frame_height_mm=60.0,
)

# ── Step 2: 焊盘识别 (重点优化) ──
pad_result = detect_pads(
    transparent_pcb_b64=pcb_result["transparent_pcb_b64"],
    outline_points_mm=pcb_result["outline"],
    side="front",
    pixels_per_mm=pcb_result["pixels_per_mm"],
)

# ── Step 3: 元器件识别 ──
comp_result = detect_components_on_pcb(
    transparent_pcb_b64=pcb_result["transparent_pcb_b64"],
    pads=pad_result["pads"],
    side="front",
    pixels_per_mm=pcb_result["pixels_per_mm"],
)

# 输出结果
print(f"PCB尺寸: {pcb_result['width_mm']}×{pcb_result['height_mm']} mm")
print(f"焊盘数量: {pad_result['pad_count']}")
print(f"元器件数量: {comp_result['component_count']}")
```

---

## 模块依赖关系

```
pcb_contour.py (已成熟)
    ↓ 提供 transparent_pcb_b64
pad_detection.py (重点优化) ← outline_points_mm
    ↓ 提供 pads
component_detection.py
```

---

## 已弃用功能

### 孔槽识别 (暂时移除)

**原因**: 孔槽本质上是槽形焊盘的一种，可以归类到焊盘识别模块中统一处理。

**后续优化**: 在焊盘识别模块中增加槽形焊盘的专项优化。

---

## 测试建议

### 单元测试

```python
# 测试PCB轮廓识别 (成熟，快速回归)
def test_pcb_contour():
    result = detect_pcb_outline(test_image_bytes, 85.0, 60.0)
    assert result["calibration_id"] is not None
    assert len(result["outline"]) > 0

# 测试焊盘识别 (重点优化，需要详尽测试)
def test_pad_detection():
    result = detect_pads(test_transparent_b64, test_outline, "front", 10.0)
    assert result["pad_count"] >= 0
    
    # 检查焊盘类型分类
    for pad in result["pads"]:
        assert pad["type"] in ["rect", "round", "oval", "slot"]
        
    # 检查焊盘尺寸合理性
    for pad in result["pads"]:
        bbox = pad["bbox"]
        assert 0.5 < bbox["w_mm"] < 10.0  # 焊盘宽度合理范围
        assert 0.5 < bbox["h_mm"] < 10.0  # 焊盘高度合理范围

# 测试元器件识别
def test_component_detection():
    result = detect_components_on_pcb(test_transparent_b64, test_pads, "front", 10.0)
    assert result["component_count"] >= 0
```

---

## 后续优化计划

### 优先级 1: 焊盘识别优化

1. **焊盘候选检测**
   - 提高VLM识别准确率
   - 增加传统CV辅助检测（颜色/形状）
   - 多尺度检测（不同尺寸焊盘）

2. **焊盘类型分类**
   - 增加训练样本
   - 实现本地分类器（降低VLM调用成本）
   - 基于几何特征的快速分类

3. **焊盘位置精修**
   - 实现亚像素级定位
   - 基于轮廓的精修算法
   - 多次迭代收敛

### 优先级 2: 元器件识别增强

1. IC型号识别（本地OCR + 数据库）
2. 元器件方向检测
3. 元器件封装识别

### 优先级 3: 性能优化

1. 减少VLM调用次数
2. 实现缓存机制
3. 并行处理多个焊盘

---

## 文件清单

```
battery_designer/
├── pcb_contour.py          # PCB轮廓识别 (已成熟)
├── pad_detection.py        # 焊盘识别 (重点优化)
├── component_detection.py  # 元器件识别
├── vision.py               # 视觉工具函数 (保留)
├── vlm_detection.py        # VLM调用工具 (保留)
└── __init__.py             # 导出API
```

---

## 注意事项

1. **PCB轮廓识别模块已稳定，请勿修改** - 如有问题请报告
2. **焊盘识别是核心模块，任何修改需要充分测试**
3. **三个模块可以独立测试，降低回归风险**
4. **保留原有的vision.py和vlm_detection.py作为工具模块**

---

## 版本历史

- **v0.2.0** (2026-08-03): 流程拆分，焊盘识别独立优化
- **v0.1.0**: 初始版本，单流程实现