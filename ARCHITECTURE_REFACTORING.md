# PCB轮廓识别架构重构总结

## 重构目标

按照用户建议，将原来的简单包装模式重构为清晰的模块化架构：
- ✅ 创建子文件夹 `battery_designer/pcb_recognition/`
- ✅ 每个流程步骤独立成类
- ✅ 明确的输入输出接口
- ✅ 便于独立测试和问题定位

---

## 新架构设计

### 文件结构

```
battery_designer/pcb_recognition/
├── __init__.py                   # 模块初始化
├── orientation_detector.py       # Step 1: 方向检测
├── black_frame_detector.py       # Step 2: 黑色方框检测
├── perspective_calibrator.py     # Step 3: 透视校正
├── hsv_pcb_extractor.py          # Step 4: HSV PCB提取
├── paper_model_builder.py        # Step 5: 纸色模型构建
├── groove_validator.py           # Step 6: 凹槽验证
├── transparent_png_generator.py  # Step 7: 透明PNG生成
└── pipeline.py                   # 流程编排器
```

---

## 流程步骤详解

### Step 1: OrientationDetector (方向检测)

**职责**: 检测图片方向并旋转至横屏正方向

**输入**: `image: PIL.Image` (原始图片)

**输出**: 
```python
{
    "orientation": int,         # 检测到的方向 (0/90/180/270)
    "method": str,              # 检测方法 ("vlm"/"heuristic")
    "confidence": float,        # 置信度 (0.0-1.0)
    "needs_rotation": bool,     # 是否需要旋转
}
```

**新增功能**: 使用 VLM 智能检测图片方向

---

### Step 2: BlackFrameDetector (黑色方框检测)

**职责**: 检测图片中的黑色校准方框

**输入**: 
- `image_bytes: bytes` (图片)
- `frame_width_mm: float` (方框宽度 mm)
- `frame_height_mm: float` (方框高度 mm)

**输出**:
```python
{
    "corners": [(x1,y1), (x2,y2), (x3,y3), (x4,y4)],  # 方框四角 (pixel)
    "pixels_per_mm": float,  # 像素密度
    "frame_width_px": int,   # 方框宽度 (pixel)
    "frame_height_px": int,  # 方框高度 (pixel)
    "detection_id": str,     # 检测ID
    "debug_image_b64": str,  # 调试图片
}
```

---

### Step 3: PerspectiveCalibrator (透视校正)

**职责**: 基于方框四角进行透视变换，校正为正视图

**输入**:
- `image_bytes: bytes`
- `corners: list` (方框四角)
- `pixels_per_mm: float`
- `frame_width_mm: float`
- `frame_height_mm: float`

**输出**:
```python
{
    "calibration_id": str,        # 校准ID
    "rectified_bytes": bytes,     # 校正后图片
    "rectified_width_px": int,    # 校正后宽度
    "rectified_height_px": int,   # 校正后高度
    "pixels_per_mm": float,       # 像素密度（验证）
    "calibration_dir": str,       # 校准目录路径
}
```

---

### Step 4: HSVPCBExtractor (HSV PCB提取)

**职责**: 使用HSV颜色空间提取PCB区域

**输入**: `image_bytes: bytes` (校正后图片)

**输出**:
```python
{
    "pcb_mask": np.ndarray,       # PCB掩码
    "pcb_contour": np.ndarray,    # PCB轮廓
    "color_name": str,            # 检测到的PCB颜色 (green/blue/dark)
    "pcb_area_ratio": float,      # PCB面积占比
}
```

**支持颜色**: 绿色PCB、蓝色PCB、黑色/深色PCB

---

### Step 5: PaperModelBuilder (纸色模型构建)

**职责**: 基于物理约束优化PCB轮廓

**输入**:
- `image_bytes: bytes`
- `pcb_contour: np.ndarray`
- `pixels_per_mm: float`

**输出**:
```python
{
    "refined_contour": np.ndarray,  # 优化后的轮廓
    "paper_model": dict,            # 纸色模型参数
    "is_rectangular": bool,         # 是否为矩形
}
```

**优化策略**:
- 矩形PCB: 使用最小外接矩形
- 异形PCB: 保持原轮廓

---

### Step 6: GrooveValidator (凹槽验证)

**职责**: 检测并验证PCB轮廓边缘的凹槽特征

**输入**:
- `image_bytes: bytes`
- `pcb_contour: np.ndarray`
- `pixels_per_mm: float`

**输出**:
```python
{
    "grooves": [
        {
            "position": "top"|"bottom"|"left"|"right",
            "depth_mm": float,
            "width_mm": float,
            "start_idx": int,
            "end_idx": int,
        },
        ...
    ],
    "groove_count": int,
}
```

**过滤条件**:
- 最小凹槽深度: 2.0 mm
- 最小凹槽宽度: 3.0 mm

---

### Step 7: TransparentPNGGenerator (透明PNG生成)

**职责**: 生成背景透明的PCB图片

**输入**:
- `image_bytes: bytes`
- `pcb_contour: np.ndarray`
- `calibration_id: str`

**输出**:
```python
{
    "transparent_bytes": bytes,  # 透明PNG字节
    "transparent_b64": str,      # 透明PNG (base64)
    "bbox": (x, y, w, h),        # PCB边界框
    "save_path": str,            # 保存路径
}
```

---

### Pipeline (流程编排器)

**职责**: 串联所有步骤，提供完整识别流程

**API**:
```python
from battery_designer import PCBRecognitionPipeline

pipeline = PCBRecognitionPipeline()
result = pipeline.run(
    image_bytes=image_bytes,
    frame_width_mm=85.0,
    frame_height_mm=60.0,
    enable_groove_detection=True,  # 可选
)

# 返回:
{
    "calibration_id": str,
    "pixels_per_mm": float,
    "outline": [{"x_mm": ..., "y_mm": ...}, ...],
    "grooves": [...],
    "transparent_pcb_b64": str,
    "rectified_png_b64": str,
    "steps": {  # 各步骤详细结果
        "orientation_detection": {...},
        "frame_detection": {...},
        "calibration": {...},
        "pcb_extraction": {...},
        "paper_model": {...},
        "groove_validation": {...},
        "png_generation": {...},
    }
}
```

---

## 重要变更

### ❌ 移除的功能

**孔洞检测** - 原在PCB轮廓识别阶段，现已移除

**原因**:
- 孔洞/槽形焊盘本质上是焊盘的一种，应该在焊盘识别阶段处理
- 避免流程混乱，保持PCB轮廓识别的职责单一

---

## 使用示例

### 方式1: 使用高层API（推荐）

```python
from battery_designer import detect_pcb_outline

result = detect_pcb_outline(
    image_bytes=image_bytes,
    frame_width_mm=85.0,
    frame_height_mm=60.0,
)
```

### 方式2: 使用Pipeline类

```python
from battery_designer import PCBRecognitionPipeline

pipeline = PCBRecognitionPipeline()
result = pipeline.run(
    image_bytes=image_bytes,
    frame_width_mm=85.0,
    frame_height_mm=60.0,
    enable_groove_detection=True,
)
```

### 方式3: 单独测试某个步骤

```python
from battery_designer.pcb_recognition import BlackFrameDetector

detector = BlackFrameDetector()
result = detector.detect(
    image_bytes,
    frame_width_mm=85.0,
    frame_height_mm=60.0,
)
```

---

## 优势对比

### 旧架构（简单包装）

```python
# pcb_contour.py
from .vision import calibrate_black_frame, extract_pcb

def detect_pcb_outline(...):
    cal_result = calibrate_black_frame(...)  # 黑盒调用
    pcb_result = extract_pcb(...)             # 黑盒调用
    return {...}
```

**问题**:
- ❌ 流程不清晰，只是简单包装
- ❌ 无法单独测试某个步骤
- ❌ 问题定位困难
- ❌ 核心逻辑分散在vision.py中

### 新架构（模块化）

```python
# pipeline.py
class PCBRecognitionPipeline:
    def run(self, ...):
        # Step 1
        result_orient = self.orientation_detector.detect(...)
        
        # Step 2
        result_frame = self.frame_detector.detect(...)
        
        # Step 3-7
        ...
```

**优势**:
- ✅ 每个步骤独立成类，职责清晰
- ✅ 明确的输入输出接口
- ✅ 可独立测试每个步骤
- ✅ 便于问题定位和调试
- ✅ 核心逻辑直接实现在类中，不依赖外部函数

---

## 测试结果

**测试通过率**: 4/4 (100%)

- ✅ 步骤类导入和初始化
- ✅ Pipeline API
- ✅ 文件结构
- ✅ API 兼容性

---

## 后续优化方向

### 优先级 1: 算法优化

1. **BlackFrameDetector** - 提高方框检测鲁棒性
2. **HSVPCBExtractor** - 扩展更多PCB颜色类型
3. **GrooveValidator** - 优化凹槽检测算法

### 优先级 2: 性能优化

1. 减少图片编解码次数
2. 使用内存缓存（避免频繁读写文件）
3. 并行化处理（多步骤并行）

### 优先级 3: 测试完善

1. 为每个步骤类编写单元测试
2. 增加边界情况测试
3. 性能基准测试

---

## 总结

本次重构完全按照用户建议实施，成功将PCB轮廓识别流程拆分为7个独立的步骤类，每个类都有明确的输入输出接口。架构清晰，便于测试和维护。

**核心改进**:
- ✅ 移除了不属于PCB轮廓识别阶段的孔洞检测
- ✅ 使用 VLM 智能检测图片方向，确保横屏正方向
- ✅ 每个步骤可独立测试，便于问题定位
- ✅ 核心逻辑直接实现在类中，不再依赖外部包装

**测试结果**: 4/4 全部通过 ✅

**建议**: 保持PCB轮廓识别模块稳定，后续重点优化焊盘识别模块。