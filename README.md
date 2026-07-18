# Battery Protection Board Designer

本地参数化电池保护板设计服务。系统将照片轮廓、触点角色、电池参数和版本化 IC 器件包转换为机械预览，并在存在审核过的 KiCad 模板时生成候选样板生产文件。

## 启动

```powershell
& "C:\Users\26509\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe" -m venv .venv
& .\.venv\Scripts\python.exe -m pip install -r requirements.txt
& .\.venv\Scripts\python.exe -m pip install --no-deps -r requirements-ocr.txt
.\run.ps1
```

打开 `http://127.0.0.1:8000`。健康检查位于 `/api/health`，API 文档位于 `/docs`。

## 正反面照片

- 正面照片建立 PCB 的统一毫米坐标系。
- 对没有标记的历史照片，可输入已确认的矩形板框宽高。系统优先识别四个可见板角并执行单应性透视矫正；板角不可靠时回退为旋转矩形裁切，并在界面明确提示。该模式必须人工确认，不能宣称达到 ArUco 模式的 0.2 mm 精度。
- 反面照片单独完成 ArUco 标定，然后选择左右翻转、上下翻转或旋转 180°映射到正面坐标。
- 正反面轮廓平均对齐误差不超过 0.2 mm 为良好，0.2～0.5 mm 必须人工仔细确认，超过 0.5 mm 拒绝创建项目。
- 每个触点保存 `front/back` 板面；反面触点必须存在对应的背面标定记录。
- `TH`、`ID` 等辅助信号可保存为无极性触点，但仅凭丝印不会自动推断其内部网络。
- “自动识别当前面”先识别有限标签集 `B+ / B- / P+ / P- / C+ / C- / NTC / N / TH / ID`，再按文字框边缘距离匹配最近的银白色焊盘区域或孔位。候选保存文字框、区域轮廓、外接矩形、中心、尺寸、区域类型和匹配距离；没有匹配到实体区域的文字不能采纳。低置信度误读会保留原始识别文本供复核。
- 原始照片、校正图和标定结果会随项目保存，便于审计。

## 安全边界

- 只接受 1～5 颗纯串联或纯并联。
- 共口/分口由同一个触点是否同时具备充电和放电角色推导。
- 未验证 IC 始终标记为 `candidate`。
- 机械预览不代表电气设计完成。
- 生产输出必须有版本化 KiCad 模板、模板适配器，并通过 ERC、DRC 和零未连接门禁。
- 不存在空 PCB、静默丢线或失败后继续导出的降级路径。

## IC 自动解析

内置目录位于 `data/ic_catalog`。未知型号通过环境变量 `IC_RESOLVER_ENDPOINT` 指定的结构化解析服务检索；解析服务应返回按官方资料提取的候选 JSON。系统确定性排序并自动使用第一项，缓存原始结果和来源。无法得到完整引脚、封装、串数及端口拓扑时会失败。

Fortune Semiconductor 的 DW01-G 当前只登记为元数据候选，尚未绑定可信 KiCad 模板，因此可以完成输入检查与机械预览，但不会伪造 Gerber。

## 测试

```powershell
& .\.venv\Scripts\python.exe -m pytest
```
