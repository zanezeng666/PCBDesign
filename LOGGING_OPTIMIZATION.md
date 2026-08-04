# 日志体系优化总结

## 优化完成时间
2026-08-03

## 优化内容

### 1. 创建统一日志配置模块

**文件**: `battery_designer/logger.py`

**功能**:
- 统一的日志格式管理
- 支持控制台和文件输出
- 可配置的日志级别
- 按模块自动命名 logger
- 环境变量配置支持
- 日志装饰器（自动记录函数执行时间和异常）

**核心函数**:
- `get_logger(name)` - 获取模块 logger
- `configure_logging()` - 配置日志系统
- `set_log_level(level)` - 动态调整日志级别
- `log_function()` - 函数日志装饰器
- `log_errors()` - 异常捕获装饰器

### 2. 更新现有模块

#### battery_designer 包
- ✅ `battery_designer/__init__.py` - 导出日志功能
- ✅ `battery_designer/app.py` - 应用启动时配置日志
- ✅ `battery_designer/vlm_detection.py` - 使用统一 logger
- ✅ `battery_designer/vision.py` - 使用统一 logger

#### engine 包
- ✅ `engine/logger.py` - 创建 engine 专用日志配置
- ✅ `engine/schematic.py` - 替换 print 为 logger
- ✅ `engine/pcb.py` - 替换 print 为 logger
- ✅ `engine/schematic_png.py` - 替换 print 为 logger
- ✅ `engine/circuit_helpers.py` - 替换 print 为 logger
- ✅ `engine/gen_schematic.py` - 替换 print 为 logger

### 3. 配置文件更新

- ✅ `.gitignore` - 添加 `logs/` 目录排除

### 4. 文档和测试

- ✅ `LOGGING_GUIDE.md` - 详细的使用指南
- ✅ `test_logger.py` - 完整的测试脚本

## 日志系统特性

### 1. 统一格式
```
2026-08-03 22:30:44 | INFO     | engine.schematic | [1/3] 解析电路: BMS-1S
```

### 2. 灵活配置
```python
# 基本配置
configure_logging(level=logging.INFO, console=True)

# 输出到文件
configure_logging(level=logging.DEBUG, log_dir="logs")

# 环境变量控制
export LOG_LEVEL=DEBUG
export LOG_DIR=logs
```

### 3. 模块化命名
每个模块自动获得独立命名的 logger：
- `battery_designer.app`
- `engine.schematic`
- `engine.pcb`

### 4. 高级功能

#### 函数日志装饰器
```python
@log_function(level=logging.INFO, log_time=True)
def process_data(data):
    return data.upper()

# 自动输出: INFO: Calling process_data() - 0.100s
```

#### 异常捕获装饰器
```python
@log_errors("处理失败", reraise=False)
def risky_function():
    raise ValueError("错误")

# 自动记录异常，返回 None
```

## 使用示例

### 在新模块中使用

```python
from battery_designer import get_logger

logger = get_logger(__name__)

def generate():
    logger.info("开始生成原理图")
    try:
        # ... 业务逻辑 ...
        logger.info("生成完成")
    except Exception as e:
        logger.exception("生成失败")
```

### 在应用入口配置

```python
import logging
from battery_designer import configure_logging

# 启动时配置一次
configure_logging(
    level=logging.INFO,
    log_dir="logs",
    console=True
)
```

## 性能优化

1. **惰性求值**: 推荐使用 `%` 格式化而非 f-string
   ```python
   logger.debug("处理文件: %s", filename)  # 推荐
   logger.debug(f"处理文件: {filename}")    # 避免
   ```

2. **级别控制**: 生产环境使用 INFO 级别，避免 DEBUG 日志影响性能

3. **日志轮转**: 日志文件按日期自动生成，便于管理

## 后续建议

### 1. 继续迁移 print 语句

以下文件仍有 print 语句，建议后续迁移：
- `scripts/test_full_flow.py`
- `scripts/test_freerouting_api.py`
- `scripts/build_kicad_template.py`
- `data/ic_templates/adapt_common.py`
- 其他 `data/ic_templates/` 下的文件

迁移方法：
```python
# 旧代码
print(f"处理中: {file}")

# 新代码
from battery_designer import get_logger
logger = get_logger(__name__)
logger.info(f"处理中: {file}")
```

### 2. 日志分析

建议使用日志分析工具：
- **ELK Stack** (Elasticsearch, Logstash, Kibana)
- **Graylog**
- **Sentry** (错误追踪)

### 3. 监控告警

对于生产环境，建议：
- 监控 ERROR 级别日志
- 设置日志异常告警
- 定期归档历史日志

## 测试验证

运行测试脚本验证日志系统：
```bash
python test_logger.py
```

测试内容：
- ✅ 基本日志功能
- ✅ 文件日志输出
- ✅ 日志装饰器
- ✅ 异常捕获装饰器
- ✅ 动态日志级别调整

## 总结

本次优化建立了完整的日志体系：

1. **统一管理**: 所有日志通过 `battery_designer.logger` 配置
2. **模块化**: 每个模块独立命名 logger
3. **灵活性**: 支持控制台、文件、环境变量配置
4. **易用性**: 提供装饰器和便捷函数
5. **可维护**: 详细文档和测试覆盖

日志系统的完善将大大提高项目的可维护性和调试效率。