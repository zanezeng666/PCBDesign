# 日志系统使用指南

本项目采用统一的日志管理体系，提供一致的日志格式、级别控制和输出管理。

## 快速开始

### 1. 基本使用

```python
from battery_designer import get_logger

# 在模块开头创建 logger
logger = get_logger(__name__)

# 使用不同级别的日志
logger.debug("调试信息 - 仅在开发时显示")
logger.info("普通信息 - 记录正常操作")
logger.warning("警告信息 - 提示潜在问题")
logger.error("错误信息 - 记录错误，但不中断程序")
```

### 2. 在应用启动时配置日志

```python
import logging
from battery_designer import configure_logging

# 配置日志（通常在应用入口处调用一次）
configure_logging(
    level=logging.INFO,      # 控制台日志级别
    log_dir="logs",          # 日志文件输出目录
    console=True             # 是否输出到控制台
)
```

## 日志级别

| 级别 | 用途 | 示例 |
|------|------|------|
| DEBUG | 调试信息，详细的执行流程 | 变量值、函数调用 |
| INFO | 普通信息，记录关键操作 | 启动完成、处理完成 |
| WARNING | 警告信息，提示潜在问题 | 配置缺失、性能问题 |
| ERROR | 错误信息，但不中断程序 | 处理失败、文件不存在 |
| CRITICAL | 严重错误，可能导致程序终止 | 致命错误、系统崩溃 |

## 配置选项

### 1. 控制台输出

```python
configure_logging(
    level=logging.INFO,
    console=True  # 输出到控制台
)
```

### 2. 文件输出

```python
# 固定文件名
configure_logging(
    log_file="app.log"
)

# 自动生成带日期的日志文件
configure_logging(
    log_dir="logs"  # 生成 logs/battery_designer_20260803.log
)
```

### 3. 不同级别的控制台和文件输出

```python
configure_logging(
    level=logging.INFO,        # 控制台只显示 INFO 及以上
    file_level=logging.DEBUG,  # 文件记录 DEBUG 及以上
    log_dir="logs"
)
```

### 4. 自定义日志格式

```python
configure_logging(
    format_string="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    date_format="%Y-%m-%d %H:%M:%S"
)
```

## 从 print 迁移

### 旧代码（使用 print）

```python
def generate_schematic(yaml_path):
    print(f"开始处理: {yaml_path}")
    # ... 处理逻辑 ...
    print(f"生成完成")
    print(f"错误: 文件不存在")
```

### 新代码（使用 logging）

```python
from battery_designer import get_logger

logger = get_logger(__name__)

def generate_schematic(yaml_path):
    logger.info(f"开始处理: {yaml_path}")
    # ... 处理逻辑 ...
    logger.info(f"生成完成")
    logger.error(f"错误: 文件不存在")
```

## 环境变量控制

可以通过环境变量配置日志：

```bash
# 设置日志级别
export LOG_LEVEL=DEBUG

# 设置日志文件
export LOG_FILE=app.log

# 设置日志目录
export LOG_DIR=logs
```

在代码中：

```python
from battery_designer.logger import configure_from_env

configure_from_env()  # 自动读取环境变量配置
```

## 最佳实践

### 1. 模块级 logger

每个模块都应该创建自己的 logger：

```python
# engine/schematic.py
from battery_designer import get_logger

logger = get_logger(__name__)  # 自动命名为 "engine.schematic"

def generate():
    logger.info("生成原理图")
```

### 2. 异常记录

记录异常时使用 `exc_info=True` 或 `logger.exception()`：

```python
try:
    process()
except Exception as e:
    # 方法1: 自动包含堆栈信息
    logger.exception("处理失败")
    
    # 方法2: 手动包含堆栈信息
    logger.error("处理失败", exc_info=True)
    
    # 方法3: 只记录错误消息（不推荐）
    logger.error(f"处理失败: {e}")
```

### 3. 性能考虑

- 在生产环境使用 INFO 级别
- DEBUG 日志应该避免在高频循环中记录
- 使用惰性格式化（%格式）而非 f-string（性能更好）

```python
# 推荐（惰性格式化）
logger.debug("处理文件: %s", filename)

# 不推荐（总是执行字符串格式化）
logger.debug(f"处理文件: {filename}")
```

### 4. 敏感信息

避免记录敏感信息：

```python
# 错误示例
logger.info(f"用户登录: {username}, 密码: {password}")

# 正确示例
logger.info(f"用户登录: {username}")
```

## 日志文件管理

日志文件按日期自动生成，建议：

1. 定期清理旧日志（通过 cron job 或 Windows Task Scheduler）
2. 日志目录不应包含在版本控制中（已在 .gitignore 中排除）
3. 生产环境建议使用日志轮转工具（如 logrotate）

## 示例：完整应用

```python
# app.py
import logging
from battery_designer import configure_logging, get_logger

# 配置日志系统
configure_logging(
    level=logging.INFO,
    log_dir="logs",
    console=True
)

logger = get_logger(__name__)

def main():
    logger.info("应用启动")
    
    try:
        # 业务逻辑
        result = process_data()
        logger.info(f"处理完成: {result}")
    except Exception as e:
        logger.exception("处理失败")
        return 1
    
    logger.info("应用退出")
    return 0

if __name__ == "__main__":
    exit(main())
```

## 常见问题

### Q: 如何临时开启 DEBUG 日志？

```python
from battery_designer import set_log_level
import logging

set_log_level(logging.DEBUG)
```

### Q: 如何禁用某个模块的日志？

```python
import logging

# 禁用某个模块的日志
logging.getLogger("engine.pcb").setLevel(logging.WARNING)

# 禁用第三方库的日志
logging.getLogger("urllib3").setLevel(logging.WARNING)
```

### Q: 日志文件在哪里？

默认在 `logs/battery_designer_YYYYMMDD.log`，可以通过配置修改。

### Q: 如何查看实时日志？

Linux/macOS:
```bash
tail -f logs/battery_designer_*.log
```

Windows PowerShell:
```powershell
Get-Content logs\battery_designer_*.log -Wait
```

---

更多配置选项，请参考 `battery_designer/logger.py` 源代码。