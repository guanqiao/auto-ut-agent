# 修复计划：批量生成测试日志在GUI上不显示

## 问题分析

### 根本原因
日志系统存在断层：
- **后端**：Agent 使用 Python `logging` 模块输出日志到控制台/文件
- **前端**：GUI 依赖 Qt 信号机制接收日志
- **断层**：两者之间没有建立有效的桥接机制

### 具体问题

#### 1. 单文件生成模式
- `AgentWorker` 有 `log_message` 信号定义，但从未被触发
- `_on_progress()` 只发送 `state_changed` 信号，不发送详细日志
- Agent 内部日志通过 `logger.info()` 输出，无法传递到 GUI

#### 2. 批量生成模式
- `BatchGenerateWorker` 没有 `log_message` 信号
- `BatchProgress` 数据结构不包含日志字段
- `BatchGenerateDialog` 没有日志显示区域
- Agent 执行过程中的日志完全丢失

## 修复方案

### 方案：创建日志桥接机制

#### 步骤 1：创建 QtLogHandler
创建一个自定义的 logging Handler，将日志转发到 Qt 信号。

**文件**：`pyutagent/ui/log_handler.py`

```python
class QtLogHandler(logging.Handler):
    """Custom logging handler that emits Qt signals."""
    
    log_signal = pyqtSignal(str, str)  # message, level
    
    def emit(self, record):
        # 将日志记录转换为信号
```

#### 步骤 2：修改 BatchGenerateWorker
添加 `log_message` 信号，并在 Worker 中安装 QtLogHandler。

**修改文件**：`pyutagent/ui/batch_generate_dialog.py`

```python
class BatchGenerateWorker(QThread):
    # 新增信号
    log_message = pyqtSignal(str, str)  # message, level
    
    def run(self):
        # 安装日志处理器
        self.log_handler = QtLogHandler()
        self.log_handler.log_signal.connect(self._emit_log)
        logging.getLogger('pyutagent').addHandler(self.log_handler)
        
        # ... 原有逻辑
```

#### 步骤 3：修改 BatchGenerateDialog
添加日志显示区域，连接日志信号。

**修改文件**：`pyutagent/ui/batch_generate_dialog.py`

```python
class BatchGenerateDialog(QDialog):
    def setup_ui(self):
        # 在 Progress Group 下方添加日志区域
        # 类似 ProgressWidget 的 log_area
        
    def on_log_message(self, message: str, level: str):
        # 显示日志
```

#### 步骤 4：修改 AgentWorker（单文件模式）
在 AgentWorker 中也安装 QtLogHandler，触发 log_message 信号。

**修改文件**：`pyutagent/ui/main_window.py`

```python
class AgentWorker(QThread):
    def run(self):
        # 安装日志处理器
        self.log_handler = QtLogHandler()
        self.log_handler.log_signal.connect(self._emit_log)
        logging.getLogger('pyutagent').addHandler(self.log_handler)
        
    def _emit_log(self, message: str, level: str):
        self.log_message.emit(message, level)
```

## 实施顺序

1. ✅ 创建 `QtLogHandler` 类
2. ✅ 修改 `BatchGenerateWorker` - 添加日志信号和处理
3. ✅ 修改 `BatchGenerateDialog` - 添加日志显示区域
4. ✅ 修改 `AgentWorker` - 添加日志桥接
5. ✅ 测试验证

## 预期效果

- 批量生成时，GUI 上显示详细的执行日志
- 单文件生成时，GUI 上显示更完整的日志
- 日志按级别着色显示（DEBUG/INFO/WARNING/ERROR）
