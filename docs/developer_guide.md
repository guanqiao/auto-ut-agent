# PyUT Agent V2 GUI 开发者指南

## 目录

1. [架构概述](#架构概述)
2. [快速开始](#快速开始)
3. [核心组件](#核心组件)
4. [开发规范](#开发规范)
5. [调试技巧](#调试技巧)
6. [常见问题](#常见问题)

## 架构概述

### 整体架构

```
┌─────────────────────────────────────────────────────────────┐
│                    App V2 Main Window                       │
├─────────────────────────────────────────────────────────────┤
│  ┌──────────────┐  ┌──────────────────────┐  ┌───────────┐ │
│  │   Sidebar    │  │      Content         │  │   Agent   │ │
│  │              │  │                      │  │   Panel   │ │
│  │ - File Tree  │  │ - Code Editor        │  │ - Chat    │ │
│  │ - Git Status │  │ - Terminal           │  │ - Agent   │ │
│  │ - Search     │  │                      │  │   Mode    │ │
│  └──────────────┘  └──────────────────────┘  └───────────┘ │
│         ▲                    ▲                      ▲       │
│         └────────────────────┼──────────────────────┘       │
│                              ▼                              │
│  ┌─────────────────────────────────────────────────────┐   │
│  │              Integration Layer                      │   │
│  │  ┌──────────┐ ┌──────────┐ ┌──────────┐            │   │
│  │  │ Markdown │ │ Streaming│ │ Ghost    │            │   │
│  │  │ Renderer │ │ Handler  │ │ Text     │            │   │
│  │  └──────────┘ └──────────┘ └──────────┘            │   │
│  └─────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

### 模块说明

| 模块 | 路径 | 职责 |
|------|------|------|
| 主窗口 | `ui/main_window_v2.py` | 应用主窗口，协调各组件 |
| 侧边栏 | `ui/panels/sidebar_panel.py` | 文件树、项目导航 |
| 内容区 | `ui/panels/content_panel.py` | 编辑器、终端 |
| Agent 面板 | `ui/agent_panel/` | 聊天、Agent 模式 |
| UI 组件 | `ui/components/` | 可复用 UI 组件 |
| 编辑器 | `ui/editor/` | 代码编辑器增强 |
| 服务层 | `ui/services/` | 业务逻辑服务 |

## 快速开始

### 环境准备

```bash
# 克隆项目
git clone https://github.com/your-org/pyutagent.git
cd pyutagent

# 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或
venv\Scripts\activate  # Windows

# 安装依赖
pip install -r requirements.txt
```

### 运行应用

```bash
# 运行主应用
python -m pyutagent.app_v2

# 或
python pyutagent/app_v2.py
```

### 运行测试

```bash
# 运行所有测试
pytest tests/

# 运行 UI 测试
pytest tests/unit/ui/

# 运行集成测试
pytest tests/integration/

# 生成覆盖率报告
pytest --cov=pyutagent --cov-report=html
```

## 核心组件

### 1. Markdown 渲染

**文件**: `pyutagent/ui/components/markdown_renderer.py`

**使用示例**:

```python
from pyutagent.ui.components.markdown_renderer import MarkdownRenderer

# 创建渲染器
renderer = MarkdownRenderer()

# 渲染 Markdown
html = renderer.render("""
# Hello World

```python
print("Hello")
```
""")

# 在 QTextEdit 中显示
text_edit.setHtml(html)
```

**关键类**:
- `MarkdownRenderer` - 主渲染器
- `CodeBlockWidget` - 代码块组件
- `MarkdownViewer` - Markdown 查看器

### 2. 流式响应

**文件**: `pyutagent/ui/components/streaming_handler.py`

**使用示例**:

```python
from pyutagent.ui.components.streaming_handler import (
    StreamingHandler, StreamingConfig
)

# 配置
config = StreamingConfig(
    mode="word",  # character/word/chunk/instant
    chunk_delay_ms=20
)

# 创建处理器
handler = StreamingHandler(config)

# 流式处理文本
for chunk in handler.stream_text("Hello world"):
    print(chunk)
```

**关键类**:
- `StreamingHandler` - 流式处理器
- `StreamingConfig` - 配置类
- `OptimizedStreamingHandler` - 优化版本

### 3. 幽灵文本

**文件**: `pyutagent/ui/editor/ghost_text.py`

**使用示例**:

```python
from pyutagent.ui.editor.ghost_text import (
    GhostTextSuggestion, GhostTextRenderer
)

# 创建建议
suggestion = GhostTextSuggestion(
    text="def hello():\n    pass",
    start_line=1,
    start_column=0
)

# 在编辑器中渲染
renderer = GhostTextRenderer(editor)
renderer.set_suggestion(suggestion)
```

**关键类**:
- `GhostTextSuggestion` - 建议数据
- `GhostTextRenderer` - 渲染器

### 4. 行内 Diff

**文件**: `pyutagent/ui/editor/inline_diff.py`

**使用示例**:

```python
from pyutagent.ui.editor.inline_diff import InlineDiffCalculator

# 创建计算器
calculator = InlineDiffCalculator()

# 计算差异
diffs = calculator.calculate_diff(
    old_text="def hello():\n    pass",
    new_text="def hello():\n    print('hello')"
)

# 应用高亮
renderer = InlineDiffRenderer(editor)
renderer.render_diffs(diffs)
```

**关键类**:
- `InlineDiffCalculator` - 差异计算器
- `InlineDiffRenderer` - 差异渲染器
- `DiffBlock` - 差异块

### 5. Agent Worker

**文件**: `pyutagent/ui/agent_panel/agent_worker.py`

**使用示例**:

```python
from pyutagent.ui.agent_panel.agent_worker import (
    AgentWorker, AgentStateSignals
)

# 创建工作线程
worker = AgentWorker(autonomous_loop)

# 连接信号
worker.signals.state_changed.connect(on_state_changed)
worker.signals.progress_updated.connect(on_progress_updated)
worker.signals.tool_call_started.connect(on_tool_started)

# 启动
worker.start()
```

**关键类**:
- `AgentWorker` - Agent 工作线程
- `AgentStateSignals` - 状态信号
- `AgentState` - 状态枚举

### 6. 符号索引

**文件**: `pyutagent/ui/services/symbol_indexer.py`

**使用示例**:

```python
from pyutagent.ui.services.symbol_indexer import SymbolIndexer

# 创建索引器
indexer = SymbolIndexer(project_path="/path/to/project")

# 索引项目
indexer.index_project()

# 搜索符号
results = indexer.search_symbols("TestClass")

# 添加符号
indexer.add_symbol(
    name="MyClass",
    symbol_type=SymbolType.CLASS,
    file_path="test.py",
    line_number=10
)
```

**关键类**:
- `SymbolIndexer` - 符号索引器
- `SymbolIndexEntry` - 索引条目
- `SymbolType` - 符号类型

### 7. 语义搜索

**文件**: `pyutagent/ui/services/semantic_search.py`

**使用示例**:

```python
from pyutagent.ui.services.semantic_search import SemanticSearchService

# 创建服务
search_service = SemanticSearchService(symbol_indexer)

# 搜索
results = search_service.search("find function for adding numbers")

# 处理结果
for result in results:
    print(f"{result.name}: {result.score}")
```

**关键类**:
- `SemanticSearchService` - 搜索服务
- `SearchResult` - 搜索结果
- `SearchWorker` - 搜索工作线程

## 开发规范

### 代码风格

1. **命名规范**
   - 类名: `PascalCase`
   - 函数/变量: `snake_case`
   - 常量: `UPPER_SNAKE_CASE`
   - 私有: `_leading_underscore`

2. **文档字符串**
   ```python
   def my_function(param1: str, param2: int) -> bool:
       """Brief description.
       
       Detailed description if needed.
       
       Args:
           param1: Description of param1
           param2: Description of param2
           
       Returns:
           Description of return value
           
       Raises:
           ValueError: When something is wrong
       """
       pass
   ```

3. **类型注解**
   ```python
   from typing import Optional, List, Dict
   
   def process_data(
       data: List[Dict[str, Any]],
       options: Optional[Dict] = None
   ) -> List[str]:
       pass
   ```

### Qt 开发规范

1. **信号命名**
   ```python
   # 使用过去式或完成式
   data_loaded = pyqtSignal(list)
   processing_finished = pyqtSignal()
   error_occurred = pyqtSignal(str)
   ```

2. **线程安全**
   ```python
   # 耗时操作使用 QThread
   class Worker(QThread):
       result_ready = pyqtSignal(object)
       
       def run(self):
           result = self.do_heavy_work()
           self.result_ready.emit(result)
   ```

3. **内存管理**
   ```python
   # 设置父对象
   widget = QWidget(parent=self)
   
   # 或使用 deleteLater
   widget.deleteLater()
   ```

## 调试技巧

### 1. 日志调试

```python
import logging

logger = logging.getLogger(__name__)

# 不同级别
logger.debug("Debug info")
logger.info("General info")
logger.warning("Warning")
logger.error("Error occurred")
```

### 2. Qt 调试

```python
# 检查对象是否存在
if widget and not sip.isdeleted(widget):
    widget.do_something()

# 检查线程
from PyQt6.QtCore import QThread
print(f"Current thread: {QThread.currentThread()}")
```

### 3. 性能分析

```python
import time

class Timer:
    def __enter__(self):
        self.start = time.time()
        return self
    
    def __exit__(self, *args):
        elapsed = time.time() - self.start
        print(f"Elapsed: {elapsed:.3f}s")

# 使用
with Timer():
    do_something()
```

## 常见问题

### Q1: 流式响应卡顿

**原因**: 渲染频率过高

**解决**:
```python
config = StreamingConfig(
    mode="chunk",
    chunk_size=50,
    max_chunks_per_second=30
)
```

### Q2: 符号索引慢

**原因**: 项目太大或没有使用缓存

**解决**:
```python
# 使用增量更新
indexer.update_file("changed.py")

# 或限制索引范围
indexer.index_project(extensions=[".py", ".js"])
```

### Q3: 幽灵文本不显示

**原因**: 渲染时机或坐标计算问题

**解决**:
```python
# 确保在 paintEvent 中渲染
def paintEvent(self, event):
    super().paintEvent(event)
    if self._suggestion:
        self._ghost_renderer.render(self._suggestion)
```

### Q4: 内存泄漏

**原因**: 循环引用或未及时释放

**解决**:
```python
# 使用弱引用
import weakref
self._parent = weakref.ref(parent)

# 断开信号
worker.finished.disconnect(on_finished)

# 删除对象
widget.deleteLater()
```

## 贡献指南

1. **Fork 项目**
2. **创建分支**: `git checkout -b feature/my-feature`
3. **提交更改**: `git commit -am 'Add feature'`
4. **推送分支**: `git push origin feature/my-feature`
5. **创建 Pull Request**

## 参考资源

- [PyQt6 文档](https://www.riverbankcomputing.com/static/Docs/PyQt6/)
- [Qt 文档](https://doc.qt.io/)
- [pytest-qt](https://pytest-qt.readthedocs.io/)
- [项目 Wiki](https://github.com/your-org/pyutagent/wiki)

---

**最后更新**: 2026-03-06
**版本**: 2.0.0
