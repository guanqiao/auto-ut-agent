# App V2 GUI 功能增强指南

## 概述

本文档介绍 App V2 GUI 的功能增强，包括 AI 流式响应、行内编辑、Agent 可视化等核心功能。

## 新增功能

### 1. AI 流式响应与 Markdown 渲染

**功能描述**：
- 实时打字机效果流式输出
- 完整 Markdown 渲染（代码块、列表、表格、引用）
- 代码块语法高亮 + 复制/插入按钮
- 思考过程可折叠展示

**使用方式**：
```python
from pyutagent.ui.components.markdown_renderer import MarkdownRenderer
from pyutagent.ui.components.streaming_handler import StreamingHandler

# 创建渲染器
renderer = MarkdownRenderer()
html = renderer.render("# Hello\n\n```python\nprint('hello')\n```")

# 流式处理
config = StreamingConfig(mode="word")
handler = StreamingHandler(config)
for chunk in handler.stream_text("Hello world"):
    print(chunk)
```

**快捷键**：
- 无特定快捷键，自动应用于 AI 响应

### 2. 行内代码编辑（Inline Edit）

**功能描述**：
- 幽灵文本层显示 AI 建议
- 行内 Diff 高亮（绿/红背景）
- Tab 接受 / Esc 拒绝交互
- 多行修改支持

**使用方式**：
```python
from pyutagent.ui.editor.ghost_text import GhostTextSuggestion, GhostTextRenderer
from pyutagent.ui.editor.inline_diff import InlineDiffCalculator

# 创建建议
suggestion = GhostTextSuggestion(
    text="print('hello')",
    start_line=1,
    start_column=0
)

# 计算 Diff
calculator = InlineDiffCalculator()
diffs = calculator.calculate_diff(old_text, new_text)
```

**快捷键**：
- `Tab` - 接受建议
- `Esc` - 拒绝建议
- `Ctrl+Right` - 接受下一个词
- `F7` - 请求 AI 建议

### 3. Agent 执行可视化

**功能描述**：
- 实时连接 AutonomousLoop 到 UI
- 工具调用状态实时更新
- 执行进度条和错误处理可视化

**使用方式**：
```python
from pyutagent.ui.agent_panel.agent_worker import AgentWorker, AgentStateSignals

# 创建 Agent Worker
worker = AgentWorker(autonomous_loop)

# 连接信号
worker.signals.state_changed.connect(on_state_changed)
worker.signals.progress_updated.connect(on_progress_updated)
worker.signals.tool_call_started.connect(on_tool_started)
```

### 4. @符号引用增强

**功能描述**：
- @symbol 引用（类/方法/函数）
- 智能自动完成（< 50ms 响应）
- 类型分组显示（类/方法/变量）

**使用方式**：
```python
from pyutagent.ui.services.symbol_indexer import SymbolIndexer

# 创建索引器
indexer = SymbolIndexer(project_path="/path/to/project")

# 搜索符号
results = indexer.search_symbols("TestClass")
```

**快捷键**：
- `@` - 触发符号引用
- `Tab` - 选择建议

### 5. 代码库语义搜索

**功能描述**：
- 自然语言搜索代码
- Ctrl+Shift+F 快捷键
- 搜索结果预览

**使用方式**：
```python
from pyutagent.ui.services.semantic_search import SemanticSearchService

# 创建搜索服务
search_service = SemanticSearchService(symbol_indexer)

# 搜索
results = search_service.search("find function for adding numbers")
```

**快捷键**：
- `Ctrl+Shift+F` - 打开语义搜索

### 6. 终端 AI 集成

**功能描述**：
- 终端错误自动检测
- "Ask AI" 按钮
- 一键应用修复

**使用方式**：
- 终端中出现错误时，点击右下角的 "🤖 Ask AI" 按钮
- AI 会分析错误并提供修复建议
- 点击 "Apply Fix" 应用修复

### 7. 文件树增强

**功能描述**：
- Git 状态标识（修改/新增/删除）
- 文件树快速搜索
- 拖拽到上下文/编辑器

**使用方式**：
- 文件树中显示 Git 状态图标
- `Ctrl+F` - 在文件树中搜索
- 拖拽文件到编辑器或上下文区域

### 8. 命令面板完善

**功能描述**：
- 30+ 命令完整列表
- 模糊搜索
- 快捷键配置对话框

**使用方式**：
```python
from pyutagent.ui.command_palette import CommandPalette, FuzzyMatcher

# 创建命令面板
palette = CommandPalette(main_window)
palette.show()

# 模糊匹配
matcher = FuzzyMatcher()
results = matcher.filter_commands(commands, "open file")
```

**快捷键**：
- `Ctrl+Shift+P` - 打开命令面板
- `Ctrl+K Ctrl+S` - 打开快捷键配置

## 性能优化

### 流式渲染优化

```python
from pyutagent.ui.components.streaming_handler import (
    StreamingHandler, StreamingConfig, OptimizedStreamingHandler
)

# 高性能配置
config = StreamingConfig(
    mode="word",
    chunk_delay_ms=10,  # 更快的渲染
    max_chunks_per_second=60  # 限制帧率
)
handler = OptimizedStreamingHandler(config)
```

### 符号索引优化

```python
from pyutagent.ui.services.symbol_indexer import SymbolIndexer

# 增量更新
indexer = SymbolIndexer(project_path)
indexer.index_project()  # 首次完整索引
indexer.update_file("changed.py")  # 仅更新变更文件
```

## 配置选项

### 快捷键配置

快捷键配置保存在 `~/.pyutagent/shortcuts.json`：

```json
{
  "open_file": "Ctrl+O",
  "save_file": "Ctrl+S",
  "command_palette": "Ctrl+Shift+P",
  "semantic_search": "Ctrl+Shift+F"
}
```

### 流式响应配置

```python
# 在 main_window_v2.py 中配置
self.streaming_config = StreamingConfig(
    mode="word",  # character/word/chunk/instant
    chunk_delay_ms=20,
    max_chunks_per_second=30
)
```

## 故障排除

### Qt 测试环境问题

如果运行测试时遇到 Qt 相关问题，请确保：
1. 已安装 PyQt6 或 PySide6
2. 有图形显示环境（或设置 `QT_QPA_PLATFORM=offscreen`）

### 符号索引失败

检查项目路径是否正确：
```python
indexer = SymbolIndexer(project_path="/absolute/path/to/project")
```

### 流式响应卡顿

调整渲染参数：
```python
config = StreamingConfig(
    mode="chunk",  # 使用块模式减少渲染次数
    chunk_size=50,  # 增大块大小
    chunk_delay_ms=5  # 减少延迟
)
```

## 开发指南

### 添加新的 GUI 组件

1. 在 `pyutagent/ui/components/` 创建组件文件
2. 添加单元测试到 `tests/unit/ui/components/`
3. 更新集成测试 `tests/integration/test_gui_integration.py`

### 扩展现有组件

参考现有组件的实现模式：
- 继承 Qt 基础类（QWidget, QFrame, QDialog）
- 使用 pyqtSignal 进行事件通知
- 提供清晰的公共 API

## 参考

- [Cursor 文档](https://www.cursor.com/blog/llm-chat-code-context)
- [PyQt6 文档](https://www.riverbankcomputing.com/static/Docs/PyQt6/)
- [pytest-qt 文档](https://pytest-qt.readthedocs.io/)
