# App V2 GUI 功能整合和实现计划

## 概述

本计划详细说明如何将已开发的 GUI 组件（Markdown 渲染、流式响应、行内编辑、Agent 可视化等）完整整合到 App V2 中，确保所有功能协同工作。

## 当前状态分析

### 已完成的组件
1. ✅ **Markdown 渲染** (`markdown_renderer.py`)
2. ✅ **流式响应处理** (`streaming_handler.py`)
3. ✅ **思考过程展示** (`thinking_expander.py`)
4. ✅ **幽灵文本** (`ghost_text.py`)
5. ✅ **行内 Diff** (`inline_diff.py`)
6. ✅ **AI 建议生成** (`ai_suggestion_provider.py`)
7. ✅ **Agent Worker** (`agent_worker.py`)
8. ✅ **进度追踪** (`progress_tracker.py`)
9. ✅ **错误显示** (`error_display.py`)
10. ✅ **符号索引** (`symbol_indexer.py`)
11. ✅ **语义搜索** (`semantic_search.py`)
12. ✅ **Git 状态** (`git_status_service.py`)
13. ✅ **命令面板** (`command_palette.py`)

### 需要整合的模块
- `main_window_v2.py` - 主窗口
- `chat_mode.py` - 聊天模式
- `agent_mode.py` - Agent 模式
- `code_editor.py` - 代码编辑器
- `context_manager.py` - 上下文管理
- `mention_system.py` - @提及系统

## 实施步骤

### Phase 1: 核心聊天功能整合 (优先级: P0)

#### Step 1.1: 整合 Markdown 渲染到 ChatMode
**目标**: 让 AI 响应显示 Markdown 格式

**具体任务**:
- [ ] 修改 `chat_mode.py` 中的 `ChatMessageWidget`
- [ ] 使用 `MarkdownRenderer` 替换纯文本显示
- [ ] 添加代码块复制按钮
- [ ] 添加代码块插入到编辑器按钮
- [ ] 处理流式内容更新

**代码修改位置**:
```python
# pyutagent/ui/agent_panel/chat_mode.py
class ChatMessageWidget(QWidget):
    def __init__(self, ...):
        # 添加 Markdown 渲染器
        self._markdown_renderer = MarkdownRenderer()
        self._content_label = QLabel()
        
    def set_content(self, content: str, is_markdown: bool = True):
        if is_markdown:
            html = self._markdown_renderer.render(content)
            self._content_label.setText(html)
```

**验收标准**:
- AI 响应显示为格式化 Markdown
- 代码块有语法高亮
- 代码块可复制

#### Step 1.2: 整合流式响应
**目标**: 实现打字机效果流式输出

**具体任务**:
- [ ] 在 `main_window_v2.py` 中添加流式处理逻辑
- [ ] 创建 `StreamingResponseWorker` 类
- [ ] 连接 LLM 流式 API
- [ ] 实现逐字/逐词渲染
- [ ] 添加取消流式生成功能

**代码修改位置**:
```python
# pyutagent/ui/main_window_v2.py
class StreamingResponseWorker(QThread):
    chunk_ready = pyqtSignal(str)
    finished = pyqtSignal()
    
    def run(self):
        # 调用 LLM 流式 API
        for chunk in self.llm_client.astream(...):
            self.chunk_ready.emit(chunk)
```

**验收标准**:
- 首 token 响应 < 2s
- 流式渲染流畅，无卡顿
- 可随时取消生成

#### Step 1.3: 整合思考过程展示
**目标**: 在 Agent 响应中显示思考过程

**具体任务**:
- [ ] 在 `agent_mode.py` 中集成 `ThinkingExpander`
- [ ] 添加思考步骤状态更新
- [ ] 显示思考耗时
- [ ] 支持折叠/展开

**验收标准**:
- 思考过程可见
- 可折叠/展开
- 显示耗时

### Phase 2: 编辑器增强整合 (优先级: P0)

#### Step 2.1: 整合幽灵文本到 CodeEditor
**目标**: 在编辑器中显示 AI 建议

**具体任务**:
- [ ] 修改 `code_editor.py`
- [ ] 添加 `GhostTextRenderer` 层
- [ ] 实现半透明灰色文本显示
- [ ] 支持多行建议

**代码修改位置**:
```python
# pyutagent/ui/editor/code_editor.py
class CodeEditor(QTextEdit):
    def __init__(self, ...):
        self._ghost_renderer = GhostTextRenderer(self)
        
    def show_suggestion(self, suggestion: GhostTextSuggestion):
        self._ghost_renderer.set_suggestion(suggestion)
```

**验收标准**:
- 幽灵文本正确显示
- 半透明效果
- 多行支持

#### Step 2.2: 整合行内 Diff
**目标**: 显示代码修改建议

**具体任务**:
- [ ] 添加 `InlineDiffRenderer`
- [ ] 实现添加行绿色高亮
- [ ] 实现删除行红色高亮
- [ ] 添加 Diff 导航按钮

**验收标准**:
- 添加/删除行正确高亮
- 可导航到下一个/上一个修改

#### Step 2.3: 实现 Tab/Esc 交互
**目标**: 接受或拒绝 AI 建议

**具体任务**:
- [ ] 添加键盘事件监听
- [ ] Tab 键接受建议
- [ ] Esc 键拒绝建议
- [ ] Ctrl+Right 接受下一个词
- [ ] 添加视觉提示

**代码修改位置**:
```python
def keyPressEvent(self, event):
    if event.key() == Qt.Key.Key_Tab:
        self._accept_suggestion()
    elif event.key() == Qt.Key.Key_Escape:
        self._reject_suggestion()
```

**验收标准**:
- Tab 接受建议
- Esc 拒绝建议
- 视觉提示清晰

### Phase 3: Agent 可视化整合 (优先级: P0)

#### Step 3.1: 整合 AgentWorker
**目标**: 连接 Agent 执行到 UI

**具体任务**:
- [ ] 在 `main_window_v2.py` 中创建 `AgentWorker`
- [ ] 连接 `AutonomousLoop` 到 `AgentWorker`
- [ ] 实现信号转发机制
- [ ] 处理状态更新

**代码修改位置**:
```python
# pyutagent/ui/main_window_v2.py
def _setup_agent_worker(self):
    self._agent_worker = AgentWorker(self.autonomous_loop)
    self._agent_worker.signals.state_changed.connect(
        self._on_agent_state_changed
    )
```

**验收标准**:
- Agent 状态实时更新
- 工具调用可见

#### Step 3.2: 整合进度追踪
**目标**: 显示 Agent 执行进度

**具体任务**:
- [ ] 在 `agent_mode.py` 中添加 `ProgressTracker`
- [ ] 显示当前步骤/总步骤
- [ ] 显示进度百分比
- [ ] 添加步骤详情展开

**验收标准**:
- 进度条显示正确
- 步骤详情可展开

#### Step 3.3: 整合错误处理
**目标**: 显示和处理 Agent 错误

**具体任务**:
- [ ] 添加 `ErrorListWidget` 到 `agent_mode.py`
- [ ] 显示错误详情
- [ ] 添加重试按钮
- [ ] 添加跳过按钮

**验收标准**:
- 错误正确显示
- 可重试/跳过

### Phase 4: 上下文系统整合 (优先级: P1)

#### Step 4.1: 整合符号索引到 MentionSystem
**目标**: 支持 @symbol 引用

**具体任务**:
- [ ] 修改 `mention_system.py`
- [ ] 集成 `SymbolIndexer`
- [ ] 添加 @symbol 类型
- [ ] 实现符号搜索
- [ ] 添加类型分组显示

**代码修改位置**:
```python
# pyutagent/ui/commands/mention_system.py
class MentionSystem:
    def __init__(self):
        self._symbol_indexer = SymbolIndexer(project_path)
        
    def search_symbols(self, query: str) -> List[SymbolIndexEntry]:
        return self._symbol_indexer.search_symbols(query)
```

**验收标准**:
- @ 触发符号搜索
- 类型分组显示
- 响应 < 200ms

#### Step 4.2: 整合语义搜索对话框
**目标**: 自然语言搜索代码

**具体任务**:
- [ ] 在 `main_window_v2.py` 中添加语义搜索快捷键
- [ ] 创建 `SemanticSearchDialog` 实例
- [ ] 连接搜索结果到上下文
- [ ] 添加 Ctrl+Shift+F 快捷键

**验收标准**:
- Ctrl+Shift+F 打开搜索
- 结果可添加到上下文

#### Step 4.3: 增强 ContextManager
**目标**: 支持符号类型上下文

**具体任务**:
- [ ] 修改 `context_manager.py`
- [ ] 添加符号类型上下文项
- [ ] 显示符号定义预览
- [ ] 双击打开符号详情

**验收标准**:
- 符号可添加到上下文
- 显示预览

### Phase 5: 终端和文件树整合 (优先级: P1)

#### Step 5.1: 整合终端 AI 修复
**目标**: 终端错误一键修复

**具体任务**:
- [ ] 修改 `embedded_terminal.py`
- [ ] 添加错误检测逻辑
- [ ] 添加 "Ask AI" 按钮
- [ ] 实现一键修复流程

**验收标准**:
- 错误自动检测
- 可一键修复

#### Step 5.2: 整合 Git 状态到文件树
**目标**: 显示文件 Git 状态

**具体任务**:
- [ ] 修改 `file_tree.py`
- [ ] 集成 `GitStatusService`
- [ ] 添加状态图标
- [ ] 添加颜色标识

**验收标准**:
- Git 状态正确显示
- 颜色标识清晰

#### Step 5.3: 添加文件树搜索
**目标**: 快速定位文件

**具体任务**:
- [ ] 添加搜索框到文件树面板
- [ ] 实现模糊匹配
- [ ] 高亮匹配结果
- [ ] 添加 Ctrl+F 快捷键

**验收标准**:
- 可搜索文件
- 模糊匹配

### Phase 6: 命令面板和快捷键 (优先级: P2)

#### Step 6.1: 完善命令面板
**目标**: 完整命令列表

**具体任务**:
- [ ] 确保所有命令已注册
- [ ] 测试模糊搜索
- [ ] 验证快捷键显示
- [ ] 添加 Ctrl+Shift+P 快捷键

**验收标准**:
- 30+ 命令可用
- 模糊搜索工作

#### Step 6.2: 整合快捷键配置
**目标**: 可自定义快捷键

**具体任务**:
- [ ] 添加快捷键配置菜单项
- [ ] 保存配置到文件
- [ ] 加载配置到命令面板
- [ ] 添加 Ctrl+K Ctrl+S 快捷键

**验收标准**:
- 可修改快捷键
- 配置持久化

### Phase 7: 性能优化 (优先级: P2)

#### Step 7.1: 流式渲染优化
**目标**: 提高流式响应性能

**具体任务**:
- [ ] 实现 `OptimizedStreamingHandler`
- [ ] 添加自适应渲染速度
- [ ] 限制最大帧率
- [ ] 优化内存使用

**代码示例**:
```python
config = StreamingConfig(
    mode="word",
    chunk_delay_ms=10,
    max_chunks_per_second=60
)
```

**验收标准**:
- 渲染帧率 > 30fps
- 内存占用 < 500MB

#### Step 7.2: 符号索引优化
**目标**: 提高符号搜索速度

**具体任务**:
- [ ] 实现增量更新
- [ ] 优化搜索算法
- [ ] 添加缓存机制
- [ ] 异步索引

**验收标准**:
- 搜索响应 < 50ms
- 增量更新 < 1s

### Phase 8: 测试和验证 (优先级: P0)

#### Step 8.1: 单元测试
**目标**: 确保所有组件正常工作

**具体任务**:
- [ ] 运行所有单元测试
- [ ] 修复失败的测试
- [ ] 添加缺失的测试
- [ ] 确保覆盖率 > 90%

#### Step 8.2: 集成测试
**目标**: 验证模块协同工作

**具体任务**:
- [ ] 运行集成测试
- [ ] 修复集成问题
- [ ] 添加端到端测试

#### Step 8.3: 手动测试
**目标**: 验证用户体验

**具体任务**:
- [ ] 测试流式响应
- [ ] 测试行内编辑
- [ ] 测试 Agent 可视化
- [ ] 测试 @symbol 引用
- [ ] 测试语义搜索
- [ ] 测试终端 AI 修复

## 实施顺序

```
Phase 1: 核心聊天功能
  ├─ Step 1.1: Markdown 渲染
  ├─ Step 1.2: 流式响应
  └─ Step 1.3: 思考过程

Phase 2: 编辑器增强
  ├─ Step 2.1: 幽灵文本
  ├─ Step 2.2: 行内 Diff
  └─ Step 2.3: Tab/Esc 交互

Phase 3: Agent 可视化
  ├─ Step 3.1: AgentWorker
  ├─ Step 3.2: 进度追踪
  └─ Step 3.3: 错误处理

Phase 4: 上下文系统
  ├─ Step 4.1: 符号索引
  ├─ Step 4.2: 语义搜索
  └─ Step 4.3: ContextManager

Phase 5: 终端和文件树
  ├─ Step 5.1: 终端 AI
  ├─ Step 5.2: Git 状态
  └─ Step 5.3: 文件树搜索

Phase 6: 命令面板
  ├─ Step 6.1: 命令面板
  └─ Step 6.2: 快捷键配置

Phase 7: 性能优化
  ├─ Step 7.1: 流式优化
  └─ Step 7.2: 索引优化

Phase 8: 测试验证
  ├─ Step 8.1: 单元测试
  ├─ Step 8.2: 集成测试
  └─ Step 8.3: 手动测试
```

## 验收标准

### 功能验收
- [ ] AI 流式响应可用
- [ ] Markdown 渲染正确
- [ ] 行内编辑 Tab/Esc 工作
- [ ] Agent 执行过程可见
- [ ] @symbol 引用可用
- [ ] 语义搜索工作
- [ ] 终端 AI 修复可用
- [ ] Git 状态显示正确

### 性能验收
- [ ] 首 token 响应 < 2s
- [ ] 流式渲染 > 30fps
- [ ] 符号搜索 < 200ms
- [ ] 内存占用 < 500MB

### 质量验收
- [ ] 单元测试通过率 > 90%
- [ ] 集成测试通过
- [ ] 代码覆盖率 > 80%
- [ ] 文档完整

## 风险和对策

| 风险 | 可能性 | 影响 | 对策 |
|------|--------|------|------|
| Qt 测试环境问题 | 高 | 中 | 使用 pytest-qt 的 qtbot fixture |
| 性能不达标 | 中 | 高 | 提前进行性能测试，优化关键路径 |
| 组件集成冲突 | 中 | 高 | 分阶段集成，每阶段充分测试 |
| API 不兼容 | 低 | 高 | 提前检查 API 兼容性 |

## 时间估算

| Phase | 预计时间 |
|-------|----------|
| Phase 1 | 3 天 |
| Phase 2 | 3 天 |
| Phase 3 | 2 天 |
| Phase 4 | 2 天 |
| Phase 5 | 2 天 |
| Phase 6 | 1 天 |
| Phase 7 | 2 天 |
| Phase 8 | 2 天 |
| **总计** | **17 天** |
