# Tasks - App V2 GUI 功能整合

## Phase 1: 核心聊天功能整合 (P0) ✅

### Task 1.1: 整合 Markdown 渲染到 ChatMode
**描述**: 让 AI 响应显示 Markdown 格式
**优先级**: P0
**预计时间**: 1天
**依赖**: 无
**状态**: ✅ 已完成

- [x] 修改 `pyutagent/ui/agent_panel/chat_mode.py`
  - [x] 在 `ChatMessageWidget` 中添加 `MarkdownRenderer` 实例
  - [x] 修改 `set_content()` 方法支持 Markdown 渲染
  - [x] 添加 `is_markdown` 参数控制渲染模式
  - [x] 处理 HTML 内容的安全转义

- [x] 添加代码块交互功能
  - [x] 在 `MarkdownRenderer` 中添加代码块按钮生成
  - [x] 实现复制代码功能
  - [x] 实现插入到编辑器功能
  - [x] 添加复制成功提示

- [x] 更新样式
  - [x] 添加 Markdown 渲染的 CSS 样式
  - [x] 确保代码块语法高亮主题一致
  - [x] 处理暗黑/亮色模式切换

**实现文件**: `pyutagent/ui/agent_panel/chat_mode.py`, `pyutagent/ui/components/markdown_renderer.py`

---

### Task 1.2: 整合流式响应处理
**描述**: 实现打字机效果流式输出
**优先级**: P0
**预计时间**: 1.5天
**依赖**: Task 1.1
**状态**: ✅ 已完成

- [x] 创建 `StreamingResponseWorker` 类
  - [x] 在 `pyutagent/ui/workers/` 创建文件
  - [x] 继承 `QThread`
  - [x] 定义信号: `chunk_ready`, `finished`, `error`
  - [x] 实现 `run()` 方法调用 LLM 流式 API

- [x] 修改 `main_window_v2.py`
  - [x] 添加流式响应状态管理
  - [x] 实现 `_start_streaming_response()` 方法
  - [x] 实现 `_on_streaming_chunk()` 槽函数
  - [x] 实现 `_on_streaming_finished()` 槽函数
  - [x] 添加取消流式生成按钮

- [x] 修改 `chat_mode.py`
  - [x] 添加 `append_streaming_content()` 方法
  - [x] 支持增量更新消息内容
  - [x] 处理 Markdown 渲染的增量更新
  - [x] 添加流式状态指示器

- [x] 错误处理
  - [x] 处理流式中断错误
  - [x] 处理网络超时
  - [x] 保留已生成内容

**实现文件**: `pyutagent/ui/main_window_v2.py`, `pyutagent/ui/components/streaming_handler.py`

---

### Task 1.3: 整合思考过程展示
**描述**: 在 Agent 响应中显示思考过程
**优先级**: P0
**预计时间**: 0.5天
**依赖**: 无
**状态**: ✅ 已完成

- [x] 修改 `agent_mode.py`
  - [x] 在 `AgentMode` 中添加 `ThinkingExpander` 实例
  - [x] 实现 `_on_thinking_started()` 方法
  - [x] 实现 `_on_thinking_step()` 方法
  - [x] 实现 `_on_thinking_finished()` 方法
  - [x] 添加思考耗时显示

- [x] 连接信号
  - [x] 连接 `AgentWorker` 的思考信号
  - [x] 更新思考步骤状态
  - [x] 处理思考过程折叠/展开

**实现文件**: `pyutagent/ui/agent_panel/agent_mode.py`, `pyutagent/ui/components/thinking_expander.py`

---

## Phase 2: 编辑器增强整合 (P0) ✅

### Task 2.1: 整合幽灵文本到 CodeEditor
**描述**: 在编辑器中显示 AI 建议
**优先级**: P0
**预计时间**: 1.5天
**依赖**: 无
**状态**: ✅ 已完成

- [x] 修改 `pyutagent/ui/editor/code_editor.py`
  - [x] 添加 `GhostTextRenderer` 实例
  - [x] 在 `paintEvent` 中调用幽灵文本渲染
  - [x] 实现 `show_suggestion()` 方法
  - [x] 实现 `hide_suggestion()` 方法
  - [x] 处理光标位置变化

- [x] 添加建议管理
  - [x] 创建 `AISuggestionProvider` 实例
  - [x] 实现 `_request_suggestion()` 方法
  - [x] 实现 `_on_suggestion_ready()` 槽函数
  - [x] 添加建议缓存机制

- [x] 视觉优化
  - [x] 确保幽灵文本半透明效果
  - [x] 处理多行建议对齐
  - [x] 添加建议边框提示

**实现文件**: `pyutagent/ui/editor/code_editor.py`, `pyutagent/ui/editor/ghost_text.py`

---

### Task 2.2: 整合行内 Diff
**描述**: 显示代码修改建议
**优先级**: P0
**预计时间**: 1天
**依赖**: Task 2.1
**状态**: ✅ 已完成

- [x] 修改 `code_editor.py`
  - [x] 添加 `InlineDiffRenderer` 实例
  - [x] 实现 `show_diff()` 方法
  - [x] 添加 Diff 高亮样式
  - [x] 实现 Diff 导航按钮

- [x] 添加 Diff 计算
  - [x] 创建 `InlineDiffCalculator` 实例
  - [x] 计算当前代码和建议的差异
  - [x] 生成 Diff blocks

- [x] 视觉优化
  - [x] 添加行绿色背景 (#1E4620)
  - [x] 添加行红色背景 (#4A1E1E)
  - [x] 字符级差异高亮

**实现文件**: `pyutagent/ui/editor/code_editor.py`, `pyutagent/ui/editor/inline_diff.py`

---

### Task 2.3: 实现 Tab/Esc 交互
**描述**: 接受或拒绝 AI 建议
**优先级**: P0
**预计时间**: 0.5天
**依赖**: Task 2.1, Task 2.2
**状态**: ✅ 已完成

- [x] 修改 `code_editor.py`
  - [x] 重写 `keyPressEvent()` 方法
  - [x] Tab 键接受建议
  - [x] Esc 键拒绝建议
  - [x] Ctrl+Right 接受下一个词
  - [x] 其他按键正常处理

- [x] 实现接受逻辑
  - [x] `_accept_suggestion()` 方法
  - [x] 插入建议文本
  - [x] 支持撤销操作
  - [x] 清除幽灵文本

- [x] 实现拒绝逻辑
  - [x] `_reject_suggestion()` 方法
  - [x] 隐藏幽灵文本
  - [x] 不清除编辑器内容

- [x] 视觉提示
  - [x] 状态栏显示提示
  - [x] 添加接受/拒绝按钮
  - [x] 快捷键提示

**实现文件**: `pyutagent/ui/editor/code_editor.py`

---

## Phase 3: Agent 可视化整合 (P0) ✅

### Task 3.1: 整合 AgentWorker
**描述**: 连接 Agent 执行到 UI
**优先级**: P0
**预计时间**: 1天
**依赖**: 无
**状态**: ✅ 已完成

- [x] 修改 `main_window_v2.py`
  - [x] 创建 `AgentWorker` 实例
  - [x] 连接 `AutonomousLoop` 到 `AgentWorker`
  - [x] 实现 `_setup_agent_worker()` 方法
  - [x] 实现所有信号处理器

- [x] 信号连接
  - [x] `state_changed` -> `_on_agent_state_changed()`
  - [x] `progress_updated` -> `_on_agent_progress_updated()`
  - [x] `tool_call_started` -> `_on_tool_call_started()`
  - [x] `tool_call_completed` -> `_on_tool_call_completed()`
  - [x] `error_occurred` -> `_on_agent_error()`

**实现文件**: `pyutagent/ui/main_window_v2.py`, `pyutagent/ui/agent_panel/agent_worker.py`

---

### Task 3.2: 整合进度追踪
**描述**: 显示 Agent 执行进度
**优先级**: P0
**预计时间**: 0.5天
**依赖**: Task 3.1
**状态**: ✅ 已完成

- [x] 修改 `agent_mode.py`
  - [x] 添加 `ProgressTracker` 实例
  - [x] 实现进度更新逻辑
  - [x] 显示当前步骤/总步骤
  - [x] 添加步骤详情展开

**实现文件**: `pyutagent/ui/agent_panel/agent_mode.py`, `pyutagent/ui/components/progress_tracker.py`

---

### Task 3.3: 整合错误处理
**描述**: 显示和处理 Agent 错误
**优先级**: P0
**预计时间**: 0.5天
**依赖**: Task 3.1
**状态**: ✅ 已完成

- [x] 修改 `agent_mode.py`
  - [x] 添加 `ErrorListWidget` 实例
  - [x] 实现错误显示逻辑
  - [x] 添加重试按钮
  - [x] 添加跳过按钮

**实现文件**: `pyutagent/ui/agent_panel/agent_mode.py`, `pyutagent/ui/components/error_display.py`

---

## Phase 4: 上下文系统整合 (P1) ✅

### Task 4.1: 整合符号索引到 MentionSystem
**描述**: 支持 @symbol 引用
**优先级**: P1
**预计时间**: 1天
**依赖**: 无
**状态**: ✅ 已完成

- [x] 修改 `pyutagent/ui/commands/mention_system.py`
  - [x] 添加 `SymbolIndexer` 实例
  - [x] 实现符号搜索逻辑
  - [x] 添加 @symbol 类型处理
  - [x] 实现类型分组显示

- [x] 修改 MentionPopup
  - [x] 添加符号类型图标
  - [x] 实现模糊匹配
  - [x] 添加最近使用排序

**实现文件**: `pyutagent/ui/commands/mention_system.py`, `pyutagent/ui/services/symbol_indexer.py`

---

### Task 4.2: 整合语义搜索对话框
**描述**: 自然语言搜索代码
**优先级**: P1
**预计时间**: 0.5天
**依赖**: Task 4.1
**状态**: ✅ 已完成

- [x] 修改 `main_window_v2.py`
  - [x] 添加 `SemanticSearchDialog` 实例
  - [x] 实现 `_show_semantic_search()` 方法
  - [x] 添加 Ctrl+Shift+F 快捷键
  - [x] 连接搜索结果到上下文

**实现文件**: `pyutagent/ui/main_window_v2.py`, `pyutagent/ui/dialogs/semantic_search_dialog.py`

---

### Task 4.3: 增强 ContextManager
**描述**: 支持符号类型上下文
**优先级**: P1
**预计时间**: 0.5天
**依赖**: Task 4.1
**状态**: ✅ 已完成

- [x] 修改 `pyutagent/ui/agent_panel/context_manager.py`
  - [x] 添加符号类型上下文项
  - [x] 实现 `add_symbol()` 方法
  - [x] 显示符号定义预览
  - [x] 双击打开符号详情

**实现文件**: `pyutagent/ui/agent_panel/context_manager.py`

---

## Phase 5: 终端和文件树整合 (P1) ✅

### Task 5.1: 整合终端 AI 修复
**描述**: 终端错误一键修复
**优先级**: P1
**预计时间**: 1天
**依赖**: 无
**状态**: ✅ 已完成

- [x] 修改 `pyutagent/ui/terminal/embedded_terminal.py`
  - [x] 添加错误检测逻辑
  - [x] 实现 `ErrorDetector` 实例
  - [x] 添加 "Ask AI" 浮动按钮
  - [x] 实现一键修复流程

**实现文件**: `pyutagent/ui/terminal/embedded_terminal.py`

---

### Task 5.2: 整合 Git 状态到文件树
**描述**: 显示文件 Git 状态
**优先级**: P1
**预计时间**: 0.5天
**依赖**: 无
**状态**: ✅ 已完成

- [x] 修改 `pyutagent/ui/widgets/file_tree.py`
  - [x] 添加 `GitStatusService` 实例
  - [x] 实现状态检测逻辑
  - [x] 添加状态图标显示
  - [x] 添加颜色标识

**实现文件**: `pyutagent/ui/widgets/file_tree.py`, `pyutagent/ui/services/git_status_service.py`

---

### Task 5.3: 添加文件树搜索
**描述**: 快速定位文件
**优先级**: P1
**预计时间**: 0.5天
**依赖**: 无
**状态**: ✅ 已完成

- [x] 修改 `pyutagent/ui/panels/sidebar_panel.py`
  - [x] 添加搜索框到文件树面板
  - [x] 实现模糊匹配
  - [x] 高亮匹配结果
  - [x] 添加 Ctrl+F 快捷键

**实现文件**: `pyutagent/ui/panels/sidebar_panel.py`, `pyutagent/ui/widgets/file_tree.py`

---

## Phase 6: 命令面板和快捷键 (P2) ✅

### Task 6.1: 完善命令面板
**描述**: 完整命令列表
**优先级**: P2
**预计时间**: 0.5天
**依赖**: 无
**状态**: ✅ 已完成

- [x] 修改 `pyutagent/ui/command_palette.py`
  - [x] 注册所有新命令
  - [x] 测试模糊搜索
  - [x] 验证快捷键显示
  - [x] 添加 Ctrl+Shift+P 快捷键

**实现文件**: `pyutagent/ui/command_palette.py`

---

### Task 6.2: 整合快捷键配置
**描述**: 可自定义快捷键
**优先级**: P2
**预计时间**: 0.5天
**依赖**: Task 6.1
**状态**: ✅ 已完成

- [x] 修改 `main_window_v2.py`
  - [x] 添加 `_show_shortcuts_dialog()` 方法
  - [x] 添加菜单项
  - [x] 添加 Ctrl+K Ctrl+S 快捷键
  - [x] 保存配置到文件

**实现文件**: `pyutagent/ui/main_window_v2.py`, `pyutagent/ui/dialogs/keyboard_shortcuts_dialog.py`

---

## Phase 7: 性能优化 (P2) ✅

### Task 7.1: 流式渲染优化
**描述**: 提高流式响应性能
**优先级**: P2
**预计时间**: 1天
**依赖**: Task 1.2
**状态**: ✅ 已完成

- [x] 实现 `OptimizedStreamingHandler`
  - [x] 添加自适应渲染速度
  - [x] 限制最大帧率
  - [x] 优化内存使用

**实现文件**: `pyutagent/ui/components/streaming_handler.py`

---

### Task 7.2: 符号索引优化
**描述**: 提高符号搜索速度
**优先级**: P2
**预计时间**: 1天
**依赖**: Task 4.1
**状态**: ✅ 已完成

- [x] 优化 `SymbolIndexer`
  - [x] 实现增量更新
  - [x] 优化搜索算法
  - [x] 添加缓存机制

**实现文件**: `pyutagent/ui/services/symbol_indexer.py`

---

## Phase 8: 测试验证 (P0) ✅

### Task 8.1: 单元测试
**描述**: 确保所有组件正常工作
**优先级**: P0
**预计时间**: 1天
**依赖**: 所有 Phase
**状态**: ✅ 已完成

- [x] 运行所有单元测试
- [x] 修复失败的测试
- [x] 添加缺失的测试
- [x] 确保覆盖率 > 90%

**测试文件**: `tests/unit/ui/` 目录下 270+ 测试

---

### Task 8.2: 集成测试
**描述**: 验证模块协同工作
**优先级**: P0
**预计时间**: 0.5天
**依赖**: Task 8.1
**状态**: ✅ 已完成

- [x] 运行集成测试
- [x] 修复集成问题
- [x] 添加端到端测试

**测试文件**: `tests/integration/test_gui_integration.py`

---

### Task 8.3: 手动测试
**描述**: 验证用户体验
**优先级**: P0
**预计时间**: 0.5天
**依赖**: Task 8.2
**状态**: ✅ 已完成

- [x] 测试流式响应
- [x] 测试行内编辑
- [x] 测试 Agent 可视化
- [x] 测试 @symbol 引用
- [x] 测试语义搜索
- [x] 测试终端 AI 修复

---

## 后续工作建议

虽然所有功能已经实现，但以下工作可以进一步提升项目质量：

### 1. 代码审查和重构
- [ ] 审查所有新增代码的质量
- [ ] 确保代码风格一致
- [ ] 优化性能瓶颈

### 2. 文档完善
- [ ] 更新 API 文档
- [ ] 添加更多使用示例
- [ ] 编写开发者指南

### 3. 用户体验优化
- [ ] 收集用户反馈
- [ ] 优化界面布局
- [ ] 添加更多动画效果

### 4. 测试增强
- [ ] 添加更多边界测试
- [ ] 提高测试覆盖率到 95%
- [ ] 添加性能测试

### 5. 部署准备
- [ ] 打包应用程序
- [ ] 创建安装程序
- [ ] 编写发布说明

---

## 项目统计

| 指标 | 数值 |
|------|------|
| 新增文件 | 20+ |
| 修改文件 | 10+ |
| 单元测试 | 270+ |
| 集成测试 | 12 |
| 代码行数 | 5000+ |
| 文档页数 | 3 份 |

---

**最后更新**: 2026-03-06
**状态**: ✅ 所有任务完成
