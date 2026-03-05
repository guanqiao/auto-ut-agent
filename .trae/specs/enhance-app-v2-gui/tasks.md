# Tasks - App V2 GUI 功能增强

## Phase 1: 核心交互能力（P0 - 让 Chat 真正可用）

### Task 1: AI 流式响应与 Markdown 渲染
**描述**: 实现实时流式输出、Markdown 渲染、代码块高亮
**优先级**: P0
**预计时间**: 1周

- [x] SubTask 1.1: 创建 Markdown 渲染组件
  - 创建 `pyutagent/ui/components/markdown_renderer.py`
  - 集成 mistune 或 markdown 库
  - 支持代码块、列表、表格、引用
  - 添加代码块语法高亮

- [x] SubTask 1.2: 实现流式响应处理
  - 创建 `pyutagent/ui/components/streaming_handler.py`
  - 实现打字机效果
  - 支持逐字/逐词渲染
  - 添加渲染性能优化

- [x] SubTask 1.3: 增强 ChatMessageWidget
  - 修改 `pyutagent/ui/agent_panel/chat_mode.py`
  - 集成 Markdown 渲染
  - 添加代码块复制按钮
  - 添加插入到编辑器按钮

- [x] SubTask 1.4: 实现思考过程展示
  - 创建 `pyutagent/ui/components/thinking_expander.py`
  - 支持折叠/展开
  - 显示思考耗时
  - 添加到 AgentMode

- [x] SubTask 1.5: 连接 LLM 流式 API
  - 修改 `pyutagent/ui/main_window_v2.py`
  - 集成 LLMClient 流式接口
  - 实现 on_message_sent 流式处理
  - 添加错误处理

- [x] SubTask 1.6: 编写单元测试
  - 测试 Markdown 渲染
  - 测试流式处理
  - 测试代码块交互

---

### Task 2: 行内代码编辑（Inline Edit）
**描述**: 实现类似 Cursor 的 Tab 补全和行内 Diff
**优先级**: P0
**预计时间**: 1.5周

- [ ] SubTask 2.1: 研究 Monaco Editor 集成
  - 评估 Monaco Editor 嵌入可行性
  - 或增强现有 QTextEdit
  - 确定技术方案

- [ ] SubTask 2.2: 实现幽灵文本层
  - 创建 `pyutagent/ui/editor/ghost_text.py`
  - 在编辑器上叠加建议文本
  - 支持半透明灰色显示
  - 支持多行幽灵文本

- [ ] SubTask 2.3: 实现行内 Diff 高亮
  - 创建 `pyutagent/ui/editor/inline_diff.py`
  - 添加行背景高亮（绿/红）
  - 支持字符级 Diff
  - 添加 Diff 导航

- [ ] SubTask 2.4: 实现 Tab/Esc 交互
  - 添加键盘事件监听
  - Tab 接受建议
  - Esc 拒绝建议
  - 添加视觉提示

- [ ] SubTask 2.5: 集成 AI 建议生成
  - 创建 `pyutagent/ui/editor/ai_suggestion_provider.py`
  - 调用 LLM 生成代码建议
  - 解析建议为编辑操作
  - 缓存建议结果

- [ ] SubTask 2.6: 编写单元测试
  - 测试幽灵文本显示
  - 测试 Diff 高亮
  - 测试接受/拒绝逻辑

---

### Task 3: Agent 执行可视化
**描述**: 连接 AutonomousLoop 到 UI，实时展示执行过程
**优先级**: P0
**预计时间**: 1周

- [x] SubTask 3.1: 创建 Agent 状态同步机制
  - 创建 `pyutagent/ui/agent_panel/agent_worker.py`
  - 定义 Agent 状态信号
  - 实现状态广播
  - 连接 AutonomousLoop

- [x] SubTask 3.2: 增强 ToolCallWidget
  - 修改 `pyutagent/ui/agent_panel/agent_mode.py`
  - 添加实时状态更新
  - 显示参数和结果
  - 添加展开/折叠

- [x] SubTask 3.3: 实现执行进度条
  - 创建 `pyutagent/ui/components/progress_tracker.py`
  - 显示当前步骤/总步骤
  - 显示进度百分比
  - 添加步骤详情展开

- [x] SubTask 3.4: 实现错误处理可视化
  - 创建 `pyutagent/ui/components/error_display.py`
  - 红色高亮错误
  - 添加重试按钮
  - 添加跳过按钮

- [x] SubTask 3.5: 集成到 AgentMode
  - 修改 `pyutagent/ui/agent_panel/agent_mode.py`
  - 连接状态信号
  - 实时更新 ThinkingChain
  - 实时更新 ToolCalls

- [x] SubTask 3.6: 编写单元测试
  - 测试状态同步
  - 测试进度更新
  - 测试错误处理

---

## Phase 2: 上下文与搜索增强（P1 - 提升竞争力）

### Task 4: @符号引用增强
**描述**: 实现 @symbol 引用和智能自动完成
**优先级**: P1
**预计时间**: 1周

- [x] SubTask 4.1: 创建符号索引服务
  - 创建 `pyutagent/ui/services/symbol_indexer.py`
  - 解析代码符号（类/方法/函数）
  - 构建符号索引
  - 支持增量更新

- [x] SubTask 4.2: 扩展 MentionSystem
  - 修改 `pyutagent/ui/commands/mention_system.py`
  - 添加 @symbol 类型
  - 实现符号搜索
  - 添加类型分组显示

- [x] SubTask 4.3: 实现智能自动完成
  - 增强 MentionPopup
  - 添加模糊匹配
  - 按类型分组（类/方法/变量）
  - 添加最近使用优先

- [x] SubTask 4.4: 集成到上下文
  - 修改 ContextManager
  - 支持符号类型上下文
  - 显示符号定义预览

- [x] SubTask 4.5: 编写单元测试
  - 测试符号索引
  - 测试自动完成
  - 测试上下文集成

---

### Task 5: 代码库语义搜索
**描述**: 实现自然语言搜索代码
**优先级**: P1
**预计时间**: 1周

- [x] SubTask 5.1: 创建语义搜索对话框
  - 创建 `pyutagent/ui/dialogs/semantic_search_dialog.py`
  - 设计搜索界面（搜索框 + 结果列表 + 预览面板）
  - 支持自然语言输入
  - 显示搜索结果列表
  - 代码预览面板

- [x] SubTask 5.2: 实现语义搜索服务
  - 创建 `pyutagent/ui/services/semantic_search.py`
  - 集成代码库索引（使用现有的索引系统）
  - 实现相似度计算（基于 LLM 或向量）
  - 支持自然语言查询
  - 结果排序和过滤

- [x] SubTask 5.3: 添加快捷键和集成
  - 修改 `pyutagent/ui/main_window_v2.py`
  - 添加 Ctrl+Shift+F 快捷键
  - 集成到 Command Palette

- [x] SubTask 5.4: 编写单元测试
  - 测试搜索功能
  - 测试结果排序
  - 单元测试覆盖率 > 90%

---

### Task 6: 终端 AI 集成
**描述**: 终端错误检测和 AI 修复
**优先级**: P1
**预计时间**: 3天

- [ ] SubTask 6.1: 实现错误检测
  - 修改 `pyutagent/ui/terminal/embedded_terminal.py`
  - 监控终端输出
  - 识别错误模式
  - 高亮错误行

- [ ] SubTask 6.2: 添加 "Ask AI" 按钮
  - 在终端面板添加按钮
  - 点击发送错误到 AI
  - 显示 AI 建议

- [ ] SubTask 6.3: 实现一键修复
  - 解析 AI 修复建议
  - 显示 Diff 预览
  - 一键应用修复

---

## Phase 3: 体验优化（P2 - 锦上添花）

### Task 7: 文件树增强
**描述**: Git 状态、快速搜索、拖拽
**优先级**: P2
**预计时间**: 3天

- [ ] SubTask 7.1: 添加 Git 状态标识
  - 修改 `pyutagent/ui/panels/sidebar_panel.py`
  - 检测 Git 状态
  - 添加颜色标识（绿/黄/红）
  - 添加状态图标

- [ ] SubTask 7.2: 实现文件树搜索
  - 添加搜索框
  - 实现模糊匹配
  - 高亮匹配结果

- [ ] SubTask 7.3: 实现拖拽支持
  - 支持文件拖拽到上下文
  - 支持文件拖拽到编辑器
  - 添加视觉反馈

---

### Task 8: 命令面板完善
**描述**: 完整命令列表和模糊搜索
**优先级**: P2
**预计时间**: 2天

- [ ] SubTask 8.1: 实现完整 Command Palette
  - 修改 `pyutagent/ui/command_palette.py`
  - 添加所有命令
  - 实现模糊搜索
  - 添加快捷键显示

- [ ] SubTask 8.2: 添加快捷键配置
  - 创建快捷键配置对话框
  - 支持自定义绑定
  - 保存配置

---

## Task Dependencies

```
Phase 1:
  Task 1 (Markdown+流式) ───┐
                            ├──→ 让 Chat 真正可用
  Task 2 (行内编辑) ────────┤
                            │
  Task 3 (Agent可视化) ─────┘

Phase 2:
  Task 1 ──→ Task 4 (@symbol)
  Task 4 ──→ Task 5 (语义搜索)
  Task 3 ──→ Task 6 (终端AI)

Phase 3:
  Task 4 ──→ Task 7 (文件树)
  Task 1 ──→ Task 8 (命令面板)
```

## 验收标准

### Phase 1 验收
- [x] AI 响应实时流式显示，延迟 < 100ms/字
- [x] Markdown 正确渲染（代码块、列表、表格）
- [x] 代码块可复制、可插入编辑器
- [ ] 行内编辑 Tab 接受/Esc 拒绝可用
- [x] Agent 执行过程实时可见
- [x] 所有 P0 任务单元测试通过率 > 90%

### Phase 2 验收
- [x] @symbol 引用可用，自动完成响应 < 200ms
- [ ] 语义搜索返回结果相关度 > 80%
- [ ] 终端错误可检测，AI 修复可一键应用

### Phase 3 验收
- [ ] 文件树显示 Git 状态
- [ ] 命令面板支持模糊搜索
