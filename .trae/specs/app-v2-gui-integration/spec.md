# App V2 GUI 功能整合 Spec

## Why

已开发的 GUI 组件（Markdown 渲染、流式响应、行内编辑、Agent 可视化等）需要完整整合到 App V2 主应用中。当前组件分散在各个模块，缺乏统一的集成点和协同工作机制，导致功能无法在实际使用中发挥作用。

## What Changes

### Phase 1: 核心聊天功能整合
- 整合 Markdown 渲染到 ChatMode，替换纯文本显示
- 整合流式响应处理，实现打字机效果
- 整合思考过程展示到 AgentMode

### Phase 2: 编辑器增强整合
- 整合幽灵文本到 CodeEditor
- 整合行内 Diff 显示
- 实现 Tab/Esc 交互接受/拒绝建议

### Phase 3: Agent 可视化整合
- 整合 AgentWorker 到主窗口
- 整合进度追踪组件
- 整合错误处理和重试机制

### Phase 4: 上下文系统整合
- 整合符号索引到 MentionSystem
- 整合语义搜索对话框
- 增强 ContextManager 支持符号类型

### Phase 5: 终端和文件树整合
- 整合终端 AI 错误检测和修复
- 整合 Git 状态到文件树
- 添加文件树搜索功能

### Phase 6: 命令面板和快捷键
- 完善命令面板注册所有命令
- 整合快捷键配置对话框

### Phase 7: 性能优化
- 流式渲染性能优化
- 符号索引性能优化

### Phase 8: 测试验证
- 单元测试验证
- 集成测试验证
- 手动测试验证

## Impact

### Affected Specs
- GUI 交互流程
- Agent 状态管理
- 编辑器组件架构
- 上下文管理系统

### Affected Code
- `pyutagent/ui/main_window_v2.py` - 主窗口集成
- `pyutagent/ui/agent_panel/chat_mode.py` - 聊天模式
- `pyutagent/ui/agent_panel/agent_mode.py` - Agent 模式
- `pyutagent/ui/editor/code_editor.py` - 代码编辑器
- `pyutagent/ui/commands/mention_system.py` - 提及系统
- `pyutagent/ui/panels/sidebar_panel.py` - 侧边栏

## ADDED Requirements

### Requirement: Markdown 渲染集成
The system SHALL render AI responses as Markdown with syntax highlighting.

#### Scenario: AI 响应显示
- **GIVEN** AI 返回响应内容
- **WHEN** 内容包含 Markdown 格式
- **THEN** 正确渲染标题、列表、代码块等元素
- **AND** 代码块显示语法高亮
- **AND** 提供复制代码按钮

#### Scenario: 代码块交互
- **GIVEN** 渲染后的消息包含代码块
- **WHEN** 用户点击复制按钮
- **THEN** 代码内容复制到剪贴板
- **AND** 显示复制成功提示

### Requirement: 流式响应集成
The system SHALL stream AI responses with typewriter effect.

#### Scenario: 流式输出
- **GIVEN** 用户发送消息
- **WHEN** AI 开始生成响应
- **THEN** 内容逐字/逐词显示
- **AND** 首 token 响应时间 < 2s
- **AND** 渲染帧率 > 30fps

#### Scenario: 取消生成
- **GIVEN** 流式响应正在进行
- **WHEN** 用户点击取消按钮
- **THEN** 立即停止生成
- **AND** 保留已生成内容

### Requirement: 行内编辑集成
The system SHALL display AI suggestions as ghost text in the editor.

#### Scenario: 显示建议
- **GIVEN** AI 生成代码建议
- **WHEN** 用户在编辑器中
- **THEN** 以半透明灰色显示建议
- **AND** 支持多行建议显示

#### Scenario: 接受建议
- **GIVEN** 幽灵文本显示中
- **WHEN** 用户按 Tab 键
- **THEN** 接受建议并插入代码
- **AND** 支持撤销操作

#### Scenario: 拒绝建议
- **GIVEN** 幽灵文本显示中
- **WHEN** 用户按 Esc 键
- **THEN** 隐藏建议
- **AND** 不清除编辑器内容

### Requirement: Agent 可视化集成
The system SHALL visualize Agent execution state in real-time.

#### Scenario: 状态更新
- **GIVEN** Agent 正在执行任务
- **WHEN** 状态发生变化
- **THEN** UI 实时更新状态显示
- **AND** 显示当前步骤和总步骤

#### Scenario: 工具调用
- **GIVEN** Agent 调用工具
- **WHEN** 工具执行完成
- **THEN** 显示工具名称和结果
- **AND** 支持展开查看详情

#### Scenario: 错误处理
- **GIVEN** Agent 执行遇到错误
- **WHEN** 错误发生
- **THEN** 显示错误信息
- **AND** 提供重试和跳过按钮

### Requirement: 符号索引集成
The system SHALL support @symbol references with autocomplete.

#### Scenario: 触发搜索
- **GIVEN** 用户输入 @ 符号
- **WHEN** 继续输入字符
- **THEN** 显示匹配的符号列表
- **AND** 按类型分组（类/方法/函数）
- **AND** 响应时间 < 200ms

#### Scenario: 选择符号
- **GIVEN** 符号列表显示中
- **WHEN** 用户选择符号
- **THEN** 添加到上下文
- **AND** 显示符号定义预览

### Requirement: 语义搜索集成
The system SHALL support natural language code search.

#### Scenario: 打开搜索
- **GIVEN** 用户按 Ctrl+Shift+F
- **WHEN** 搜索对话框打开
- **THEN** 可以输入自然语言查询
- **AND** 显示搜索结果列表

#### Scenario: 添加结果到上下文
- **GIVEN** 搜索结果显示中
- **WHEN** 用户选择结果
- **THEN** 添加到当前上下文
- **AND** 在 Agent Panel 显示

## MODIFIED Requirements

### Requirement: ChatMessageWidget
**Current**: 使用纯文本 QLabel 显示消息
**Modified**: 使用 QTextEdit 或 QWebEngineView 渲染 Markdown
**Migration**: 保留纯文本模式作为降级选项

### Requirement: CodeEditor
**Current**: 基础 QTextEdit 编辑器
**Modified**: 添加幽灵文本层和 Diff 高亮
**Migration**: 新功能默认关闭，可通过设置启用

### Requirement: MentionSystem
**Current**: 仅支持 @file 和 @folder
**Modified**: 添加 @symbol 支持
**Migration**: 自动检测符号类型

## REMOVED Requirements

无

## 技术架构

```
┌─────────────────────────────────────────────────────────────────┐
│                      App V2 Main Window                         │
├─────────────────────────────────────────────────────────────────┤
│  ┌──────────────┐  ┌──────────────────────┐  ┌──────────────┐  │
│  │   Sidebar    │  │      Content         │  │ Agent Panel  │  │
│  │              │  │                      │  │              │  │
│  │ - File Tree  │  │ - Code Editor        │  │ - Chat Mode  │  │
│  │   + Git      │  │   + Ghost Text       │  │   + Markdown │  │
│  │   + Search   │  │   + Inline Diff      │  │   + Stream   │  │
│  │              │  │                      │  │              │  │
│  │              │  │ - Terminal           │  │ - Agent Mode │  │
│  │              │  │   + Error Detect     │  │   + Progress │  │
│  │              │  │   + AI Fix           │  │   + Errors   │  │
│  └──────────────┘  └──────────────────────┘  └──────────────┘  │
│         ▲                    ▲                      ▲          │
│         └────────────────────┼──────────────────────┘          │
│                              ▼                                  │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                    Integration Layer                    │   │
│  │  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐   │   │
│  │  │ Markdown │ │ Streaming│ │ Ghost    │ │ Agent    │   │   │
│  │  │ Renderer │ │ Handler  │ │ Text     │ │ Worker   │   │   │
│  │  └──────────┘ └──────────┘ └──────────┘ └──────────┘   │   │
│  │  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐   │   │
│  │  │ Symbol   │ │ Semantic │ │ Git      │ │ Command  │   │   │
│  │  │ Indexer  │ │ Search   │ │ Status   │ │ Palette  │   │   │
│  │  └──────────┘ └──────────┘ └──────────┘ └──────────┘   │   │
│  └─────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

## 成功指标

### 定量指标
- 首 token 响应时间 < 2s
- 流式渲染帧率 > 30fps
- Markdown 渲染正确率 > 95%
- 符号搜索响应 < 200ms
- 内存占用 < 500MB

### 定性指标
- 用户感觉 "Chat 真正可用"
- 与 Cursor 相比功能覆盖 > 70%
- 开发者愿意日常使用
