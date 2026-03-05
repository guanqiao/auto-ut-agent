# App V2 GUI 功能增强 Spec

## Why

基于对 Cursor、Trae、Claude Code 等顶级 Coding Agent 的深度分析，当前 App V2 GUI 在以下核心交互能力上存在明显差距，严重影响用户体验：

1. **AI 响应展示简陋** - 缺乏流式输出、Markdown 渲染、代码块高亮
2. **无行内编辑能力** - 无法像 Cursor 那样直接编辑代码
3. **Agent 执行不可见** - 自主循环过程缺乏实时反馈
4. **上下文引用有限** - 仅支持文件级别，缺乏符号级别引用

这些差距导致用户无法获得现代 Coding Agent 应有的交互体验。

## What Changes

### Phase 1: 核心交互能力（P0 - 让 Chat 真正可用）

#### 1.1 AI 流式响应与 Markdown 渲染
- 实时打字机效果流式输出
- 完整 Markdown 渲染（代码块、列表、表格、引用）
- 代码块语法高亮 + 复制按钮
- 思考过程可折叠展示

#### 1.2 行内代码编辑（Inline Edit）
- 类似 Cursor 的 Tab 补全机制
- 行内 Diff 显示（添加/删除/修改）
- Tab 接受 / Esc 拒绝交互
- 多行修改支持

#### 1.3 Agent 执行可视化
- 实时连接 AutonomousLoop 到 UI
- 工具调用状态实时更新
- 执行进度可视化
- 错误恢复流程展示

### Phase 2: 上下文与搜索增强（P1 - 提升竞争力）

#### 2.1 @符号引用增强
- @file - 引用文件（已有，需增强）
- @symbol - 引用类/方法/函数
- @folder - 引用文件夹
- 智能自动完成

#### 2.2 代码库语义搜索
- 自然语言搜索代码
- 语义相似度排序
- 搜索结果预览

#### 2.3 终端 AI 集成
- 终端错误自动检测
- "Ask AI to Fix" 按钮
- 一键应用修复

### Phase 3: 体验优化（P2 - 锦上添花）

#### 3.1 文件树增强
- Git 状态标识（修改/新增/删除）
- 文件树快速搜索
- 拖拽到上下文

#### 3.2 命令面板完善
- 完整命令列表
- 模糊搜索
- 快捷键提示

## Impact

### Affected Specs
- GUI 交互设计
- Agent 状态同步
- 代码编辑器组件
- 上下文管理系统

### Affected Code
- `pyutagent/ui/chat_widget.py` - 聊天界面
- `pyutagent/ui/agent_panel/` - Agent 面板
- `pyutagent/ui/editor/` - 代码编辑器
- `pyutagent/ui/commands/` - 命令系统

## ADDED Requirements

### Requirement: AI 流式响应展示
The system SHALL provide real-time streaming AI response display.

#### Scenario: 流式输出
- **GIVEN** 用户发送消息给 AI
- **WHEN** AI 开始生成响应
- **THEN** 响应内容实时流式显示（打字机效果）
- **AND** 支持 Markdown 格式渲染
- **AND** 代码块显示语法高亮

#### Scenario: 代码块交互
- **GIVEN** AI 响应包含代码块
- **WHEN** 代码块渲染完成
- **THEN** 显示复制按钮
- **AND** 显示插入到编辑器按钮
- **AND** 点击复制将代码复制到剪贴板

#### Scenario: 思考过程展示
- **GIVEN** AI 正在思考
- **WHEN** 思考内容生成
- **THEN** 以可折叠形式展示思考过程
- **AND** 显示思考耗时

### Requirement: 行内代码编辑
The system SHALL support inline code editing like Cursor.

#### Scenario: Tab 补全触发
- **GIVEN** 用户在编辑器中编码
- **WHEN** AI 生成代码建议
- **THEN** 以幽灵文本形式显示建议
- **AND** 按 Tab 接受建议
- **AND** 按 Esc 拒绝建议

#### Scenario: 行内 Diff 显示
- **GIVEN** AI 建议修改现有代码
- **WHEN** 建议显示在编辑器中
- **THEN** 添加的行显示为绿色背景
- **AND** 删除的行显示为红色背景
- **AND** 修改的行显示对比

#### Scenario: 多行修改
- **GIVEN** AI 建议涉及多行修改
- **WHEN** 用户接受建议
- **THEN** 一次性应用所有修改
- **AND** 支持撤销操作

### Requirement: Agent 执行可视化
The system SHALL visualize autonomous agent execution.

#### Scenario: 工具调用展示
- **GIVEN** Agent 正在执行工具
- **WHEN** 工具被调用
- **THEN** 实时显示工具名称和参数
- **AND** 显示工具执行状态（运行中/完成/失败）
- **AND** 显示工具执行结果

#### Scenario: 执行进度
- **GIVEN** Agent 正在执行任务
- **WHEN** 任务有多个步骤
- **THEN** 显示当前步骤和总步骤数
- **AND** 显示进度百分比
- **AND** 已完成步骤可展开查看详情

#### Scenario: 错误处理
- **GIVEN** Agent 执行遇到错误
- **WHEN** 错误发生
- **THEN** 以红色高亮显示错误
- **AND** 显示重试按钮
- **AND** 显示跳过按钮

### Requirement: @符号引用增强
The system SHALL support enhanced @mention references.

#### Scenario: @symbol 引用
- **GIVEN** 用户输入 @ 符号
- **WHEN** 输入类名/方法名/函数名
- **THEN** 显示匹配的符号列表
- **AND** 选择后将符号定义加入上下文

#### Scenario: 智能自动完成
- **GIVEN** 用户输入 @ 后输入字符
- **WHEN** 字符匹配多个候选
- **THEN** 按类型分组显示（类/方法/变量）
- **AND** 支持模糊匹配

## MODIFIED Requirements

### Requirement: 现有 ChatMode
**Current**: 纯文本消息显示，无流式输出
**Modified**: 支持 Markdown 渲染和流式输出
**Migration**: 升级消息渲染组件，保持向后兼容

### Requirement: 现有 CodeEditor
**Current**: 只读查看，无 AI 集成
**Modified**: 支持行内 AI 建议和 Diff
**Migration**: 添加幽灵文本层和 Diff 高亮

## REMOVED Requirements

无

## 技术架构

```
┌─────────────────────────────────────────────────────────────┐
│                    App V2 GUI 架构                          │
├─────────────────────────────────────────────────────────────┤
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │  Chat Panel  │  │ Agent Panel  │  │ Editor Panel│       │
│  │              │  │              │  │              │      │
│  │ - Markdown   │  │ - Thinking   │  │ - Inline     │      │
│  │ - Streaming  │  │ - Tool Calls │  │   Edit       │      │
│  │ - Code Block │  │ - Progress   │  │ - Diff View  │      │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘      │
│         └─────────────────┼─────────────────┘               │
│                           ▼                                 │
│  ┌─────────────────────────────────────────────────────┐   │
│  │              UI Core Layer                          │   │
│  │  ┌──────────┐ ┌──────────┐ ┌──────────┐            │   │
│  │  │ Markdown │ │ Streaming│ │ @Mention │            │   │
│  │  │ Renderer │ │ Handler  │ │ System   │            │   │
│  │  └──────────┘ └──────────┘ └──────────┘            │   │
│  └─────────────────────────────────────────────────────┘   │
│                           ▼                                 │
│  ┌─────────────────────────────────────────────────────┐   │
│  │              Agent Integration Layer                │   │
│  │  ┌──────────┐ ┌──────────┐ ┌──────────┐            │   │
│  │  │ Autonomous│ │ Tool     │ │ State    │            │   │
│  │  │ Loop     │ │ Executor │ │ Sync     │            │   │
│  │  └──────────┘ └──────────┘ └──────────┘            │   │
│  └─────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

## 成功指标

### 定量指标
- 首 token 响应时间 < 2s
- 流式渲染帧率 > 30fps
- Markdown 渲染正确率 > 95%
- 行内编辑接受率 > 80%

### 定性指标
- 用户感觉 "Chat 真正可用"
- 与 Cursor 相比交互体验覆盖 > 70%
- 开发者愿意日常使用
