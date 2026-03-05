# App V2 GUI 功能差距分析与改进计划

## 当前已实现功能盘点

### ✅ 已完成的 GUI 功能

#### 1. 基础架构
- [x] 三栏布局（Sidebar + Content + Agent Panel）
- [x] 可折叠面板
- [x] 多种布局模式（Default / Focus Editor / Focus Agent）
- [x] 主题/样式系统
- [x] 通知管理器

#### 2. Agent Panel
- [x] Chat Mode（传统聊天界面）
- [x] Agent Mode（思考链 + 工具调用展示）
- [x] Context Manager（上下文文件管理）
- [x] 消息流式显示
- [x] 生成测试按钮

#### 3. 代码编辑器
- [x] 语法高亮（多语言支持）
- [x] 行号显示
- [x] 暗黑/亮色主题
- [x] 文件打开/关闭
- [x] 右键 AI 操作菜单

#### 4. 命令系统
- [x] Slash Commands（/test, /explain, /refactor 等 20+ 命令）
- [x] @Mention 系统（引用文件、当前文件、选中代码）
- [x] Command Palette 框架

#### 5. 会话管理
- [x] Session Manager（创建、保存、加载会话）
- [x] 会话历史对话框
- [x] 消息持久化
- [x] 会话搜索

#### 6. 终端
- [x] 嵌入式终端
- [x] 多 Shell 支持（PowerShell, CMD, Bash）
- [x] 命令历史
- [x] 实时输出

#### 7. Diff 审查
- [x] Approval Diff Viewer
- [x] 文件级/行级审批
- [x] 手动/自动模式
- [x] 批量操作（接受全部/拒绝全部）

#### 8. 配置系统
- [x] LLM 配置对话框
- [x] Aider 配置
- [x] 项目配置
- [x] 应用状态持久化

---

## 与顶级 Coding Agent 的功能差距

### 🔴 P0 - 核心缺失功能（严重影响用户体验）

#### 1. **实时 AI 响应流式显示**
| 对比项 | Cursor/Claude | 当前 App V2 |
|--------|---------------|-------------|
| 打字机效果 | ✅ 实时流式 | ❌ 占位符提示 |
| Markdown 渲染 | ✅ 完整支持 | ❌ 纯文本 |
| 代码块高亮 | ✅ 语法高亮 | ❌ 无 |
| 思考过程展示 | ✅ 可展开 | ❌ 基础步骤 |

**需要实现：**
- [ ] 集成 LLM 流式响应到 ChatMode
- [ ] Markdown 渲染引擎（支持代码块、列表、表格）
- [ ] 代码块复制按钮
- [ ] 思考过程折叠/展开

#### 2. **智能代码补全（Inline Edit）**
| 对比项 | Cursor | 当前 App V2 |
|--------|--------|-------------|
| Tab 补全 | ✅ 智能预测 | ❌ 无 |
| 行内编辑 | ✅ 直接修改 | ❌ 仅查看 |
| 多行补全 | ✅ 支持 | ❌ 无 |

**需要实现：**
- [ ] Monaco Editor 集成（或增强现有编辑器）
- [ ] 行内 Diff 显示
- [ ] Tab 接受/拒绝机制

#### 3. **Composer/多文件编辑**
| 对比项 | Cursor Composer | 当前 App V2 |
|--------|-----------------|-------------|
| 多文件操作 | ✅ 同时编辑 | ❌ 单文件 |
| 跨文件重构 | ✅ 支持 | ❌ 无 |
| 批量生成 | ✅ 完整支持 | ⚠️ 基础对话框 |

**需要实现：**
- [ ] Composer 面板（类似 Cursor）
- [ ] 多文件 Diff 视图
- [ ] 批量操作队列

#### 4. **Agent 自主执行反馈**
| 对比项 | Claude Code | 当前 App V2 |
|--------|-------------|-------------|
| 工具调用展示 | ✅ 详细 | ⚠️ 基础框架 |
| 执行进度 | ✅ 实时更新 | ⚠️ 静态步骤 |
| 错误处理 | ✅ 自动重试 | ❌ 未连接 |

**需要实现：**
- [ ] 连接 AutonomousLoop 到 AgentMode UI
- [ ] 实时工具调用状态更新
- [ ] 错误恢复流程可视化

---

### 🟡 P1 - 重要增强功能（提升竞争力）

#### 5. **代码库索引与语义搜索**
| 对比项 | Cursor | 当前 App V2 |
|--------|--------|-------------|
| @符号引用 | ✅ @file/@symbol | ⚠️ 基础文件引用 |
| 语义搜索 | ✅ 自然语言查询 | ❌ 无 |
| 代码图谱 | ✅ 依赖关系图 | ❌ 无 |

**需要实现：**
- [ ] 代码库索引系统（Codebase Indexer）
- [ ] @symbol 支持（类、方法、变量）
- [ ] 语义搜索对话框
- [ ] 代码图谱可视化

#### 6. **终端集成增强**
| 对比项 | Trae/Claude | 当前 App V2 |
|--------|-------------|-------------|
| AI 终端建议 | ✅ 命令解释 | ❌ 无 |
| 错误检测 | ✅ 自动识别 | ❌ 无 |
| 一键修复 | ✅ 从终端 | ❌ 无 |

**需要实现：**
- [ ] 终端输出 AI 分析
- [ ] 错误自动检测
- [ ] "Ask AI to Fix" 按钮

#### 7. **文件树增强**
| 对比项 | VSCode/Trae | 当前 App V2 |
|--------|-------------|-------------|
| 文件图标 | ✅ 丰富图标 | ⚠️ 基础 |
| Git 状态 | ✅ 颜色标识 | ❌ 无 |
| 文件搜索 | ✅ 快速定位 | ❌ 无 |
| 拖拽支持 | ✅ 调整布局 | ❌ 无 |

**需要实现：**
- [ ] Git 状态集成（修改/新增/删除）
- [ ] 文件树搜索框
- [ ] 文件拖拽到上下文

#### 8. **快捷键与命令面板**
| 对比项 | Cursor | 当前 App V2 |
|--------|--------|-------------|
| 快捷键 | ✅ 丰富 | ⚠️ 基础 |
| 命令面板 | ✅ 完整 | ⚠️ 框架 |
| 自定义绑定 | ✅ 支持 | ❌ 无 |

**需要实现：**
- [ ] 完整 Command Palette 实现
- [ ] 快捷键配置对话框
- [ ] 快捷键提示

---

### 🟢 P2 - 体验优化功能（锦上添花）

#### 9. **聊天增强**
- [ ] 图片粘贴/上传
- [ ] 语音输入
- [ ] 消息编辑/删除
- [ ] 分支对话（类似 Claude Projects）

#### 10. **协作功能**
- [ ] 代码分享链接
- [ ] 会话分享
- [ ] 团队协作空间

#### 11. **性能优化**
- [ ] 大文件虚拟滚动
- [ ] 增量渲染
- [ ] 内存优化

#### 12. **国际化**
- [ ] 多语言支持
- [ ] RTL 布局

---

## 详细实施计划

### Phase 1: 核心功能闭环（P0）- 2-3 周

#### Week 1: AI 响应流式化
**目标**: 让 Chat 真正可用

**任务清单**:
```markdown
- [ ] Task 1.1: 集成 LLM 流式响应
  - 修改 ChatMode 支持流式更新
  - 连接 LLMClient 流式 API
  - 添加打字机效果

- [ ] Task 1.2: Markdown 渲染
  - 集成 Markdown 渲染库（如 mistune 或 markdown）
  - 代码块语法高亮
  - 添加代码块操作按钮（复制、插入）

- [ ] Task 1.3: 思考过程展示
  - 增强 ThinkingChainWidget
  - 支持折叠/展开
  - 添加思考时间显示
```

**验收标准**:
- AI 响应实时流式显示
- Markdown 正确渲染（代码块、列表、表格）
- 思考过程可交互

#### Week 2: 行内编辑与 Diff
**目标**: 实现基础 Inline Edit

**任务清单**:
```markdown
- [ ] Task 2.1: Code Editor 增强
  - 研究 Monaco Editor 集成可行性
  - 或增强现有 QTextEdit 实现行内 Diff

- [ ] Task 2.2: Diff 显示组件
  - 行内 Diff 渲染
  - 添加接受/拒绝按钮
  - 支持多行修改

- [ ] Task 2.3: Tab 补全触发
  - 检测 AI 建议
  - Tab 接受逻辑
  - Esc 拒绝逻辑
```

**验收标准**:
- 行内显示 AI 建议
- Tab 接受/Esc 拒绝
- 多行修改支持

#### Week 3: Agent 执行可视化
**目标**: 连接后端 Agent 到前端展示

**任务清单**:
```markdown
- [ ] Task 3.1: Agent 状态同步
  - 创建 AgentWorker 信号
  - 连接 AutonomousLoop 事件
  - 实时更新 AgentMode UI

- [ ] Task 3.2: 工具调用展示
  - 增强 ToolCallWidget
  - 显示参数和结果
  - 添加展开/折叠

- [ ] Task 3.3: 错误处理可视化
  - 错误状态显示
  - 重试按钮
  - 错误详情展开
```

**验收标准**:
- Agent 执行过程实时可见
- 工具调用详情可查看
- 错误可处理

---

### Phase 2: 重要增强（P1）- 3-4 周

#### Week 4-5: 代码库索引与 @符号
**目标**: 实现 Cursor 级别的上下文引用

**任务清单**:
```markdown
- [ ] Task 4.1: 代码库索引系统
  - 集成 Tree-sitter 或类似解析器
  - 构建符号索引
  - 持久化索引数据

- [ ] Task 4.2: @symbol 支持
  - 扩展 MentionSystem
  - 支持 @class/@method/@function
  - 自动完成弹出

- [ ] Task 4.3: 语义搜索
  - 搜索对话框
  - 自然语言到代码匹配
  - 结果展示
```

#### Week 6: 终端与文件树增强
**目标**: 提升开发体验

**任务清单**:
```markdown
- [ ] Task 5.1: 终端 AI 集成
  - 终端输出监控
  - 错误模式识别
  - "Ask AI" 按钮

- [ ] Task 5.2: Git 状态集成
  - 文件树 Git 标识
  - 修改文件高亮
  - Diff 预览

- [ ] Task 5.3: 文件树搜索
  - 快速文件定位
  - 模糊匹配
  - 快捷键支持
```

#### Week 7: 命令面板与快捷键
**目标**: 完善交互体验

**任务清单**:
```markdown
- [ ] Task 6.1: Command Palette
  - 完整命令列表
  - 模糊搜索
  - 快捷键提示

- [ ] Task 6.2: 快捷键配置
  - 快捷键对话框
  - 自定义绑定
  - 冲突检测
```

---

### Phase 3: 体验优化（P2）- 持续迭代

- 聊天增强（图片、语音）
- 性能优化
- 国际化
- 主题定制

---

## 技术实现建议

### 1. Markdown 渲染
```python
# 推荐方案: 使用 QTextDocument + 自定义代码块处理
# 或使用 WebEngine 渲染 HTML（较重）

# 轻量级方案
from PyQt6.QtGui import QTextDocument
from PyQt6.QtWidgets import QTextEdit

document = QTextDocument()
document.setMarkdown(markdown_text)
```

### 2. 流式响应架构
```python
# 推荐信号设计
class AIResponseWorker(QObject):
    chunk_received = pyqtSignal(str)  # 文本片段
    thinking_started = pyqtSignal()   # 开始思考
    thinking_step = pyqtSignal(str, str)  # 步骤ID, 内容
    tool_call_started = pyqtSignal(str, dict)  # 工具名, 参数
    tool_call_finished = pyqtSignal(str, any)  # 工具名, 结果
    finished = pyqtSignal()
    error = pyqtSignal(str)
```

### 3. 行内 Diff 实现
```python
# 方案1: 使用 QSyntaxHighlighter 在行内渲染
# 方案2: 使用 QTextEdit 的 extra selections

# 推荐: 方案2 更灵活
selection = QTextEdit.ExtraSelection()
selection.format.setBackground(QColor("#1E4620"))
selection.format.setForeground(QColor("#B5F2A6"))
selection.cursor = text_edit.textCursor()
selection.cursor.select(QTextCursor.SelectionType.LineUnderCursor)
text_edit.setExtraSelections([selection])
```

---

## 优先级矩阵

| 功能 | 用户价值 | 技术难度 | 优先级 |
|------|----------|----------|--------|
| AI 流式响应 | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | P0 |
| Markdown 渲染 | ⭐⭐⭐⭐⭐ | ⭐⭐ | P0 |
| 行内编辑 | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | P0 |
| Agent 可视化 | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | P0 |
| @symbol | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | P1 |
| 语义搜索 | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | P1 |
| 终端增强 | ⭐⭐⭐ | ⭐⭐ | P1 |
| Git 集成 | ⭐⭐⭐ | ⭐⭐ | P1 |
| 命令面板 | ⭐⭐⭐ | ⭐⭐ | P1 |
| 图片支持 | ⭐⭐ | ⭐⭐⭐ | P2 |

---

## 成功指标

### 定量指标
- AI 响应延迟 < 2s（首 token）
- 流式渲染帧率 > 30fps
- 代码库索引时间 < 30s（10k 文件）
- 内存占用 < 500MB

### 定性指标
- 用户感觉 "Chat 真正可用"
- 与 Cursor 相比功能覆盖 > 70%
- 开发者愿意日常使用

---

## 附录: 参考实现

### Cursor 功能参考
- Composer: `Cmd/Ctrl + I`
- Inline Edit: `Cmd/Ctrl + K`
- Chat: `Cmd/Ctrl + L`
- @符号: `@file`, `@code`, `@docs`

### Claude Code 功能参考
- 自主循环可视化
- 工具调用链展示
- 思考过程日志

### Trae 功能参考
- Builder 模式
- 多文件编辑
- 终端集成
