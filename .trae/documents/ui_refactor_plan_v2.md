# PyUT Agent UI 改进重构计划 V2

## 对标研究总结

### 顶级Coding Agent UI特点分析

#### 1. Cursor (Anysphere)
**核心UI模式:**
- **Inline Edit模式**: 代码编辑器内直接AI交互，选中文本后快捷键触发
- **Composer模式**: 多文件编辑，类似聊天界面但支持文件操作
- **Agent模式**: 自主执行终端命令、文件操作，用户确认机制
- **上下文感知**: 自动包含当前文件、选中代码、整个项目作为上下文
- **Diff预览**: 所有修改先显示diff，用户确认后应用

**UI设计亮点:**
- 侧边栏Chat + 主编辑器双面板布局
- Tab式会话管理
- 代码块内嵌Apply按钮
- 终端集成，Agent可直接执行命令
- 文件变更树状展示

#### 2. Claude Code (Anthropic)
**核心UI模式:**
- **终端原生**: 纯CLI界面，但交互极其流畅
- **Agent Teams**: 5个原生子Agent分工协作
- **Skill系统**: 模块化Skill，可自定义扩展
- **上下文管理**: 智能上下文压缩，token优化

**UI设计亮点:**
- `/` 命令系统快速触发功能
- 文件选择器支持模糊搜索
- 多轮对话保持上下文
- 工具调用可视化
- 思考过程展示

#### 3. Trae (ByteDance)
**核心UI模式:**
- **Builder模式**: 从0到1构建项目
- **Chat模式**: 问答式交互
- **多模型切换**: Claude/GPT/自定义模型快速切换
- **MCP集成**: 工具生态扩展

**UI设计亮点:**
- 左侧文件树 + 中间编辑器 + 右侧Chat三栏布局
- 会话历史管理
- 模型选择下拉框
- 代码补全内联提示

#### 4. GitHub Copilot Chat
**核心UI模式:**
- **Edits模式**: 多文件编辑
- **Agent模式**: 自主执行
- **参与者和斜杠命令**: `@workspace`, `/explain`等

**UI设计亮点:**
- 与IDE深度集成
- 引用符号(@)快速选择上下文
- 代码引用高亮

#### 5. Windsurf (Codeium)
**核心UI模式:**
- **Cascade**: 非线性工作流，AI主动预测下一步
- **Agentic编辑**: AI自主执行多步骤任务
- **多文件协调**: 跨文件理解和修改

**UI设计亮点:**
- 预测性UI，AI主动建议
- 流式输出，实时预览
- 撤销/重做栈

---

## 当前PyUT Agent UI差距分析

| 维度 | 顶级Agent | PyUT Agent | 差距 |
|------|-----------|------------|------|
| **布局** | 三栏/双栏灵活布局 | 单窗口固定布局 | 大 |
| **上下文** | 智能上下文选择 | 仅当前文件 | 大 |
| **交互模式** | Chat/Inline/Agent多模式 | 仅Chat | 大 |
| **代码展示** | Diff预览、内联编辑 | 仅日志输出 | 大 |
| **工具集成** | 终端、文件系统、浏览器 | 仅Maven | 大 |
| **会话管理** | 多会话、历史记录 | 无 | 中 |
| **可视化** | 流程图、思维链 | 仅进度条 | 中 |
| **多语言** | 全语言支持 | 仅Java | 大 |

---

## 重构方案 V2

### 架构目标: 从"测试生成工具"到"通用Coding Agent平台"

### 核心设计原则

1. **Agent-First设计**: UI围绕Agent能力设计，而非功能列表
2. **上下文为核心**: 一切交互都考虑上下文管理
3. **渐进式披露**: 简单任务简单界面，复杂任务展开高级功能
4. **可视化反馈**: Agent思考过程、工具调用全程可见
5. **多模态支持**: 文本、代码、图片、终端输出统一展示

---

## Phase 1: 布局架构重构 (核心)

### 1.1 新布局系统: 三栏自适应布局

```
┌─────────────────────────────────────────────────────────────┐
│  Toolbar (Quick Actions + Context Selection)               │
├──────────┬──────────────────────────────┬───────────────────┤
│          │                              │                   │
│  Sidebar │      Main Content Area       │   Agent Panel     │
│  (Files) │      (Editor/Preview)        │   (Chat/Flow)     │
│          │                              │                   │
│  - File  │      - Code Editor           │   - Chat          │
│    Tree  │      - Diff View             │   - Agent Flow    │
│  - Search│      - Terminal              │   - Context       │
│  - Git   │      - Web Preview           │   - Tools         │
│          │                              │                   │
├──────────┴──────────────────────────────┴───────────────────┤
│  Status Bar (Project Info + Agent Status + Progress)       │
└─────────────────────────────────────────────────────────────┘
```

**文件变更:**
- 新建 `pyutagent/ui/layout/main_layout.py` - 主布局管理器
- 新建 `pyutagent/ui/panels/sidebar_panel.py` - 左侧面板
- 新建 `pyutagent/ui/panels/content_panel.py` - 内容面板
- 新建 `pyutagent/ui/panels/agent_panel.py` - Agent面板
- 新建 `pyutagent/ui/panels/__init__.py`

### 1.2 可拖拽分栏系统

**功能:**
- 三栏宽度可拖拽调整
- 面板可折叠/展开
- 支持全屏模式(专注编辑或专注Chat)

**文件变更:**
- 新建 `pyutagent/ui/layout/collapsible_splitter.py`
- 修改 `pyutagent/ui/layout/main_layout.py`

---

## Phase 2: Agent面板重构 (核心)

### 2.1 多模式Agent面板

**模式切换:**
- **Chat模式**: 传统对话界面
- **Agent模式**: 显示Agent思考链、工具调用
- **Flow模式**: 可视化任务流程图

**文件变更:**
- 新建 `pyutagent/ui/agent_panel/agent_panel.py` - 主容器
- 新建 `pyutagent/ui/agent_panel/chat_mode.py` - Chat模式
- 新建 `pyutagent/ui/agent_panel/agent_mode.py` - Agent模式
- 新建 `pyutagent/ui/agent_panel/flow_mode.py` - Flow模式
- 新建 `pyutagent/ui/agent_panel/__init__.py`

### 2.2 Agent思考链可视化

**展示内容:**
```
🤔 Agent正在思考...
├─ 理解任务: 生成单元测试
├─ 分析代码: UserService.java
│  ├─ 发现: 5个public方法
│  └─ 发现: 依赖OrderRepository
├─ 制定计划:
│  1. 生成基础测试框架
│  2. 添加正常用例
│  3. 添加边界用例
│  4. 运行测试验证
├─ 执行: 生成测试框架 ✓
├─ 执行: 添加用例... (进行中)
```

**文件变更:**
- 新建 `pyutagent/ui/agent_panel/thinking_chain.py`
- 新建 `pyutagent/ui/agent_panel/tool_call_widget.py`

### 2.3 上下文管理器

**功能:**
- 显示当前上下文包含的文件
- 支持@引用添加上下文
- Token使用量显示
- 上下文优先级调整

**文件变更:**
- 新建 `pyutagent/ui/agent_panel/context_manager.py`

---

## Phase 3: 代码编辑器集成 (核心)

### 3.1 Diff预览系统

**功能:**
- 左右对比视图
- 行内高亮修改
- Accept/Reject按钮
- 批量操作

**文件变更:**
- 新建 `pyutagent/ui/editor/diff_viewer.py`
- 新建 `pyutagent/ui/editor/code_editor.py`
- 新建 `pyutagent/ui/editor/__init__.py`

### 3.2 内联编辑支持

**功能:**
- 选中代码后右键AI菜单
- 内联diff显示
- 快速Accept/Reject

**文件变更:**
- 修改 `pyutagent/ui/widgets/project_tree.py` → 重命名为 `file_tree.py` 并增强
- 新建 `pyutagent/ui/editor/inline_edit.py`

### 3.3 多语言语法高亮

**支持语言:**
- Java, Python, JavaScript/TypeScript
- Go, Rust, C/C++, C#
- HTML, CSS, SQL, YAML, JSON

**文件变更:**
- 新建 `pyutagent/ui/editor/syntax_highlighter.py`
- 新建 `pyutagent/ui/language/language_support.py`

---

## Phase 4: 终端集成

### 4.1 内嵌终端

**功能:**
- 底部/侧边可切换
- Agent可直接执行命令
- 命令确认机制
- 输出实时显示

**文件变更:**
- 新建 `pyutagent/ui/terminal/embedded_terminal.py`
- 新建 `pyutagent/ui/terminal/__init__.py`

### 4.2 命令确认UI

**安全机制:**
- 危险命令高亮提示
- 一键确认/拒绝
- 白名单配置

**文件变更:**
- 新建 `pyutagent/ui/terminal/command_confirm_dialog.py`

---

## Phase 5: 会话与历史系统

### 5.1 会话管理

**功能:**
- 多会话标签页
- 会话历史列表
- 会话导入/导出
- 会话分享

**文件变更:**
- 新建 `pyutagent/ui/session/session_manager.py`
- 新建 `pyutagent/ui/session/session_tabs.py`
- 新建 `pyutagent/ui/session/__init__.py`

### 5.2 历史记录

**功能:**
- 按项目组织历史
- 搜索历史会话
- 收藏重要会话

**文件变更:**
- 修改 `pyutagent/ui/dialogs/test_history_dialog.py` → 重命名为 `session_history_dialog.py`

---

## Phase 6: 快捷命令系统

### 6.1 斜杠命令

**命令列表:**
```
/generate - 生成代码
/test - 生成测试
/explain - 解释代码
/refactor - 重构代码
/fix - 修复错误
/doc - 生成文档
/review - 代码审查
```

**文件变更:**
- 新建 `pyutagent/ui/commands/slash_commands.py`
- 修改 `pyutagent/ui/command_palette.py` - 集成斜杠命令

### 6.2 @提及系统

**可提及对象:**
- @file - 引用文件
- @folder - 引用文件夹
- @symbol - 引用符号
- @current - 当前文件
- @selection - 选中代码

**文件变更:**
- 新建 `pyutagent/ui/commands/mention_system.py`

---

## Phase 7: 项目结构重构

### 7.1 新的UI目录结构

```
pyutagent/ui/
├── __init__.py
├── main_window.py              # 精简主窗口
├── app.py                      # 应用入口
│
├── layout/                     # 布局系统
│   ├── __init__.py
│   ├── main_layout.py          # 三栏布局
│   ├── collapsible_splitter.py # 可折叠分栏
│   └── responsive.py           # 响应式适配
│
├── panels/                     # 面板组件
│   ├── __init__.py
│   ├── sidebar_panel.py        # 左侧面板
│   ├── content_panel.py        # 内容面板
│   └── agent_panel.py          # Agent面板
│
├── agent_panel/                # Agent面板详细
│   ├── __init__.py
│   ├── agent_panel.py          # 主容器
│   ├── chat_mode.py            # Chat模式
│   ├── agent_mode.py           # Agent模式
│   ├── flow_mode.py            # Flow模式
│   ├── thinking_chain.py       # 思考链
│   ├── tool_call_widget.py     # 工具调用
│   └── context_manager.py      # 上下文管理
│
├── editor/                     # 编辑器组件
│   ├── __init__.py
│   ├── code_editor.py          # 代码编辑器
│   ├── diff_viewer.py          # Diff查看器
│   ├── inline_edit.py          # 内联编辑
│   └── syntax_highlighter.py   # 语法高亮
│
├── terminal/                   # 终端组件
│   ├── __init__.py
│   ├── embedded_terminal.py    # 内嵌终端
│   └── command_confirm_dialog.py
│
├── session/                    # 会话管理
│   ├── __init__.py
│   ├── session_manager.py
│   └── session_tabs.py
│
├── commands/                   # 命令系统
│   ├── __init__.py
│   ├── slash_commands.py       # 斜杠命令
│   └── mention_system.py       # @提及系统
│
├── language/                   # 语言支持
│   ├── __init__.py
│   ├── language_support.py     # 语言定义
│   ├── file_icons.py           # 文件图标
│   └── project_detector.py     # 项目检测
│
├── widgets/                    # 通用组件
│   ├── __init__.py
│   ├── file_tree.py            # 文件树
│   ├── search_box.py           # 搜索框
│   └── status_bar.py           # 状态栏
│
├── dialogs/                    # 对话框
│   └── (保持现有，逐步迁移)
│
├── styles/                     # 样式系统
│   └── (保持现有)
│
└── components/                 # 可复用组件
    └── (保持现有)
```

---

## Phase 8: 主题与样式升级

### 8.1 深色模式优化

**改进:**
- 更精细的颜色分级
- 代码高亮主题
- Agent状态颜色

### 8.2 动画效果

**添加动画:**
- 面板展开/折叠
- Agent思考动画
- 消息渐入
- 进度指示器

**文件变更:**
- 新建 `pyutagent/ui/styles/animations.py`
- 修改 `pyutagent/ui/styles/style_manager.py`

---

## 实施路线图

### 第一阶段: 基础架构 (Week 1-2)
- [ ] 创建新目录结构
- [ ] 实现三栏布局系统
- [ ] 提取现有组件到widgets
- [ ] MainWindow瘦身

### 第二阶段: Agent面板 (Week 3-4)
- [ ] 实现多模式Agent面板
- [ ] 思考链可视化
- [ ] 上下文管理器

### 第三阶段: 编辑器集成 (Week 5-6)
- [ ] Diff查看器
- [ ] 多语言语法高亮
- [ ] 内联编辑支持

### 第四阶段: 终端与会话 (Week 7-8)
- [ ] 内嵌终端
- [ ] 会话管理系统
- [ ] 历史记录

### 第五阶段: 命令系统 (Week 9)
- [ ] 斜杠命令
- [ ] @提及系统
- [ ] 命令面板升级

### 第六阶段: 优化与测试 (Week 10)
- [ ] 动画效果
- [ ] 主题优化
- [ ] 全面测试

**总计: 10周**

---

## 技术选型

### UI框架
- **保持PyQt6**: 成熟稳定，适合桌面应用
- **QScintilla**: 代码编辑器组件(可选)
- **Pygments**: 语法高亮

### 新增依赖
```
PyQt6-QScintilla>=2.14.0  # 代码编辑器
pygments>=2.16.0           # 语法高亮
qtawesome>=1.3.0           # 图标库
```

---

## 验收标准

1. **布局**: 三栏布局可拖拽调整，面板可折叠
2. **Agent面板**: 支持Chat/Agent/Flow三种模式
3. **编辑器**: 支持Diff预览，内联编辑
4. **终端**: Agent可执行命令，用户可确认
5. **会话**: 支持多会话，历史记录
6. **命令**: 支持斜杠命令和@提及
7. **语言**: 支持至少5种编程语言
8. **性能**: UI响应时间<50ms
9. **测试**: 新增代码覆盖率>80%

---

## 风险与缓解

| 风险 | 影响 | 缓解措施 |
|------|------|----------|
| 重构周期长 | 中 | 分阶段交付，每阶段可用 |
| 学习成本 | 中 | 参考Cursor/Claude Code设计 |
| 性能问题 | 低 | 延迟加载，虚拟列表 |
| 向后兼容 | 高 | 保持原有API，渐进迁移 |
