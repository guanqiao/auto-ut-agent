## 目标
将新实现的三个Aider高级功能完美集成到现有UI和配置系统中：
1. **多编辑格式支持** (diff/udiff/whole/diff-fenced)
2. **Architect/Editor双模型模式** (高质量低成本编辑)
3. **多文件批量编辑** (依赖分析和拓扑排序)

## 实施计划

### 1. 扩展LLM配置模型
**文件**: `pyutagent/llm/config.py`
- 添加 `AiderConfig` 类，包含所有新配置选项
- 在 `LLMConfig` 中添加 `aider` 字段

### 2. 创建Aider专用配置对话框
**文件**: `pyutagent/ui/dialogs/aider_config_dialog.py` (新建)
- 创建独立的Aider配置对话框
- 包含以下配置项：
  - Architect/Editor双模型开关
  - Architect模型选择
  - Editor模型选择
  - 多文件编辑开关
  - 编辑格式选择 (diff/udiff/whole/diff-fenced/auto)
  - 自动检测格式开关

### 3. 更新LLM配置对话框
**文件**: `pyutagent/ui/dialogs/llm_config_dialog.py`
- 添加Tab页或按钮跳转到Aider配置
- 在现有对话框中添加Aider配置入口

### 4. 更新主窗口集成
**文件**: `pyutagent/ui/main_window.py`
- 在菜单栏添加"Aider配置"菜单项
- 更新AgentWorker以传递Aider配置
- 在进度显示中显示当前使用的Aider策略

### 5. 更新应用配置
**文件**: `pyutagent/config.py`
- 添加Aider相关配置项到Settings类
- 支持从环境变量读取配置

### 6. 更新ReAct Agent集成
**文件**: `pyutagent/agent/react_agent.py`
- 在创建AiderCodeFixer时传入AiderConfig
- 根据配置选择不同的修复策略

## UI设计细节

### Aider配置对话框布局
```
┌─────────────────────────────────────────┐
│ Aider 高级配置                           │
├─────────────────────────────────────────┤
│ [✓] 启用 Architect/Editor 双模型模式      │
│   Architect 模型: [gpt-4        ▼]       │
│   Editor 模型:    [gpt-3.5-turbo ▼]      │
│                                         │
│ [✓] 启用多文件批量编辑                    │
│                                         │
│ 编辑格式: [自动检测 ▼]                    │
│   (diff / udiff / whole / diff-fenced)  │
│                                         │
│ [✓] 自动检测模型最优格式                  │
│                                         │
│ [💾 保存]  [❌ 取消]                      │
└─────────────────────────────────────────┘
```

### 主窗口更新
1. 在"设置"菜单下添加"Aider高级配置..."选项
2. 在进度面板显示当前使用的编辑策略
3. 在日志中记录Aider配置信息

## 集成流程
1. 用户通过菜单打开Aider配置对话框
2. 配置保存到LLMConfig的aider字段
3. MainWindow将配置传递给AgentWorker
4. AgentWorker在创建ReActAgent时传入配置
5. ReActAgent创建AiderCodeFixer时使用AiderConfig
6. AiderCodeFixer根据配置自动选择最优策略

## 配置持久化
- 配置保存到JSON文件 (~/.pyutagent/aider_config.json)
- 支持导出/导入配置
- 提供默认配置模板

请确认这个方案后，我将立即开始实施具体的代码修改。