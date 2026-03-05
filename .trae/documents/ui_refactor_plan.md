# PyUT Agent UI 改进重构计划

## 目标
将当前专注于Java单元测试生成的UI，改进为通用的Coding Agent界面，支持多种编程语言、多种任务类型的交互式开发体验。

## 当前UI架构分析

### 现有结构
```
pyutagent/ui/
├── main_window.py          # 主窗口 (1800+ 行) - 过于臃肿
├── chat_widget.py          # 聊天组件 - 相对独立
├── command_palette.py      # 命令面板 - 功能完整
├── batch_generate_dialog.py # 批量生成对话框
├── shortcuts_manager.py    # 快捷键管理
├── log_handler.py          # 日志处理
├── styles/                 # 样式管理
│   ├── style_manager.py    # 主题管理器
│   └── themes/             # 主题文件
├── components/             # 可复用组件
│   └── notification_manager.py  # 通知管理
└── dialogs/                # 对话框集合
    ├── llm_config_dialog.py
    ├── maven_config_dialog.py
    ├── jdk_config_dialog.py
    ├── coverage_config_dialog.py
    ├── jacoco_config_dialog.py
    ├── aider_config_dialog.py
    ├── config_overview_dialog.py
    ├── project_stats_dialog.py
    ├── test_history_dialog.py
    ├── setup_wizard.py
    └── intelligence_analysis_dialog.py
```

### 主要问题

1. **MainWindow过于臃肿** (1800+行)
   - 包含AgentWorker线程类
   - 包含ProjectTreeWidget类
   - 包含ProgressWidget类
   - 所有菜单、工具栏、状态栏逻辑

2. **Java/Maven强耦合**
   - 项目树只显示Java文件
   - 硬编码Maven项目检测
   - 所有配置对话框都是Java/Maven相关

3. **缺乏通用任务支持**
   - UI只支持测试生成任务
   - 无法支持代码重构、代码解释等通用任务

4. **组件化程度不足**
   - 对话框之间缺乏统一接口
   - 样式管理虽然存在但应用不一致

---

## 重构方案

### Phase 1: 组件拆分与模块化

#### 1.1 提取独立组件

**文件变更:**
- 新建 `pyutagent/ui/widgets/project_tree.py` - 项目文件树组件
- 新建 `pyutagent/ui/widgets/progress_panel.py` - 进度面板组件
- 新建 `pyutagent/ui/widgets/agent_worker.py` - Agent工作线程
- 新建 `pyutagent/ui/widgets/__init__.py`

**MainWindow瘦身目标:** 从1800行减少到500行以内

#### 1.2 抽象通用任务接口

**文件变更:**
- 新建 `pyutagent/ui/tasks/task_interface.py` - 任务接口定义
- 新建 `pyutagent/ui/tasks/task_registry.py` - 任务注册中心
- 新建 `pyutagent/ui/tasks/__init__.py`

```python
# 任务接口示例
class TaskInterface(ABC):
    @property
    @abstractmethod
    def task_type(self) -> str: ...
    
    @abstractmethod
    def can_handle(self, file_path: str) -> bool: ...
    
    @abstractmethod
    def get_actions(self) -> List[TaskAction]: ...
```

### Phase 2: 多语言支持架构

#### 2.1 语言检测与图标系统

**文件变更:**
- 新建 `pyutagent/ui/language/language_detector.py` - 语言检测器
- 新建 `pyutagent/ui/language/file_icons.py` - 文件图标映射
- 新建 `pyutagent/ui/language/__init__.py`

**支持的文件类型:**
```python
LANGUAGE_SUPPORT = {
    'java': {'extensions': ['.java'], 'icon': '☕', 'build_tools': ['maven', 'gradle']},
    'python': {'extensions': ['.py'], 'icon': '🐍', 'build_tools': ['pip', 'poetry']},
    'javascript': {'extensions': ['.js', '.jsx'], 'icon': '📜', 'build_tools': ['npm', 'yarn']},
    'typescript': {'extensions': ['.ts', '.tsx'], 'icon': '🔷', 'build_tools': ['npm', 'yarn']},
    'go': {'extensions': ['.go'], 'icon': '🐹', 'build_tools': ['go modules']},
    'rust': {'extensions': ['.rs'], 'icon': '🦀', 'build_tools': ['cargo']},
}
```

#### 2.2 项目检测抽象

**文件变更:**
- 新建 `pyutagent/ui/project/project_detector.py` - 项目类型检测器
- 新建 `pyutagent/ui/project/base_project.py` - 项目基类
- 新建 `pyutagent/ui/project/__init__.py`

### Phase 3: 通用任务UI

#### 3.1 任务选择器

**文件变更:**
- 新建 `pyutagent/ui/tasks/task_selector.py` - 任务选择组件
- 修改 `pyutagent/ui/chat_widget.py` - 添加任务上下文

**任务类型:**
- 代码生成 (测试、实现)
- 代码重构
- 代码解释
- 代码审查
- Bug修复
- 文档生成

#### 3.2 上下文感知菜单

**文件变更:**
- 新建 `pyutagent/ui/context_menu/context_provider.py` - 上下文菜单提供者
- 修改 `pyutagent/ui/widgets/project_tree.py` - 集成动态菜单

### Phase 4: 配置系统重构

#### 4.1 通用配置框架

**文件变更:**
- 新建 `pyutagent/ui/config/config_framework.py` - 配置框架基类
- 新建 `pyutagent/ui/config/config_registry.py` - 配置注册表
- 修改现有对话框继承新框架

#### 4.2 语言特定配置

**文件变更:**
- 新建 `pyutagent/ui/config/language_configs/` - 语言配置目录
  - `java_config.py`
  - `python_config.py`
  - `javascript_config.py`

### Phase 5: 现代化UI改进

#### 5.1 响应式布局

**文件变更:**
- 修改 `pyutagent/ui/main_window.py` - 添加响应式支持
- 新建 `pyutagent/ui/layout/responsive_layout.py`

#### 5.2 动画与过渡效果

**文件变更:**
- 新建 `pyutagent/ui/animations/transitions.py` - 过渡动画
- 修改 `pyutagent/ui/styles/style_manager.py` - 添加动画支持

#### 5.3 拖拽支持

**文件变更:**
- 修改 `pyutagent/ui/widgets/project_tree.py` - 添加拖拽
- 修改 `pyutagent/ui/chat_widget.py` - 支持文件拖拽到聊天

---

## 具体实施步骤

### 步骤1: 创建widgets包并迁移组件
```
创建 pyutagent/ui/widgets/
├── __init__.py
├── project_tree.py       (从main_window.py提取)
├── progress_panel.py     (从main_window.py提取)
├── agent_worker.py       (从main_window.py提取)
└── file_preview.py       (新增文件预览组件)
```

### 步骤2: 创建language包
```
创建 pyutagent/ui/language/
├── __init__.py
├── language_detector.py
├── file_icons.py
└── syntax_highlighter.py (从chat_widget提取)
```

### 步骤3: 创建project包
```
创建 pyutagent/ui/project/
├── __init__.py
├── project_detector.py
├── base_project.py
├── java_project.py
└── python_project.py
```

### 步骤4: 创建tasks包
```
创建 pyutagent/ui/tasks/
├── __init__.py
├── task_interface.py
├── task_registry.py
├── task_selector.py
├── code_generation_task.py
├── code_refactor_task.py
└── code_explain_task.py
```

### 步骤5: 重构MainWindow
- 使用新提取的组件
- 删除内嵌类定义
- 简化到500行以内

### 步骤6: 更新对话框
- 使用config_framework基类
- 支持多语言配置

---

## 测试策略

### 单元测试
- 每个新组件独立测试
- 使用mock进行隔离测试

### 集成测试
- 组件间交互测试
- UI流程端到端测试

### 回归测试
- 确保现有功能不受影响
- 测试所有菜单和快捷键

---

## 风险评估与缓解

| 风险 | 影响 | 缓解措施 |
|------|------|----------|
| 重构引入bug | 高 | 小步重构，每次修改后测试 |
| 功能回归 | 高 | 保持原有API，渐进式迁移 |
| 开发时间超预期 | 中 | 分阶段实施，优先核心组件 |
| 性能下降 | 低 | 性能基准测试，及时优化 |

---

## 验收标准

1. MainWindow代码行数 < 500行
2. 支持至少3种编程语言的项目检测
3. 支持至少5种任务类型
4. 所有现有测试通过
5. 新增单元测试覆盖率 > 80%
6. UI响应时间 < 100ms

---

## 时间规划

| 阶段 | 预计时间 | 优先级 |
|------|----------|--------|
| Phase 1: 组件拆分 | 2-3天 | P0 |
| Phase 2: 多语言支持 | 2-3天 | P0 |
| Phase 3: 通用任务UI | 2天 | P1 |
| Phase 4: 配置系统 | 1-2天 | P1 |
| Phase 5: 现代化改进 | 2天 | P2 |
| 测试与优化 | 2天 | P0 |

**总计: 11-14天**
