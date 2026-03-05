# 增量模式UI支持优化计划

## 问题分析

### 当前状态

**已实现的功能：**
1. ✅ CLI命令支持增量模式参数
   - `generate` 命令：`-i/--incremental` 和 `--skip-analysis`
   - `generate-all` 命令：`-i/--incremental` 和 `--skip-analysis`

2. ✅ 批量生成UI支持增量模式
   - [batch_generate_dialog.py:212-222](file:///d:/opensource/github/auto-ut-agent/pyutagent/ui/batch_generate_dialog.py#L212-222) 有增量模式复选框
   - 配置会传递给 `BatchGenerator`

3. ✅ 核心逻辑完整
   - `IncrementalTestManager` 完整实现（943行）
   - `ReActAgent` 支持 `incremental_mode` 参数
   - `BatchConfig` 包含 `incremental_mode` 字段

**缺失的功能：**
1. ❌ 单文件生成UI（主窗口）不支持增量模式
   - `AgentWorker` 没有接收 `incremental_mode` 参数
   - `start_generation` 方法没有传递增量模式配置
   - UI上没有增量模式的选项

2. ❌ UI没有显示增量模式相关信息
   - 没有显示是否启用了增量模式
   - 没有显示保留了多少测试
   - 没有显示新增了多少测试
   - 没有显示增量模式的分析结果

## 解决方案

### 第一部分：单文件生成UI支持增量模式

#### 1.1 修改 AgentWorker 类
**文件**: [main_window.py:55-217](file:///d:/opensource/github/auto-ut-agent/pyutagent/ui/main_window.py#L55-217)

**修改内容**:
- 在 `__init__` 方法中添加增量模式参数：
  - `incremental_mode: bool = False`
  - `skip_test_analysis: bool = False`
- 在创建 `ReActAgent` 时传递这些参数

#### 1.2 修改 start_generation 方法
**文件**: [main_window.py:1383-1439](file:///d:/opensource/github/auto-ut-agent/pyutagent/ui/main_window.py#L1383-1439)

**修改内容**:
- 从配置中读取增量模式设置
- 传递给 `AgentWorker`
- 在日志中显示增量模式状态

#### 1.3 添加UI控件
**位置**: 主窗口的工具栏或菜单

**方案A（推荐）**: 在项目树右键菜单中添加增量模式选项
- 添加 "Generate Tests (Incremental)" 菜单项
- 添加 "Generate Tests (Skip Analysis)" 菜单项

**方案B**: 在主窗口添加配置面板
- 添加一个增量模式的复选框
- 添加跳过测试分析的复选框

#### 1.4 修改配置系统
**文件**: `pyutagent/core/config.py`

**修改内容**:
- 在 `CoverageConfig` 或新建 `GenerationConfig` 中添加：
  - `incremental_mode: bool = False`
  - `skip_test_analysis: bool = False`
- 添加对应的UI配置对话框

### 第二部分：UI显示增量模式信息

#### 2.1 进度显示增强
**文件**: [main_window.py:1456-1479](file:///d:/opensource/github/auto-ut-agent/pyutagent/ui/main_window.py#L1456-1479) (on_agent_progress方法)

**修改内容**:
- 显示增量模式状态
- 显示保留的测试数量
- 显示新增的测试数量
- 显示未覆盖代码信息

#### 2.2 完成信息增强
**文件**: [main_window.py](file:///d:/opensource/github/auto-ut-agent/pyutagent/ui/main_window.py) (on_agent_completed方法)

**修改内容**:
- 在完成消息中显示：
  - 增量模式是否启用
  - 保留了多少个通过的测试
  - 新增了多少个测试
  - 修复了多少个失败的测试

#### 2.3 ProgressWidget 增强
**文件**: `pyutagent/ui/main_window.py` (ProgressWidget类)

**修改内容**:
- 添加增量模式状态显示区域
- 添加保留测试数量显示
- 添加新增测试数量显示

### 第三部分：批量生成UI优化

#### 3.1 增量模式状态显示
**文件**: [batch_generate_dialog.py](file:///d:/opensource/github/auto-ut-agent/pyutagent/ui/batch_generate_dialog.py)

**修改内容**:
- 在进度表格中添加增量模式列
- 显示每个文件的增量生成统计

#### 3.2 完成统计增强
**修改内容**:
- 在完成统计中显示：
  - 启用增量模式的文件数
  - 总共保留的测试数
  - 总共新增的测试数

### 第四部分：配置持久化

#### 4.1 保存增量模式配置
**文件**: `pyutagent/core/config.py`

**修改内容**:
- 将增量模式设置保存到配置文件
- 下次启动时恢复设置

#### 4.2 配置对话框
**文件**: `pyutagent/ui/dialogs/coverage_config_dialog.py`

**修改内容**:
- 添加增量模式配置选项
- 添加跳过测试分析配置选项

## 实施步骤

### Phase 1: 单文件生成UI支持（高优先级）
1. 修改 `AgentWorker` 类添加增量模式参数
2. 修改 `start_generation` 方法传递参数
3. 在项目树右键菜单添加增量模式选项
4. 在配置中添加增量模式设置

### Phase 2: UI信息显示（高优先级）
1. 修改进度显示，显示增量模式信息
2. 修改完成信息，显示增量统计
3. 增强 `ProgressWidget` 显示增量信息

### Phase 3: 批量生成UI优化（中优先级）
1. 在进度表格添加增量模式列
2. 增强完成统计显示

### Phase 4: 配置持久化（中优先级）
1. 添加配置保存和加载
2. 添加配置对话框选项

## 技术细节

### AgentWorker 修改示例

```python
class AgentWorker(QThread):
    def __init__(
        self,
        llm_client: LLMClient,
        project_path: str,
        target_file: str,
        target_coverage: float = 0.8,
        max_iterations: int = 2,
        incremental_mode: bool = False,  # 新增
        skip_test_analysis: bool = False  # 新增
    ):
        super().__init__()
        self.incremental_mode = incremental_mode
        self.skip_test_analysis = skip_test_analysis
        # ... 其他初始化

    def run(self):
        # ...
        self.agent = ReActAgent(
            llm_client=self.llm_client,
            working_memory=working_memory,
            project_path=self.project_path,
            progress_callback=self._on_progress,
            model_name=self.llm_client.model,
            incremental_mode=self.incremental_mode,  # 新增
            skip_test_analysis=self.skip_test_analysis  # 新增
        )
```

### start_generation 修改示例

```python
def start_generation(self, target_file: str, incremental: bool = False, skip_analysis: bool = False):
    # ...
    settings = get_settings()
    
    # 添加增量模式日志
    if incremental:
        self.progress_widget.add_log("🔄 Incremental mode: ENABLED", "INFO")
        if skip_analysis:
            self.progress_widget.add_log("⚡ Skip test analysis: ENABLED", "INFO")
    
    self.agent_worker = AgentWorker(
        llm_client=self.llm_client,
        project_path=self.current_project,
        target_file=target_file,
        target_coverage=settings.coverage.target_coverage,
        max_iterations=settings.coverage.max_iterations,
        incremental_mode=incremental,  # 新增
        skip_test_analysis=skip_analysis  # 新增
    )
```

### 右键菜单修改示例

```python
def _show_context_menu(self, position):
    # ...
    menu = QMenu()
    
    generate_action = menu.addAction("Generate Tests")
    generate_incr_action = menu.addAction("Generate Tests (Incremental)")  # 新增
    generate_skip_action = menu.addAction("Generate Tests (Skip Analysis)")  # 新增
    
    action = menu.exec(self.viewport().mapToGlobal(position))
    
    if action == generate_action:
        self.on_generate_triggered(item)
    elif action == generate_incr_action:  # 新增
        self.on_generate_triggered(item, incremental=True)
    elif action == generate_skip_action:  # 新增
        self.on_generate_triggered(item, incremental=True, skip_analysis=True)
```

## 验收标准

1. ✅ 单文件生成UI支持增量模式选项
2. ✅ 增量模式状态在UI中清晰显示
3. ✅ 进度显示包含增量模式信息（保留测试数、新增测试数等）
4. ✅ 完成信息包含增量统计
5. ✅ 批量生成UI显示增量模式信息
6. ✅ 配置可以持久化保存
7. ✅ 所有修改保持向后兼容

## 风险评估

- **低风险**: 修改主要在UI层，不影响核心逻辑
- **向后兼容**: 所有新增参数都有默认值，不影响现有代码
- **测试覆盖**: 需要为新增UI功能添加测试

## 时间估算

- Phase 1: 2-3小时
- Phase 2: 2-3小时
- Phase 3: 1-2小时
- Phase 4: 1-2小时
- **总计**: 6-10小时
