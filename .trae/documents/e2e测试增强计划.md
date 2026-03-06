# E2E测试增强计划

## 1. 概述

### 1.1 目标

添加更多端到端（E2E）测试，确保重构没有破坏功能和主要流程，发现和解决更多bug。

### 1.2 当前测试状况

**现有E2E测试** (`tests/test_e2e.py`):
- ✅ Agent工作流测试（成功流程、暂停/恢复、终止）
- ✅ 错误恢复测试
- ✅ 检查点保存/恢复测试
- ✅ 流式生成测试
- ✅ 智能编辑器测试
- ✅ 错误学习测试
- ✅ 工具编排器测试
- ✅ 指标收集测试
- ✅ 上下文压缩测试
- ✅ 项目分析测试
- ✅ 并行恢复测试

**现有集成测试** (`tests/integration/`):
- ✅ Agent工作流集成测试
- ✅ 组件状态集成测试
- ✅ 增强Agent集成测试
- ✅ 事件总线状态存储测试
- ✅ 导入集成测试
- ✅ 增量修复器工作流测试

### 1.3 测试缺口分析

根据架构整合规范和README，以下关键流程缺少E2E测试：

| 功能模块 | 当前覆盖 | 优先级 | 风险等级 |
|---------|---------|--------|---------|
| CLI命令完整流程 | ❌ | P0 | 高 |
| GUI主要功能流程 | ❌ | P0 | 高 |
| 多文件项目测试 | ⚠️ 部分 | P0 | 高 |
| 批量生成流程 | ❌ | P0 | 高 |
| 增量模式流程 | ❌ | P1 | 中 |
| 错误场景和恢复 | ⚠️ 部分 | P0 | 高 |
| 配置管理流程 | ❌ | P1 | 中 |
| 并发和性能测试 | ❌ | P2 | 中 |
| 多LLM提供商测试 | ❌ | P2 | 低 |
| 检查点和恢复 | ⚠️ 部分 | P1 | 中 |

---

## 2. 测试用例设计

### 2.1 CLI命令E2E测试 (P0)

**文件**: `tests/e2e/test_cli_e2e.py`

#### 2.1.1 `scan` 命令测试

```python
class TestScanCommandE2E:
    """E2E tests for scan command."""
    
    def test_scan_maven_project_success(self, temp_maven_project):
        """Test scanning a valid Maven project."""
        # 创建包含多个Java文件的项目
        # 执行 scan 命令
        # 验证输出包含所有Java文件
        # 验证文件树结构正确
        
    def test_scan_non_maven_project(self, temp_dir):
        """Test scanning a non-Maven project."""
        # 创建不包含pom.xml的目录
        # 执行 scan 命令
        # 验证错误提示
        
    def test_scan_empty_project(self, temp_maven_project):
        """Test scanning an empty Maven project."""
        # 创建空的Maven项目
        # 执行 scan 命令
        # 验证输出提示无Java文件
        
    def test_scan_with_complex_structure(self, temp_maven_project):
        """Test scanning project with complex package structure."""
        # 创建多层包结构的Java项目
        # 执行 scan 命令
        # 验证所有包和类都被识别
```

#### 2.1.2 `generate` 命令测试

```python
class TestGenerateCommandE2E:
    """E2E tests for generate command."""
    
    @pytest.mark.asyncio
    async def test_generate_single_file_success(self, temp_maven_project, mock_llm_client):
        """Test generating tests for a single Java file."""
        # 创建简单的Calculator类
        # 执行 generate 命令
        # 验证测试文件生成成功
        # 验证测试文件内容正确
        # 验证编译通过
        
    @pytest.mark.asyncio
    async def test_generate_with_incremental_mode(self, temp_maven_project, mock_llm_client):
        """Test generating tests with incremental mode."""
        # 创建已有部分测试的类
        # 执行 generate -i 命令
        # 验证现有测试被保留
        # 验证新测试被添加
        
    @pytest.mark.asyncio
    async def test_generate_with_coverage_target(self, temp_maven_project, mock_llm_client):
        """Test generating tests with specific coverage target."""
        # 执行 generate --coverage-target 90
        # 验证生成的测试达到覆盖率目标
        
    @pytest.mark.asyncio
    async def test_generate_non_java_file(self, temp_dir):
        """Test generating tests for non-Java file."""
        # 创建非Java文件
        # 执行 generate 命令
        # 验证错误提示
        
    @pytest.mark.asyncio
    async def test_generate_with_compilation_error(self, temp_maven_project, mock_llm_client):
        """Test handling compilation errors during generation."""
        # Mock LLM返回有编译错误的测试代码
        # 执行 generate 命令
        # 验证错误恢复流程被触发
        # 验证最终生成成功的测试
```

#### 2.1.3 `generate-all` 命令测试

```python
class TestGenerateAllCommandE2E:
    """E2E tests for generate-all command."""
    
    @pytest.mark.asyncio
    async def test_generate_all_sequential(self, temp_multi_file_project, mock_llm_client):
        """Test batch generation in sequential mode."""
        # 创建包含10个Java文件的项目
        # 执行 generate-all -p 1
        # 验证所有文件都被处理
        # 验证成功率统计正确
        
    @pytest.mark.asyncio
    async def test_generate_all_parallel(self, temp_multi_file_project, mock_llm_client):
        """Test batch generation in parallel mode."""
        # 创建包含10个Java文件的项目
        # 执行 generate-all -p 4
        # 验证并行执行正确
        # 验证性能提升
        
    @pytest.mark.asyncio
    async def test_generate_all_with_defer_compilation(self, temp_multi_file_project, mock_llm_client):
        """Test two-phase batch generation."""
        # 执行 generate-all --defer-compilation
        # 验证先生成所有测试
        # 验证最后统一编译
        
    @pytest.mark.asyncio
    async def test_generate_all_continue_on_error(self, temp_multi_file_project, mock_llm_client):
        """Test batch generation with continue-on-error."""
        # Mock部分文件生成失败
        # 执行 generate-all --continue-on-error
        # 验证失败不影响其他文件
        
    @pytest.mark.asyncio
    async def test_generate_all_stop_on_error(self, temp_multi_file_project, mock_llm_client):
        """Test batch generation with stop-on-error."""
        # Mock第一个文件生成失败
        # 执行 generate-all --stop-on-error
        # 验证后续文件不被处理
```

#### 2.1.4 `config` 命令测试

```python
class TestConfigCommandE2E:
    """E2E tests for config command."""
    
    def test_config_llm_list(self, temp_config):
        """Test listing LLM configurations."""
        # 执行 config llm list
        # 验证输出包含所有配置
        
    def test_config_llm_set_default(self, temp_config):
        """Test setting default LLM configuration."""
        # 执行 config llm set-default <id>
        # 验证默认配置被更新
        
    def test_config_maven_show(self, temp_config):
        """Test showing Maven configuration."""
        # 执行 config maven show
        # 验证输出包含Maven路径和版本
        
    def test_config_coverage_show(self, temp_config):
        """Test showing coverage configuration."""
        # 执行 config coverage show
        # 验证输出包含JaCoCo配置
```

---

### 2.2 GUI主要功能E2E测试 (P0)

**文件**: `tests/e2e/test_gui_e2e.py`

#### 2.2.1 主窗口测试

```python
class TestMainWindowE2E:
    """E2E tests for main window."""
    
    def test_window_initialization(self, qtbot, main_window):
        """Test main window initializes correctly."""
        # 验证窗口标题
        # 验证菜单栏存在
        # 验证工具栏存在
        # 验证文件树存在
        # 验证聊天区域存在
        
    def test_open_project(self, qtbot, main_window, temp_maven_project):
        """Test opening a Maven project."""
        # 触发打开项目操作
        # 验证文件树显示项目结构
        # 验证项目路径被记录
        
    def test_open_non_maven_project(self, qtbot, main_window, temp_dir):
        """Test opening a non-Maven project."""
        # 触发打开项目操作
        # 验证错误提示
        
    def test_load_last_project(self, qtbot, main_window, temp_maven_project):
        """Test auto-loading last project."""
        # 打开项目并关闭窗口
        # 重新打开窗口
        # 验证上次项目自动加载
```

#### 2.2.2 LLM配置对话框测试

```python
class TestLLMConfigDialogE2E:
    """E2E tests for LLM configuration dialog."""
    
    def test_open_llm_config_dialog(self, qtbot, main_window):
        """Test opening LLM config dialog."""
        # 触发打开对话框
        # 验证对话框显示
        
    def test_add_openai_config(self, qtbot, main_window):
        """Test adding OpenAI configuration."""
        # 打开对话框
        # 填写OpenAI配置
        # 保存配置
        # 验证配置被保存
        
    def test_test_connection_success(self, qtbot, main_window, mock_llm_server):
        """Test testing LLM connection successfully."""
        # 打开对话框
        # 填写配置
        # 点击测试连接
        # 验证连接成功提示
        
    def test_test_connection_failure(self, qtbot, main_window):
        """Test testing LLM connection with failure."""
        # 打开对话框
        # 填写错误配置
        # 点击测试连接
        # 验证错误提示
```

#### 2.2.3 测试生成流程测试

```python
class TestTestGenerationE2E:
    """E2E tests for test generation flow."""
    
    @pytest.mark.asyncio
    async def test_generate_tests_from_chat(self, qtbot, main_window, temp_maven_project, mock_llm_client):
        """Test generating tests through chat interface."""
        # 打开项目
        # 选择Java文件
        # 在聊天区域输入"生成测试"
        # 验证测试生成流程启动
        # 验证进度显示
        # 验证测试文件生成
        
    @pytest.mark.asyncio
    async def test_pause_resume_generation(self, qtbot, main_window, temp_maven_project, mock_llm_client):
        """Test pausing and resuming test generation."""
        # 开始生成测试
        # 点击暂停按钮
        # 验证生成暂停
        # 点击继续按钮
        # 验证生成继续
        
    @pytest.mark.asyncio
    async def test_terminate_generation(self, qtbot, main_window, temp_maven_project, mock_llm_client):
        """Test terminating test generation."""
        # 开始生成测试
        # 点击终止按钮
        # 验证生成终止
        # 验证状态恢复
        
    @pytest.mark.asyncio
    async def test_view_coverage_report(self, qtbot, main_window, temp_maven_project, mock_llm_client):
        """Test viewing coverage report."""
        # 生成测试
        # 查看覆盖率报告
        # 验证覆盖率显示正确
```

---

### 2.3 多文件项目E2E测试 (P0)

**文件**: `tests/e2e/test_multi_file_e2e.py`

#### 2.3.1 多文件项目测试

```python
class TestMultiFileProjectE2E:
    """E2E tests for multi-file projects."""
    
    @pytest.mark.asyncio
    async def test_project_with_dependencies(self, temp_multi_file_project, mock_llm_client):
        """Test project with inter-file dependencies."""
        # 创建有依赖关系的多个类
        # Service -> Repository -> Model
        # 生成Service的测试
        # 验证依赖被正确Mock
        
    @pytest.mark.asyncio
    async def test_project_with_external_dependencies(self, temp_multi_file_project, mock_llm_client):
        """Test project with external Maven dependencies."""
        # 创建使用外部库的类
        # 如使用Jackson、Apache Commons等
        # 生成测试
        # 验证依赖被正确处理
        
    @pytest.mark.asyncio
    async def test_large_project_performance(self, temp_large_project, mock_llm_client):
        """Test performance with large project."""
        # 创建包含100个Java文件的项目
        # 执行批量生成
        # 验证内存使用合理
        # 验证执行时间可接受
        
    @pytest.mark.asyncio
    async def test_project_with_complex_hierarchy(self, temp_multi_file_project, mock_llm_client):
        """Test project with complex class hierarchy."""
        # 创建多层继承的类结构
        # BaseClass -> AbstractClass -> ConcreteClass
        # 生成测试
        # 验证继承关系被正确处理
```

---

### 2.4 错误场景和恢复E2E测试 (P0)

**文件**: `tests/e2e/test_error_recovery_e2e.py`

#### 2.4.1 编译错误恢复测试

```python
class TestCompilationErrorRecoveryE2E:
    """E2E tests for compilation error recovery."""
    
    @pytest.mark.asyncio
    async def test_missing_import_recovery(self, temp_maven_project, mock_llm_client):
        """Test recovering from missing import errors."""
        # Mock LLM返回缺少import的测试代码
        # 触发编译
        # 验证错误被识别
        # 验证自动添加import
        # 验证编译成功
        
    @pytest.mark.asyncio
    async def test_missing_dependency_recovery(self, temp_maven_project, mock_llm_client):
        """Test recovering from missing dependency errors."""
        # Mock LLM返回使用外部库的测试代码
        # 触发编译
        # 验证依赖被识别
        # 验证pom.xml被更新
        # 验证依赖被下载
        # 验证编译成功
        
    @pytest.mark.asyncio
    async def test_syntax_error_recovery(self, temp_maven_project, mock_llm_client):
        """Test recovering from syntax errors."""
        # Mock LLM返回有语法错误的测试代码
        # 触发编译
        # 验证错误被识别
        # 验证LLM重新生成
        # 验证编译成功
        
    @pytest.mark.asyncio
    async def test_type_mismatch_recovery(self, temp_maven_project, mock_llm_client):
        """Test recovering from type mismatch errors."""
        # Mock LLM返回类型不匹配的测试代码
        # 触发编译
        # 验证错误被识别
        # 验证修复建议
        # 验证编译成功
```

#### 2.4.2 测试失败恢复测试

```python
class TestFailureRecoveryE2E:
    """E2E tests for test failure recovery."""
    
    @pytest.mark.asyncio
    async def test_assertion_failure_recovery(self, temp_maven_project, mock_llm_client):
        """Test recovering from assertion failures."""
        # Mock LLM返回断言错误的测试代码
        # 执行测试
        # 验证失败被识别
        # 验证LLM分析失败原因
        # 验证测试被修复
        
    @pytest.mark.asyncio
    async def test_null_pointer_recovery(self, temp_maven_project, mock_llm_client):
        """Test recovering from null pointer exceptions."""
        # Mock LLM返回导致NPE的测试代码
        # 执行测试
        # 验证异常被识别
        # 验证修复方案
        # 验证测试通过
        
    @pytest.mark.asyncio
    async def test_timeout_recovery(self, temp_maven_project, mock_llm_client):
        """Test recovering from test timeouts."""
        # Mock LLM返回导致超时的测试代码
        # 执行测试
        # 验证超时被识别
        # 验证优化建议
        # 验证测试通过
```

#### 2.4.3 LLM错误恢复测试

```python
class TestLLMErrorRecoveryE2E:
    """E2E tests for LLM error recovery."""
    
    @pytest.mark.asyncio
    async def test_api_timeout_recovery(self, temp_maven_project, mock_llm_client):
        """Test recovering from LLM API timeouts."""
        # Mock LLM API超时
        # 触发生成
        # 验证重试机制
        # 验证最终成功
        
    @pytest.mark.asyncio
    async def test_rate_limit_recovery(self, temp_maven_project, mock_llm_client):
        """Test recovering from rate limit errors."""
        # Mock LLM API限流
        # 触发生成
        # 验证退避重试
        # 验证最终成功
        
    @pytest.mark.asyncio
    async def test_invalid_response_recovery(self, temp_maven_project, mock_llm_client):
        """Test recovering from invalid LLM responses."""
        # Mock LLM返回无效响应
        # 触发生成
        # 验证错误识别
        # 验证重新请求
        # 验证最终成功
```

---

### 2.5 增量模式E2E测试 (P1)

**文件**: `tests/e2e/test_incremental_e2e.py`

#### 2.5.1 增量模式测试

```python
class TestIncrementalModeE2E:
    """E2E tests for incremental mode."""
    
    @pytest.mark.asyncio
    async def test_preserve_passing_tests(self, temp_maven_project, mock_llm_client):
        """Test preserving existing passing tests."""
        # 创建已有测试文件的类
        # 部分测试通过，部分失败
        # 执行增量生成
        # 验证通过的测试被保留
        # 验证失败的测试被修复
        
    @pytest.mark.asyncio
    async def test_add_new_tests(self, temp_maven_project, mock_llm_client):
        """Test adding new tests in incremental mode."""
        # 创建已有测试文件的类
        # 添加新方法到类
        # 执行增量生成
        # 验证现有测试保留
        # 验证新方法的测试被添加
        
    @pytest.mark.asyncio
    async def test_merge_test_strategies(self, temp_maven_project, mock_llm_client):
        """Test merging different test strategies."""
        # 创建已有测试的类
        # 执行增量生成
        # 验证测试策略合并
        # 验证覆盖率提升
        
    @pytest.mark.asyncio
    async def test_incremental_with_refactoring(self, temp_maven_project, mock_llm_client):
        """Test incremental mode with code refactoring."""
        # 创建已有测试的类
        # 重构类代码
        # 执行增量生成
        # 验证测试适应重构
```

---

### 2.6 配置管理E2E测试 (P1)

**文件**: `tests/e2e/test_config_e2e.py`

#### 2.6.1 配置持久化测试

```python
class TestConfigPersistenceE2E:
    """E2E tests for configuration persistence."""
    
    def test_llm_config_persistence(self, temp_config):
        """Test LLM configuration persistence."""
        # 创建LLM配置
        # 保存配置
        # 重新加载
        # 验证配置一致
        
    def test_maven_config_persistence(self, temp_config):
        """Test Maven configuration persistence."""
        # 配置Maven路径
        # 保存配置
        # 重新加载
        # 验证配置一致
        
    def test_coverage_config_persistence(self, temp_config):
        """Test coverage configuration persistence."""
        # 配置JaCoCo参数
        # 保存配置
        # 重新加载
        # 验证配置一致
        
    def test_config_migration(self, temp_old_config):
        """Test configuration migration from old version."""
        # 创建旧版本配置
        # 加载配置
        # 验证自动迁移
        # 验证配置正确
```

---

### 2.7 并发和性能E2E测试 (P2)

**文件**: `tests/e2e/test_performance_e2e.py`

#### 2.7.1 并发测试

```python
class TestConcurrencyE2E:
    """E2E tests for concurrent operations."""
    
    @pytest.mark.asyncio
    async def test_parallel_file_generation(self, temp_large_project, mock_llm_client):
        """Test parallel test generation for multiple files."""
        # 创建包含50个文件的项目
        # 执行并行生成（4个worker）
        # 验证所有文件被处理
        # 验证无竞态条件
        # 验证性能提升
        
    @pytest.mark.asyncio
    async def test_concurrent_user_operations(self, qtbot, main_window, temp_maven_project):
        """Test concurrent user operations."""
        # 同时触发多个操作
        # 验证操作队列正确
        # 验证UI响应性
        
    @pytest.mark.asyncio
    async def test_cache_thread_safety(self, temp_maven_project, mock_llm_client):
        """Test cache thread safety."""
        # 并发访问缓存
        # 验证无数据竞争
        # 验证缓存一致性
```

#### 2.7.2 性能基准测试

```python
class TestPerformanceBenchmarkE2E:
    """E2E performance benchmark tests."""
    
    @pytest.mark.asyncio
    async def test_generation_speed_benchmark(self, temp_maven_project, mock_llm_client):
        """Benchmark test generation speed."""
        # 测量单个文件生成时间
        # 验证在合理范围内
        
    @pytest.mark.asyncio
    async def test_memory_usage_benchmark(self, temp_large_project, mock_llm_client):
        """Benchmark memory usage."""
        # 监控内存使用
        # 执行批量生成
        # 验证内存使用合理
        
    @pytest.mark.asyncio
    async def test_cache_effectiveness_benchmark(self, temp_maven_project, mock_llm_client):
        """Benchmark cache effectiveness."""
        # 执行相同操作两次
        # 验证第二次更快
        # 验证缓存命中率
```

---

## 3. 实施计划

### 3.1 阶段一：P0优先级测试（第1-2周）

**目标**: 覆盖核心功能和主要流程

**任务列表**:

1. **CLI命令E2E测试** (3天)
   - [ ] 创建 `tests/e2e/test_cli_e2e.py`
   - [ ] 实现 `scan` 命令测试（4个测试用例）
   - [ ] 实现 `generate` 命令测试（5个测试用例）
   - [ ] 实现 `generate-all` 命令测试（5个测试用例）
   - [ ] 实现 `config` 命令测试（4个测试用例）
   - [ ] 运行测试并修复发现的问题

2. **GUI主要功能E2E测试** (3天)
   - [ ] 创建 `tests/e2e/test_gui_e2e.py`
   - [ ] 实现主窗口测试（4个测试用例）
   - [ ] 实现LLM配置对话框测试（4个测试用例）
   - [ ] 实现测试生成流程测试（4个测试用例）
   - [ ] 运行测试并修复发现的问题

3. **多文件项目E2E测试** (2天)
   - [ ] 创建 `tests/e2e/test_multi_file_e2e.py`
   - [ ] 实现多文件项目测试（4个测试用例）
   - [ ] 运行测试并修复发现的问题

4. **错误场景和恢复E2E测试** (2天)
   - [ ] 创建 `tests/e2e/test_error_recovery_e2e.py`
   - [ ] 实现编译错误恢复测试（4个测试用例）
   - [ ] 实现测试失败恢复测试（3个测试用例）
   - [ ] 实现LLM错误恢复测试（3个测试用例）
   - [ ] 运行测试并修复发现的问题

**预期成果**:
- 新增约40个E2E测试用例
- 覆盖核心功能和主要流程
- 发现并修复关键bug

### 3.2 阶段二：P1优先级测试（第3周）

**目标**: 覆盖重要功能和边界场景

**任务列表**:

1. **增量模式E2E测试** (2天)
   - [ ] 创建 `tests/e2e/test_incremental_e2e.py`
   - [ ] 实现增量模式测试（4个测试用例）
   - [ ] 运行测试并修复发现的问题

2. **配置管理E2E测试** (2天)
   - [ ] 创建 `tests/e2e/test_config_e2e.py`
   - [ ] 实现配置持久化测试（4个测试用例）
   - [ ] 运行测试并修复发现的问题

3. **补充和优化** (1天)
   - [ ] 补充遗漏的测试用例
   - [ ] 优化测试性能
   - [ ] 完善测试文档

**预期成果**:
- 新增约8个E2E测试用例
- 覆盖重要功能和边界场景
- 提高测试稳定性

### 3.3 阶段三：P2优先级测试（第4周）

**目标**: 覆盖性能和并发场景

**任务列表**:

1. **并发和性能E2E测试** (3天)
   - [ ] 创建 `tests/e2e/test_performance_e2e.py`
   - [ ] 实现并发测试（3个测试用例）
   - [ ] 实现性能基准测试（3个测试用例）
   - [ ] 运行测试并修复发现的问题

2. **测试完善和文档** (2天)
   - [ ] 补充遗漏的测试用例
   - [ ] 编写测试指南文档
   - [ ] 更新README测试统计

**预期成果**:
- 新增约6个E2E测试用例
- 覆盖性能和并发场景
- 完善测试文档

---

## 4. 测试基础设施

### 4.1 测试固件（Fixtures）

**文件**: `tests/e2e/conftest.py`

需要创建以下测试固件：

```python
@pytest.fixture
def temp_maven_project():
    """创建临时Maven项目"""
    # 创建基本项目结构
    # 包含pom.xml、src/main/java、src/test/java
    # 返回项目路径
    
@pytest.fixture
def temp_multi_file_project():
    """创建多文件Maven项目"""
    # 创建包含多个Java文件的项目
    # 包含依赖关系
    
@pytest.fixture
def temp_large_project():
    """创建大型Maven项目"""
    # 创建包含100+Java文件的项目
    # 用于性能测试
    
@pytest.fixture
def mock_llm_client():
    """创建Mock LLM客户端"""
    # Mock LLM API响应
    # 支持多种响应场景
    
@pytest.fixture
def mock_llm_server():
    """创建Mock LLM服务器"""
    # 启动本地Mock服务器
    # 模拟真实API行为
    
@pytest.fixture
def temp_config():
    """创建临时配置"""
    # 创建临时配置文件
    # 隔离测试环境
    
@pytest.fixture
def main_window(qtbot):
    """创建主窗口实例"""
    # 初始化MainWindow
    # 返回窗口实例
```

### 4.2 测试工具函数

**文件**: `tests/e2e/utils.py`

```python
def create_java_class(package: str, class_name: str, methods: List[str]) -> str:
    """创建Java类代码"""
    
def create_test_class(class_name: str, test_methods: List[str]) -> str:
    """创建测试类代码"""
    
def run_maven_command(project_path: str, command: str) -> Tuple[int, str, str]:
    """执行Maven命令"""
    
def check_compilation(project_path: str) -> bool:
    """检查项目编译"""
    
def check_test_execution(project_path: str) -> Tuple[bool, float]:
    """检查测试执行"""
    
def get_coverage_report(project_path: str) -> Dict:
    """获取覆盖率报告"""
```

### 4.3 Mock策略

**LLM Mock**:
- 使用 `unittest.mock.AsyncMock` mock LLM客户端
- 准备多种响应模板（成功、失败、部分成功）
- 支持流式响应mock

**Maven Mock**:
- 使用真实Maven执行（确保环境可用）
- Mock耗时操作（如依赖下载）
- 使用本地Maven仓库缓存

**文件系统Mock**:
- 使用 `tempfile.TemporaryDirectory` 创建临时目录
- 不mock文件系统操作（测试真实IO）

---

## 5. 测试执行策略

### 5.1 本地开发测试

```bash
# 运行所有E2E测试
pytest tests/e2e/ -v

# 运行特定测试文件
pytest tests/e2e/test_cli_e2e.py -v

# 运行特定测试类
pytest tests/e2e/test_cli_e2e.py::TestGenerateCommandE2E -v

# 运行特定测试用例
pytest tests/e2e/test_cli_e2e.py::TestGenerateCommandE2E::test_generate_single_file_success -v

# 带覆盖率报告
pytest tests/e2e/ --cov=pyutagent --cov-report=html

# 并行执行
pytest tests/e2e/ -n 4
```

### 5.2 CI/CD集成

**文件**: `.github/workflows/e2e_tests.yml`

```yaml
name: E2E Tests

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]

jobs:
  e2e-tests:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.11'
    
    - name: Set up Java
      uses: actions/setup-java@v3
      with:
        java-version: '11'
        distribution: 'temurin'
    
    - name: Cache Maven packages
      uses: actions/cache@v3
      with:
        path: ~/.m2
        key: ${{ runner.os }}-m2-${{ hashFiles('**/pom.xml') }}
    
    - name: Install dependencies
      run: |
        pip install -e ".[dev]"
    
    - name: Run E2E tests
      run: |
        pytest tests/e2e/ -v --tb=short
    
    - name: Upload test results
      if: always()
      uses: actions/upload-artifact@v3
      with:
        name: e2e-test-results
        path: test-results/
```

### 5.3 测试报告

使用 `pytest-html` 生成HTML测试报告：

```bash
pytest tests/e2e/ --html=report.html --self-contained-html
```

---

## 6. 风险和缓解措施

| 风险 | 影响 | 概率 | 缓解措施 |
|------|------|------|----------|
| 测试环境依赖（Maven、Java） | 高 | 中 | 使用Docker容器统一环境，添加环境检查脚本 |
| LLM API调用成本 | 中 | 高 | 使用Mock LLM客户端，仅在必要时调用真实API |
| 测试执行时间长 | 中 | 中 | 优化测试性能，使用并行执行，缓存测试固件 |
| GUI测试不稳定 | 高 | 中 | 使用qtbot提供的等待机制，添加重试逻辑 |
| 测试数据管理复杂 | 中 | 低 | 使用工厂模式创建测试数据，统一管理测试固件 |

---

## 7. 成功指标

### 7.1 测试覆盖率目标

- **E2E测试用例数量**: 50+ 新增用例
- **核心流程覆盖**: 100%
- **主要功能覆盖**: 90%+
- **边界场景覆盖**: 80%+

### 7.2 质量指标

- **测试通过率**: 95%+（允许少量不稳定测试）
- **Bug发现数量**: 预期发现10+个bug
- **Bug修复时间**: P0 bug在24小时内修复
- **测试执行时间**: 全部E2E测试在10分钟内完成

### 7.3 文档完整性

- **测试指南**: 完整的E2E测试执行指南
- **测试固件文档**: 所有测试固件的使用说明
- **最佳实践**: E2E测试编写最佳实践文档

---

## 8. 后续改进

### 8.1 持续集成优化

- 添加测试结果趋势分析
- 集成代码覆盖率趋势
- 添加性能回归检测

### 8.2 测试数据管理

- 建立测试数据仓库
- 实现测试数据版本管理
- 添加测试数据生成工具

### 8.3 测试自动化

- 自动生成测试报告
- 自动标记不稳定测试
- 自动修复简单测试失败

---

## 9. 参考资料

- [pytest文档](https://docs.pytest.org/)
- [pytest-asyncio文档](https://pytest-asyncio.readthedocs.io/)
- [pytest-qt文档](https://pytest-qt.readthedocs.io/)
- [项目架构文档](../ARCHITECTURE.md)
- [架构整合规范](../.trae/documents/架构整合规范/spec.md)
