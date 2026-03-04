# 增量模式功能开发计划

## 1. 功能概述

### 1.1 目标
为 GUI 和 CLI 添加增量测试生成模式，在重新生成测试时：
- 保留已存在的通过测试
- 只新增测试覆盖新的代码逻辑
- 修复失败的测试

### 1.2 核心价值
- 减少重复工作，保留已验证的测试
- 提高生成效率，只关注增量部分
- 降低引入新bug的风险

---

## 2. 现有架构分析

### 2.1 关键组件
| 组件 | 文件路径 | 现状 |
|-----|---------|------|
| GUI入口 | `pyutagent/main.py` | PyQt6 + qasync |
| CLI入口 | `pyutagent/cli/main.py` | Click + Rich |
| 单文件生成 | `pyutagent/agent/react_agent.py` | ReActAgent → FeedbackLoopExecutor |
| 批量生成 | `pyutagent/services/batch_generator.py` | BatchGenerator + BatchConfig |
| 步骤执行 | `pyutagent/agent/components/execution_steps.py` | StepExecutor |
| 部分成功处理 | `pyutagent/agent/partial_success_handler.py` | **已存在但未集成** |
| 测试代码解析 | `pyutagent/agent/partial_success_handler.py::TestCodeParser` | **已存在** |

### 2.2 现有可复用组件
1. **PartialSuccessHandler** - 分析测试结果，识别通过/失败的测试
2. **TestCodeParser** - 解析测试方法，提取方法信息
3. **IncrementalFixResult** - 增量修复结果数据结构

### 2.3 当前缺失
- 增量模式配置选项
- 增量生成流程集成
- 增量模式提示词模板
- 已有测试检测和分析逻辑

---

## 3. 技术方案

### 3.1 新增配置项

```python
# BatchConfig 新增
incremental_mode: bool = False  # 是否启用增量模式

# 单文件生成配置
class IncrementalConfig:
    enabled: bool = False
    preserve_passing_tests: bool = True
    analyze_existing_coverage: bool = True
    max_preserved_tests: int = 50  # 最大保留测试数
```

### 3.2 增量生成流程

```
增量模式流程:
1. 检测是否存在测试文件
   ├─ 不存在 → 正常生成流程
   └─ 存在 → 进入增量流程

2. 增量流程:
   a. 解析已有测试文件 (TestCodeParser)
   b. 运行已有测试，收集结果
   c. 分析测试结果 (PartialSuccessHandler)
      ├─ 全部通过 → 分析覆盖率
      │   └─ 覆盖率达标 → 跳过生成
      │   └─ 覆盖率不足 → 只生成补充测试
      └─ 部分失败 → 保留通过测试 + 修复失败测试

3. 构建增量提示词
   - 包含已有测试代码
   - 标注通过的测试（需保留）
   - 标注失败的测试（需修复）
   - 包含未覆盖的代码逻辑

4. LLM 生成增量测试

5. 合并测试代码
   - 保留通过的测试
   - 添加新生成的测试
   - 替换修复的测试
```

### 3.3 新增组件

#### 3.3.1 IncrementalTestManager
位置: `pyutagent/agent/incremental_manager.py`

职责:
- 检测已有测试文件
- 分析已有测试状态
- 构建增量生成上下文
- 合并测试代码

```python
class IncrementalTestManager:
    def detect_existing_test(self, target_file: str) -> Optional[str]
    def analyze_existing_tests(self, test_file: str) -> ExistingTestAnalysis
    def build_incremental_context(self, analysis: ExistingTestAnalysis) -> IncrementalContext
    def merge_tests(self, preserved: List[str], new: List[str]) -> str
```

#### 3.3.2 增量提示词模板
位置: `pyutagent/agent/prompt_builder.py` (扩展)

新增方法:
- `build_incremental_test_prompt()` - 增量生成提示词
- `build_test_fix_prompt()` - 测试修复提示词

---

## 4. 详细任务分解

### Phase 1: 核心组件开发

#### Task 1.1: 创建 IncrementalTestManager
- 文件: `pyutagent/agent/incremental_manager.py`
- 内容:
  - `ExistingTestAnalysis` 数据类
  - `IncrementalContext` 数据类
  - `IncrementalTestManager` 类
- 依赖: `PartialSuccessHandler`, `TestCodeParser`

#### Task 1.2: 扩展配置
- 文件: `pyutagent/services/batch_generator.py`
- 修改: `BatchConfig` 添加 `incremental_mode`
- 文件: `pyutagent/core/config.py`
- 修改: 添加增量模式相关配置

#### Task 1.3: 扩展 PromptBuilder
- 文件: `pyutagent/agent/prompt_builder.py`
- 新增: `build_incremental_test_prompt()` 方法
- 新增: `build_test_fix_prompt()` 方法

### Phase 2: 集成到生成流程

#### Task 2.1: 修改 StepExecutor
- 文件: `pyutagent/agent/components/execution_steps.py`
- 新增: `generate_incremental_tests()` 方法
- 修改: `generate_initial_tests()` 支持增量模式

#### Task 2.2: 修改 FeedbackLoopExecutor
- 文件: `pyutagent/agent/components/feedback_loop.py`
- 修改: `_phase_generate_initial_tests()` 检测增量模式
- 新增: `_phase_incremental_generation()` 方法

#### Task 2.3: 修改 ReActAgent
- 文件: `pyutagent/agent/react_agent.py`
- 新增: `incremental_mode` 属性
- 新增: `incremental_manager` 组件

### Phase 3: CLI 集成

#### Task 3.1: 单文件生成命令
- 文件: `pyutagent/cli/commands/generate.py`
- 新增: `--incremental` / `-i` 选项
- 新增: `--preserve-passing` 选项

#### Task 3.2: 批量生成命令
- 文件: `pyutagent/cli/commands/generate_all.py`
- 新增: `--incremental` / `-i` 选项
- 修改: 传递增量配置到 BatchGenerator

### Phase 4: GUI 集成

#### Task 4.1: 批量生成对话框
- 文件: `pyutagent/ui/batch_generate_dialog.py`
- 新增: 增量模式复选框
- 新增: 增量模式说明标签
- 修改: `BatchGenerateWorker` 支持增量配置

#### Task 4.2: 主窗口集成
- 文件: `pyutagent/ui/main_window.py`
- 新增: 单文件生成的增量模式选项
- 修改: 测试生成对话框/面板

### Phase 5: 测试与文档

#### Task 5.1: 单元测试
- 文件: `tests/unit/agent/test_incremental_manager.py`
- 测试: IncrementalTestManager 各方法

#### Task 5.2: 集成测试
- 文件: `tests/integration/test_incremental_generation.py`
- 测试: 完整增量生成流程

---

## 5. 数据结构设计

### 5.1 ExistingTestAnalysis
```python
@dataclass
class ExistingTestAnalysis:
    test_file_path: str
    exists: bool
    test_methods: List[TestMethodInfo]
    passing_tests: List[str]
    failing_tests: List[str]
    current_coverage: float
    uncovered_lines: List[int]
    uncovered_branches: List[int]
    last_run_time: Optional[datetime]
```

### 5.2 IncrementalContext
```python
@dataclass
class IncrementalContext:
    existing_tests_code: str
    preserved_test_names: List[str]
    tests_to_fix: List[TestMethodResult]
    new_code_to_cover: List[Dict[str, Any]]  # 新增/修改的方法
    uncovered_code: List[Dict[str, Any]]
    target_coverage_gap: float
```

### 5.3 IncrementalConfig
```python
@dataclass
class IncrementalConfig:
    enabled: bool = False
    preserve_passing_tests: bool = True
    analyze_existing_coverage: bool = True
    max_preserved_tests: int = 50
    min_tests_to_preserve: int = 1  # 最少保留测试数才启用增量
    force_regenerate_failed: bool = True
```

---

## 6. 提示词模板设计

### 6.1 增量生成提示词
```
You are generating INCREMENTAL unit tests for a Java class.

## Context
- Target class: {class_name}
- Existing test file: {test_file_name}
- Current coverage: {current_coverage}%
- Target coverage: {target_coverage}%

## Existing Tests (PRESERVE THESE)
The following tests are PASSING and should be preserved:
{preserved_tests}

## Tests to Fix
The following tests are FAILING and need to be fixed:
{failing_tests_with_errors}

## New Code to Cover
The following code is not covered by existing tests:
{uncovered_code}

## Instructions
1. PRESERVE all passing tests EXACTLY as they are
2. FIX the failing tests to address the errors
3. ADD new tests for uncovered code
4. Output the COMPLETE test class

## Output Format
Output only the complete Java test class code.
```

---

## 7. 风险与缓解

### 7.1 风险
1. **测试合并冲突**: 新测试可能与保留测试有依赖冲突
   - 缓解: 智能合并，检测依赖关系

2. **性能影响**: 增量模式需要先运行现有测试
   - 缓解: 可配置跳过分析，直接基于文件内容

3. **覆盖率计算**: 需要准确计算已有测试的覆盖率
   - 缓解: 利用现有 CoverageAnalyzer

### 7.2 回退策略
- 增量模式失败时，自动回退到完全重新生成
- 提供强制重新生成的选项

---

## 8. 实施顺序

1. **Phase 1** (核心组件) - 1-2天
   - 创建 IncrementalTestManager
   - 扩展配置
   - 扩展提示词构建

2. **Phase 2** (流程集成) - 1-2天
   - 修改 StepExecutor
   - 修改 FeedbackLoopExecutor
   - 修改 ReActAgent

3. **Phase 3** (CLI集成) - 0.5天
   - 单文件命令
   - 批量命令

4. **Phase 4** (GUI集成) - 0.5天
   - 批量对话框
   - 主窗口

5. **Phase 5** (测试) - 1天
   - 单元测试
   - 集成测试

---

## 9. 验收标准

1. CLI 支持 `--incremental` 选项
2. GUI 批量生成支持增量模式选择
3. 增量模式能正确保留通过的测试
4. 增量模式能为新代码生成测试
5. 增量模式能修复失败的测试
6. 增量模式失败时能回退到完全生成
7. 有完整的单元测试覆盖
