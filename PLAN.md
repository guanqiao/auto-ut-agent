# 测试生成智能重试机制改进计划

## 背景

当前测试生成过程中的重试机制存在以下问题：
1. **简单重试过多**：很多时候只是简单地重试，没有充分利用 LLM 分析能力
2. **编译/测试失败处理不够智能**：失败信息没有完整传递给 LLM 进行分析
3. **LLM 反馈没有被充分利用**：LLM 的分析结果没有转化为具体的工具调用

## 目标

增强 LLM 分析编译和测试失败的能力，让 Agent 根据 LLM 的反馈调用合适的工具执行修复方案。

## 改进方案

### 1. 增强错误上下文收集

**文件**: `pyutagent/core/error_context.py` (新建)

**功能**:
- 收集完整的编译错误输出（`mvn clean compile` 的完整结果）
- 收集完整的测试失败输出（`mvn clean test` 的完整结果）
- 收集相关的源代码和测试代码上下文
- 结构化错误信息，便于 LLM 分析

**关键类**:
```python
@dataclass
class CompilationErrorContext:
    """编译错误上下文"""
    compiler_output: str           # 完整编译输出
    error_lines: List[str]         # 错误行列表
    missing_imports: List[str]     # 缺失的导入
    missing_dependencies: List[str] # 缺失的依赖
    syntax_errors: List[str]       # 语法错误
    type_errors: List[str]         # 类型错误
    source_file: str               # 源文件路径
    test_file: str                 # 测试文件路径
    source_code: str               # 源代码
    test_code: str                 # 测试代码

@dataclass
class TestFailureContext:
    """测试失败上下文"""
    test_output: str               # 完整测试输出
    failed_tests: List[Dict]       # 失败的测试列表
    passed_tests: List[str]        # 通过的测试列表
    failure_reasons: List[str]     # 失败原因
    stack_traces: List[str]        # 堆栈跟踪
    test_file: str                 # 测试文件路径
    test_code: str                 # 测试代码
    source_code: str               # 源代码
```

### 2. 增强 LLM 分析 Prompt

**文件**: `pyutagent/agent/prompts/prompts.py`

**新增方法**:

```python
def build_compilation_analysis_prompt(
    self,
    error_context: CompilationErrorContext,
    attempt_history: List[Dict[str, Any]]
) -> str:
    """构建编译错误分析 Prompt，包含完整的编译输出和上下文"""

def build_test_failure_analysis_prompt(
    self,
    error_context: TestFailureContext,
    attempt_history: List[Dict[str, Any]]
) -> str:
    """构建测试失败分析 Prompt，包含完整的测试输出和上下文"""

def build_action_plan_prompt(
    self,
    analysis_result: Dict[str, Any],
    available_tools: List[str]
) -> str:
    """构建行动方案 Prompt，让 LLM 给出具体的工具调用建议"""
```

**Prompt 结构**:
- 完整的错误输出（不截断关键信息）
- 相关的源代码和测试代码
- 历史尝试记录
- 可用的工具列表
- 期望的输出格式（包含具体行动方案）

### 3. 增强错误恢复管理器

**文件**: `pyutagent/core/error_recovery.py`

**改进点**:

1. **增强 `recover` 方法**:
   - 收集完整的错误上下文
   - 调用 LLM 进行深度分析
   - 解析 LLM 返回的行动方案
   - 返回具体的工具调用建议

2. **新增方法**:

```python
async def analyze_with_full_context(
    self,
    error: Exception,
    context: Dict[str, Any]
) -> Dict[str, Any]:
    """使用完整上下文进行 LLM 分析"""

def parse_action_plan(self, llm_response: str) -> List[Dict[str, Any]]:
    """解析 LLM 返回的行动方案，转换为工具调用列表"""
```

### 4. 增强步骤执行器

**文件**: `pyutagent/agent/components/execution_steps.py`

**改进点**:

1. **增强 `compile_with_recovery` 方法**:
   - 收集完整的编译输出
   - 调用增强的错误恢复流程
   - 根据 LLM 建议执行具体操作

2. **增强 `run_tests_with_recovery` 方法**:
   - 收集完整的测试输出
   - 调用增强的错误恢复流程
   - 根据 LLM 建议执行具体操作

3. **新增方法**:

```python
async def _collect_compilation_context(self, errors: List[str]) -> CompilationErrorContext:
    """收集编译错误上下文"""

async def _collect_test_failure_context(self, failures: List[Dict]) -> TestFailureContext:
    """收集测试失败上下文"""

async def _execute_llm_action_plan(self, action_plan: List[Dict[str, Any]]) -> bool:
    """执行 LLM 给出的行动方案"""
```

### 5. 新增工具调用执行器

**文件**: `pyutagent/agent/tools/action_executor.py` (新建)

**功能**:
- 根据 LLM 的建议执行具体的工具调用
- 支持的工具:
  - `fix_imports`: 修复导入问题
  - `add_dependency`: 添加依赖
  - `fix_syntax`: 修复语法错误
  - `fix_type_error`: 修复类型错误
  - `regenerate_test`: 重新生成测试
  - `fix_test_logic`: 修复测试逻辑
  - `add_missing_mock`: 添加缺失的 Mock
  - `fix_assertion`: 修复断言
  - `skip_test`: 跳过失败的测试

**关键类**:
```python
class ActionExecutor:
    """执行 LLM 建议的工具调用"""

    async def execute_action(self, action: Dict[str, Any]) -> ActionResult:
        """执行单个行动"""

    async def execute_action_plan(self, actions: List[Dict[str, Any]]) -> List[ActionResult]:
        """执行行动方案列表"""

    def get_available_actions(self) -> List[str]:
        """获取可用的行动类型"""
```

### 6. 改进重试配置

**文件**: `pyutagent/core/retry_config.py`

**改进点**:
- 增加智能重试相关配置
- 区分网络错误的简单重试和代码错误的智能重试

```python
@dataclass
class RetryConfig:
    # ... 现有配置 ...

    # 智能重试配置
    enable_smart_retry: bool = True       # 启用智能重试
    simple_retry_for_network: bool = True # 网络错误使用简单重试
    llm_analysis_for_code: bool = True    # 代码错误使用 LLM 分析
    max_llm_analysis_attempts: int = 3    # 最大 LLM 分析尝试次数
```

### 7. 改进失败模式追踪器

**文件**: `pyutagent/core/failure_pattern_tracker.py`

**改进点**:
- 增加对 LLM 分析结果的追踪
- 记录 LLM 建议的成功率
- 根据历史 LLM 建议调整策略

## 实施步骤

### Phase 1: 基础设施 (1-2 天)
1. 创建 `error_context.py`，定义错误上下文数据结构
2. 创建 `action_executor.py`，实现工具调用执行器

### Phase 2: LLM 分析增强 (2-3 天)
1. 增强 `prompts.py`，添加新的 Prompt 构建方法
2. 增强 `error_recovery.py`，实现深度 LLM 分析
3. 实现行动方案解析逻辑

### Phase 3: 执行流程改进 (2-3 天)
1. 增强 `execution_steps.py`，集成新的错误恢复流程
2. 实现 LLM 行动方案执行
3. 改进重试配置

### Phase 4: 测试和优化 (1-2 天)
1. 编写单元测试
2. 集成测试
3. 性能优化

## 预期效果

1. **减少无效重试**：通过 LLM 分析，避免重复相同的错误
2. **提高修复成功率**：LLM 能够理解错误根因并给出针对性建议
3. **更好的可观测性**：记录 LLM 分析过程和建议，便于调试
4. **更智能的决策**：Agent 能够根据 LLM 反馈选择最佳行动

## 风险和缓解措施

1. **LLM 调用成本增加**
   - 缓解：只在必要时调用 LLM（非网络错误）
   - 缓解：使用缓存避免重复分析相同错误

2. **LLM 分析可能不准确**
   - 缓解：保留现有的重试限制
   - 缓解：记录 LLM 建议的成功率，低成功率时回退到简单重试

3. **响应时间增加**
   - 缓解：异步执行 LLM 分析
   - 缓解：设置合理的超时时间

## 验收标准

1. 编译失败时，能够将完整的编译输出传递给 LLM 分析
2. 测试失败时，能够将完整的测试输出传递给 LLM 分析
3. LLM 能够返回具体的修复建议和工具调用方案
4. Agent 能够根据 LLM 建议执行相应的工具调用
5. 网络错误仍然使用简单重试机制
6. 整体重试次数减少，成功率提高
