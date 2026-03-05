# 计划：LLM 覆盖率评估持续集成到主流程

## 当前状态分析

### 已完成的集成
1. **StepExecutor.analyze_coverage()** - 已有 LLM fallback 机制
2. **FeedbackLoopExecutor** - 编译/测试失败时调用 LLM 估算
3. **WorkingMemory** - 记录覆盖率历史

### 需要改进的集成点

| 集成点 | 文件 | 当前状态 | 改进需求 |
|--------|------|----------|----------|
| CoverageAnalysisService | `services/coverage_analysis_service.py` | 仅支持 JaCoCo | 添加 LLM fallback |
| IncrementalManager | `incremental_manager.py` | 仅使用 JaCoCo | 添加 LLM fallback |
| AgentResult | `core/protocols.py` | 无来源字段 | 添加 coverage_source, coverage_confidence |
| UI 进度更新 | `ui/main_window.py` | 未区分来源 | 显示覆盖率来源和置信度 |

---

## 实现步骤

### Step 1: 增强 CoverageAnalysisService

**文件**: `pyutagent/agent/services/coverage_analysis_service.py`

**改动内容**:
1. 添加 `llm_client`, `source_code`, `test_code`, `class_info` 参数
2. 修改 `analyze_coverage()` 方法，添加 LLM fallback
3. 返回数据中添加 `source` 和 `confidence` 字段

**代码改动**:
```python
def __init__(
    self,
    project_path: str,
    maven_runner: Optional[MavenRunner] = None,
    coverage_analyzer: Optional[CoverageAnalyzer] = None,
    progress_callback: Optional[Callable[[AgentState, str], None]] = None,
    llm_client: Optional[Any] = None,  # 新增
    source_code: Optional[str] = None,  # 新增
    test_code: Optional[str] = None,    # 新增
    class_info: Optional[Dict[str, Any]] = None  # 新增
):
    ...
    self.llm_client = llm_client
    self.source_code = source_code
    self.test_code = test_code
    self.class_info = class_info

async def analyze_coverage(self) -> StepResult:
    # 尝试 JaCoCo
    report = self._coverage_analyzer.parse_report()
    
    if report:
        return StepResult(
            success=True,
            data={
                "line_coverage": report.line_coverage,
                "source": "jacoco",
                "confidence": 1.0
            }
        )
    
    # Fallback to LLM
    return await self._fallback_to_llm_estimation()
```

---

### Step 2: 增强 IncrementalManager._analyze_coverage()

**文件**: `pyutagent/agent/incremental_manager.py`

**改动内容**:
1. 添加 `llm_client` 参数到 `__init__`
2. 修改 `_analyze_coverage()` 方法，添加 LLM fallback
3. 返回数据中添加 `source` 和 `confidence` 字段

**代码改动**:
```python
def _analyze_coverage(self) -> Optional[Dict[str, Any]]:
    # 尝试 JaCoCo
    report = self.coverage_analyzer.parse_report()
    
    if report:
        return {
            "line_coverage": report.line_coverage,
            "source": "jacoco",
            "confidence": 1.0,
            ...
        }
    
    # Fallback to LLM estimation
    if self.llm_client and self.source_code and self.test_code:
        from ..llm_coverage_evaluator import LLMCoverageEvaluator
        evaluator = LLMCoverageEvaluator(self.llm_client)
        llm_report = evaluator.quick_estimate(
            self.source_code, self.test_code, self.class_info
        )
        return {
            "line_coverage": llm_report.line_coverage,
            "source": "llm_estimated",
            "confidence": llm_report.confidence,
            ...
        }
    
    return None
```

---

### Step 3: 扩展 AgentResult 数据结构

**文件**: `pyutagent/core/protocols.py`

**改动内容**:
添加 `coverage_source` 和 `coverage_confidence` 字段

**代码改动**:
```python
@dataclass
class AgentResult(Generic[T]):
    """Result from agent execution."""
    success: bool
    message: str = ""
    state: AgentState = AgentState.IDLE
    data: Optional[T] = None
    test_file: Optional[str] = None
    coverage: float = 0.0
    coverage_source: str = "jacoco"        # 新增
    coverage_confidence: float = 1.0       # 新增
    iterations: int = 0
    errors: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
```

---

### Step 4: 更新 CoreAgent 结果创建方法

**文件**: `pyutagent/agent/components/core_agent.py`

**改动内容**:
修改 `_create_success_result()` 和 `_create_final_result()` 方法，传递覆盖率来源信息

**代码改动**:
```python
def _create_success_result(self, coverage: float, source: str = "jacoco", confidence: float = 1.0) -> AgentResult:
    return AgentResult(
        success=True,
        message=f"Successfully generated tests with {coverage:.1%} coverage",
        test_file=self.current_test_file,
        coverage=coverage,
        coverage_source=source,
        coverage_confidence=confidence,
        iterations=self.current_iteration
    )
```

---

### Step 5: 增强 UI 进度显示

**文件**: `pyutagent/ui/main_window.py`

**改动内容**:
1. 修改 `update_coverage()` 方法，接受来源和置信度参数
2. 显示格式区分 JaCoCo 和 LLM 估算

**代码改动**:
```python
def update_coverage(self, coverage: float, target: float, source: str = "jacoco", confidence: float = 1.0):
    """Update coverage display with source information."""
    if source == "llm_estimated":
        text = f"Coverage: {coverage:.1%} (LLM估算, 置信度: {confidence:.0%}) / Target: {target:.1%}"
    else:
        text = f"Coverage: {coverage:.1%} / Target: {target:.1%}"
    
    self.coverage_label.setText(text)
    # ... 颜色逻辑保持不变
```

---

### Step 6: 更新 FeedbackLoopExecutor 传递来源信息

**文件**: `pyutagent/agent/components/feedback_loop.py`

**改动内容**:
在 `_create_final_result()` 中传递覆盖率来源信息

---

## 验收标准

1. ✅ CoverageAnalysisService 支持 LLM fallback
2. ✅ IncrementalManager 支持 LLM 覆盖率评估
3. ✅ AgentResult 包含 coverage_source 和 coverage_confidence 字段
4. ✅ UI 正确显示覆盖率来源和置信度
5. ✅ 现有测试不受影响
6. ✅ 新增功能有对应的测试覆盖

---

## 风险与缓解

### 风险 1: LLM 估算不准确影响用户判断
- **缓解**: UI 明确显示来源和置信度，用户可自行判断

### 风险 2: 增量模式下的性能影响
- **缓解**: 仅在 JaCoCo 不可用时才调用 LLM

### 风险 3: 向后兼容性
- **缓解**: AgentResult 新字段有默认值，不影响现有代码
