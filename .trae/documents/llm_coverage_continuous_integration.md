# 计划：LLM 覆盖率评估持续集成

## 当前状态

### 已完成集成的位置
| 文件 | 方法 | 状态 |
|------|------|------|
| `coverage_analysis_service.py` | `analyze_coverage()`, `_fallback_to_llm_estimation()` | ✅ |
| `coverage_handler.py` | `analyze_coverage()`, `_fallback_to_llm_estimation()` | ✅ |
| `incremental_manager.py` | `_analyze_coverage()`, `_fallback_to_llm_coverage()` | ✅ |
| `execution_steps.py` | `analyze_coverage()`, `_fallback_to_llm_coverage_estimation()` | ✅ |

### 待集成的位置
| 文件 | 优先级 | 改动复杂度 |
|------|--------|-----------|
| `test_generator.py` | 高 | 中等 |
| `actions.py` | 高 | 中等 |
| `maven_tools.py` | 中 | 低 |
| `batch_generator.py` | 中 | 低 |
| `generate_all.py` | 低 | 低 |

---

## 实现步骤

### Step 1: 增强 TestGeneratorAgent

**文件**: `pyutagent/agent/test_generator.py`

**改动位置**:
- 第 217 行: `generate_tests()` 方法中的覆盖率解析
- 第 263 行: 迭代优化循环中的覆盖率解析
- 第 488 行: `generate_tests_with_aider()` 方法
- 第 546 行: Aider 迭代优化

**改动内容**:
1. 添加 `llm_client` 参数到 `__init__`
2. 添加 `_analyze_coverage_with_fallback()` 方法
3. 替换所有 `self.coverage_analyzer.parse_report()` 调用

### Step 2: 增强 AnalyzeCoverageAction

**文件**: `pyutagent/agent/actions.py`

**改动位置**: 第 379-433 行 `AnalyzeCoverageAction` 类

**改动内容**:
1. 添加 `llm_client` 参数
2. 添加 `set_context()` 方法
3. 添加 `_fallback_to_llm()` 方法
4. 修改 `execute()` 方法支持 LLM fallback

### Step 3: 增强 CoverageAnalyzer (可选)

**文件**: `pyutagent/tools/maven_tools.py`

**改动位置**: 第 622-837 行 `CoverageAnalyzer` 类

**改动内容**:
1. 添加 `llm_client` 参数
2. 添加 `set_estimation_context()` 方法
3. 添加 `parse_report_with_fallback()` 方法

### Step 4: 增强 BatchGenerator 数据结构

**文件**: `pyutagent/services/batch_generator.py`

**改动位置**: 第 633-636 行 `FileResult` 类

**改动内容**:
1. 添加 `coverage_source` 字段
2. 添加 `coverage_confidence` 字段
3. 更新结果提取逻辑

### Step 5: 增强 CLI 显示

**文件**: `pyutagent/cli/commands/generate_all.py`

**改动位置**: 第 172-189 行结果显示

**改动内容**:
1. 区分 JaCoCo 和 LLM 估算的覆盖率显示
2. 添加覆盖率来源统计

---

## 详细设计

### 1. TestGeneratorAgent 改动

```python
def __init__(
    self,
    project_path: str,
    llm_config: LLMConfig,
    conversation: ConversationManager,
    working_memory: WorkingMemory,
    llm_client: Optional[Any] = None,  # 新增
):
    ...
    self._llm_client_for_coverage = llm_client

async def _analyze_coverage_with_fallback(
    self, 
    source_code: str, 
    test_code: str, 
    class_info: Dict
) -> Tuple[Optional[CoverageReport], str, float]:
    """分析覆盖率，支持 LLM fallback
    
    Returns:
        Tuple[report, source, confidence]
    """
    try:
        report = self.coverage_analyzer.parse_report()
        if report:
            return report, "jacoco", 1.0
    except Exception as e:
        logger.warning(f"JaCoCo 解析失败: {e}")
    
    if self._llm_client_for_coverage and source_code and test_code:
        from .llm_coverage_evaluator import LLMCoverageEvaluator
        evaluator = LLMCoverageEvaluator(self._llm_client_for_coverage)
        llm_report = evaluator.quick_estimate(source_code, test_code, class_info)
        return llm_report, "llm_estimated", llm_report.confidence
    
    return None, "unknown", 0.0
```

### 2. AnalyzeCoverageAction 改动

```python
class AnalyzeCoverageAction(Action):
    def __init__(self, coverage_analyzer, llm_client=None):
        super().__init__(
            name="analyze_coverage",
            action_type=ActionType.ANALYZE_COVERAGE,
            description="Analyze test coverage with LLM fallback"
        )
        self.coverage_analyzer = coverage_analyzer
        self.llm_client = llm_client
        self._source_code = None
        self._test_code = None
        self._class_info = None
    
    def set_context(self, source_code: str, test_code: str, class_info: Dict):
        self._source_code = source_code
        self._test_code = test_code
        self._class_info = class_info
    
    async def execute(self, **kwargs) -> ActionResult:
        report = self.coverage_analyzer.parse_report()
        if report:
            return ActionResult(
                success=True,
                data={"line_coverage": report.line_coverage, "source": "jacoco"}
            )
        return await self._fallback_to_llm()
```

### 3. FileResult 扩展

```python
@dataclass
class FileResult:
    file_path: str
    success: bool
    coverage: float = 0.0
    coverage_source: str = "jacoco"  # 新增
    coverage_confidence: float = 1.0  # 新增
    iterations: int = 0
    # ...
```

---

## 验收标准

1. ✅ TestGeneratorAgent 支持 LLM 覆盖率评估
2. ✅ AnalyzeCoverageAction 支持 LLM fallback
3. ✅ BatchGenerator 正确记录覆盖率来源
4. ✅ CLI 正确显示覆盖率来源
5. ✅ 现有测试不受影响
