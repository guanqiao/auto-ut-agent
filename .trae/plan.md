# 计划：增强覆盖率评估机制

## 问题分析

### 当前实现的问题

1. **编译失败时无法评估覆盖率**
   - 当编译失败时，`_iteration_compile()` 返回 `False`，整个迭代终止
   - 即使测试代码部分正确，也无法获取覆盖率评估
   - 用户无法了解当前测试代码的质量

2. **JaCoCo 报告不存在时的处理不完善**
   - `StepExecutor.analyze_coverage()` 没有 LLM 回退机制
   - 只有 `CoverageHandler.analyze_coverage()` 有 LLM 回退，但未被主流程使用
   - 当 JaCoCo 未配置或执行失败时，直接返回失败

3. **测试运行失败时的处理**
   - 测试失败时无法获取部分覆盖率数据
   - 无法评估已通过测试的覆盖情况

### 现有资源

- `LLMCoverageEvaluator` - 已实现 LLM 覆盖率评估器
- `CoverageHandler` - 已有完整的 JaCoCo + LLM 回退机制
- `CoverageSource` 枚举 - 区分 JaCoCo 和 LLM 估算来源

## 改进方案

### 方案概述

在以下场景中增加 LLM 覆盖率评估作为回退机制：

1. **编译失败时**：调用 LLM 评估测试代码覆盖率
2. **JaCoCo 报告不存在时**：回退到 LLM 估算
3. **测试运行失败时**：尝试获取部分覆盖率或 LLM 估算

### 详细设计

#### 1. 修改 `StepExecutor.analyze_coverage()`

**文件**: `pyutagent/agent/components/execution_steps.py`

**改动**:
- 当 JaCoCo 报告不存在时，调用 LLM 覆盖率评估
- 添加 `CoverageHandler` 作为组件或直接使用 `LLMCoverageEvaluator`
- 返回覆盖率数据时标记来源（JaCoCo 或 LLM_ESTIMATED）

```python
async def analyze_coverage(self) -> StepResult:
    # ... 现有 JaCoCo 逻辑 ...
    
    if report:
        # JaCoCo 成功
        return StepResult(success=True, data={..., "source": "jacoco"})
    else:
        # 回退到 LLM 估算
        return await self._fallback_to_llm_coverage_estimation()
```

#### 2. 新增 `_fallback_to_llm_coverage_estimation()` 方法

**文件**: `pyutagent/agent/components/execution_steps.py`

**功能**:
- 获取源代码和测试代码
- 调用 `LLMCoverageEvaluator` 评估覆盖率
- 返回带有 `source: "llm_estimated"` 的结果

#### 3. 修改 `_iteration_compile()` 失败处理

**文件**: `pyutagent/agent/components/feedback_loop.py`

**改动**:
- 编译失败时，尝试 LLM 评估覆盖率
- 将评估结果记录到 `working_memory`
- 决定是否继续迭代或终止

```python
async def _iteration_compile(self) -> bool:
    compile_success = await self.step_executor.compile_with_recovery()
    
    if not compile_success:
        # 尝试 LLM 评估覆盖率
        coverage_result = await self._estimate_coverage_on_failure("compilation")
        if coverage_result:
            self.agent_core.working_memory.update_coverage(
                coverage_result.data.get("line_coverage", 0.0)
            )
        return False
    
    return True
```

#### 4. 新增 `_estimate_coverage_on_failure()` 方法

**文件**: `pyutagent/agent/components/feedback_loop.py`

**功能**:
- 在编译或测试失败时调用 LLM 评估覆盖率
- 接受失败原因作为参数（compilation/test_failure）
- 返回覆盖率评估结果

#### 5. 修改 `_iteration_test()` 失败处理

**文件**: `pyutagent/agent/components/feedback_loop.py`

**改动**:
- 测试失败时，尝试获取部分覆盖率或 LLM 评估
- 记录覆盖率数据供后续分析

## 实现步骤

### Step 1: 增强 `StepExecutor.analyze_coverage()`
- 添加 LLM 回退机制
- 确保有必要的上下文（source_code, test_code, class_info）

### Step 2: 添加 `_fallback_to_llm_coverage_estimation()` 方法
- 实现 LLM 覆盖率评估逻辑
- 处理异常情况

### Step 3: 修改 `FeedbackLoopExecutor._iteration_compile()`
- 编译失败时调用 LLM 评估
- 更新覆盖率记录

### Step 4: 添加 `_estimate_coverage_on_failure()` 方法
- 实现失败场景下的覆盖率评估
- 支持不同的失败原因

### Step 5: 修改 `FeedbackLoopExecutor._iteration_test()`
- 测试失败时的覆盖率评估

### Step 6: 更新测试用例
- 添加编译失败场景的测试
- 添加 JaCoCo 不存在场景的测试

## 风险与缓解

### 风险 1: LLM 估算不准确
- **缓解**: 在结果中标记来源为 `llm_estimated`，并添加置信度
- **缓解**: 优先使用 JaCoCo，LLM 仅作为回退

### 风险 2: 增加执行时间
- **缓解**: LLM 评估仅在失败场景触发
- **缓解**: 可以配置是否启用 LLM 回退

### 风险 3: 测试代码语法错误导致 LLM 评估困难
- **缓解**: 在 prompt 中说明代码可能有语法问题
- **缓解**: 使用启发式估算作为最终回退

## 验收标准

1. ✅ 编译失败时能够返回 LLM 估算的覆盖率
2. ✅ JaCoCo 报告不存在时能够回退到 LLM 估算
3. ✅ 覆盖率结果中正确标记来源（jacoco/llm_estimated）
4. ✅ 现有测试不受影响
5. ✅ 新增功能有对应的测试覆盖
