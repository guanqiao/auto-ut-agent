# 无限循环风险分析与修复计划

## 一、发现的问题

### 🔴 问题1: 递归调用时 attempt 计数器重置

**位置**: `execution_steps.py:127-130`

```python
elif action == "reset":
    logger.info("[StepExecutor] Resetting and regenerating")
    return await self.execute_with_recovery(
        self.generate_initial_tests,
        step_name="regenerating tests"  # ❌ attempt 会重置为 0
    )
```

**风险**: 每次递归调用 `execute_with_recovery` 都会重置 `attempt = 0`，如果恢复策略一直是 "reset"，会导致无限递归。

### 🔴 问题2: compile_with_recovery 中的递归调用

**位置**: `execution_steps.py:917-922`

```python
elif action == "reset":
    reset_result = await self.execute_with_recovery(
        self.generate_initial_tests,
        step_name="regenerating after compilation failure"  # ❌ attempt 会重置
    )
```

### 🔴 问题3: run_tests_with_recovery 中的递归调用

**位置**: `execution_steps.py:1083-1088`

```python
elif action == "reset":
    reset_result = await self.execute_with_recovery(
        self.generate_initial_tests,
        step_name="regenerating after test failure"  # ❌ attempt 会重置
    )
```

### 🟡 问题4: 全局重置计数器缺失

当前没有跟踪全局重置次数的机制，无法限制总的重置次数。

---

## 二、修复方案

### 方案1: 添加全局重置计数器

在 `AgentCore` 或 `RetryConfig` 中添加 `reset_count` 和 `max_reset_count`：

```python
# RetryConfig 添加:
max_reset_count: int = 2  # 最大重置次数

# AgentCore 添加:
_reset_count: int = 0
```

### 方案2: 在 execute_with_recovery 中传递和检查重置计数

```python
async def execute_with_recovery(
    self,
    operation,
    *args,
    step_name: str = "operation",
    reset_count: int = 0,  # 新增参数
    **kwargs
) -> StepResult:
    attempt = 0
    max_attempts = self.retry_config.get_max_attempts(step_name)
    
    # 检查重置次数
    if reset_count > self.retry_config.max_reset_count:
        logger.error(f"[StepExecutor] Exceeded max reset count ({self.retry_config.max_reset_count})")
        return StepResult(
            success=False,
            state=AgentState.FAILED,
            message=f"Exceeded maximum reset count ({self.retry_config.max_reset_count})"
        )
    
    # ... 在递归调用时传递 reset_count + 1
    elif action == "reset":
        return await self.execute_with_recovery(
            self.generate_initial_tests,
            step_name="regenerating tests",
            reset_count=reset_count + 1  # 递增重置计数
        )
```

### 方案3: 使用 TerminationChecker 跟踪重置次数

在 `TerminationChecker` 中添加重置计数：

```python
class TerminationChecker:
    def __init__(self, ..., max_reset_count: int = 2):
        self.max_reset_count = max_reset_count
        self._reset_count = 0
    
    def record_reset(self):
        self._reset_count += 1
    
    def can_reset(self) -> bool:
        return self._reset_count < self.max_reset_count
```

---

## 三、修复步骤

### Step 1: 修改 RetryConfig

添加 `max_reset_count` 配置：

```python
# pyutagent/core/retry_config.py
@dataclass
class RetryConfig:
    max_total_attempts: int = 50
    max_step_attempts: int = 2
    max_compilation_attempts: int = 2
    max_test_attempts: int = 2
    max_reset_count: int = 2  # 新增
```

### Step 2: 修改 StepExecutor

添加 `reset_count` 参数和检查：

```python
# pyutagent/agent/components/execution_steps.py

async def execute_with_recovery(
    self,
    operation,
    *args,
    step_name: str = "operation",
    reset_count: int = 0,
    **kwargs
) -> StepResult:
    attempt = 0
    max_attempts = self.retry_config.get_max_attempts(step_name)
    
    # 检查重置次数限制
    if reset_count >= self.retry_config.max_reset_count:
        logger.error(f"[StepExecutor] Exceeded max reset count - ResetCount: {reset_count}/{self.retry_config.max_reset_count}")
        return StepResult(
            success=False,
            state=AgentState.FAILED,
            message=f"Exceeded maximum reset count ({self.retry_config.max_reset_count})"
        )
    
    # ... 其他代码 ...
    
    # 在递归调用时传递 reset_count + 1
    elif action == "reset":
        logger.info(f"[StepExecutor] Resetting and regenerating - ResetCount: {reset_count + 1}/{self.retry_config.max_reset_count}")
        return await self.execute_with_recovery(
            self.generate_initial_tests,
            step_name="regenerating tests",
            reset_count=reset_count + 1
        )
```

### Step 3: 修改 compile_with_recovery 和 run_tests_with_recovery

同样添加重置计数检查。

### Step 4: 添加单元测试

测试重置计数限制是否生效。

---

## 四、预期效果

| 场景 | 修复前 | 修复后 |
|------|--------|--------|
| 恢复策略一直是 "reset" | 无限递归 | 最多重置 2 次后失败 |
| 编译失败后重置 | 可能无限循环 | 最多重置 2 次 |
| 测试失败后重置 | 可能无限循环 | 最多重置 2 次 |

---

## 五、文件修改清单

| 文件 | 修改内容 |
|------|---------|
| `pyutagent/core/retry_config.py` | 添加 `max_reset_count` |
| `pyutagent/agent/components/execution_steps.py` | 添加 `reset_count` 参数和检查 |
| `tests/unit/core/test_retry_config.py` | 添加重置计数测试 |

---

*分析时间: 2026-03-04*
