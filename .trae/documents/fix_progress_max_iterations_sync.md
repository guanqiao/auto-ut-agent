# GUI Progress 最大迭代次数同步修复计划

## 问题描述

GUI 上 progress 里显示的最大迭代次数不一致：
- 有时显示 10 次
- 有时显示 2 次

需要与配置里的最大迭代次数保持同步。

## 问题根因分析

### 1. 配置默认值不一致

代码库中存在多个 `max_iterations` 的默认值定义，彼此不一致：

| 文件 | 位置 | 默认值 |
|------|------|--------|
| `pyutagent/core/config.py` | `CoverageSettings` 类 | **2** |
| `pyutagent/memory/working_memory.py` | `WorkingMemory` 类 | **3** |
| `pyutagent/agent/test_generator.py` | `generate_tests()` 方法 | **10** |
| `pyutagent/agent/autonomous_loop.py` | `AutonomousLoop` 类 | **10** |
| `pyutagent/agent/base_agent.py` | `load_state()` 方法 | **10** |
| `pyutagent/agent/tool_enabled_agent.py` | 类定义 | **10** |
| `pyutagent/core/state_validator.py` | 状态验证 | **10** |
| `pyutagent/tools/tool_use.py` | 工具使用 | **10** |

### 2. 数据流问题

**当前数据流：**
1. 用户在 `CoverageConfigDialog` 设置最大迭代次数（默认2，范围1-20）
2. 配置保存到 `CoverageSettings`（默认2）
3. `main_window.py` 中 `AgentWorker` 从配置读取并传递给 `WorkingMemory`
4. 但其他组件（如 `AutonomousLoop`、`TestGenerator` 等）使用自己的默认值 **10**

**问题场景：**
- 当某些代码路径没有正确传递 `max_iterations` 参数时，会使用默认值 10
- 而 GUI 显示的是从配置读取的值（2）
- 导致显示不一致

## 修复方案

### 方案：统一使用配置中的默认值

**核心原则：** 所有 `max_iterations` 的默认值都应该引用 `CoverageSettings` 中的定义，而不是硬编码。

### 具体修改步骤

#### 步骤 1: 在 `config.py` 中定义常量

```python
# pyutagent/core/config.py

# 默认最大迭代次数常量
DEFAULT_MAX_ITERATIONS = 2

@dataclass
class CoverageSettings:
    target_coverage: float = 0.8
    min_coverage: float = 0.5
    max_iterations: int = DEFAULT_MAX_ITERATIONS  # 使用常量
    ...
```

#### 步骤 2: 修改 `working_memory.py`

```python
# pyutagent/memory/working_memory.py
from ..core.config import DEFAULT_MAX_ITERATIONS

@dataclass
class WorkingMemory:
    iteration_count: int = 0
    max_iterations: int = DEFAULT_MAX_ITERATIONS  # 从 3 改为使用配置常量
    target_coverage: float = 0.8
```

#### 步骤 3: 修改 `test_generator.py`

```python
# pyutagent/agent/test_generator.py
from ..core.config import DEFAULT_MAX_ITERATIONS

async def generate_tests(
    self,
    target_file: str,
    target_coverage: float = 0.8,
    max_iterations: int = DEFAULT_MAX_ITERATIONS,  # 从 10 改为使用配置常量
) -> Dict[str, Any]:
```

#### 步骤 4: 修改 `autonomous_loop.py`

```python
# pyutagent/agent/autonomous_loop.py
from ..core.config import DEFAULT_MAX_ITERATIONS

class AutonomousLoop:
    def __init__(
        self,
        tool_service: AgentToolService,
        max_iterations: int = DEFAULT_MAX_ITERATIONS,  # 从 10 改为使用配置常量
        ...
    ):
```

#### 步骤 5: 修改 `base_agent.py`

```python
# pyutagent/agent/base_agent.py
from ..core.config import DEFAULT_MAX_ITERATIONS

def load_state(self, path: str) -> bool:
    ...
    self.max_iterations = state_data.get("max_iterations", DEFAULT_MAX_ITERATIONS)  # 从 10 改为使用配置常量
```

#### 步骤 6: 修改其他文件

- `tool_enabled_agent.py`
- `state_validator.py`
- `tools/tool_use.py`

所有硬编码的 `max_iterations = 10` 或 `max_iterations: int = 10` 都改为使用 `DEFAULT_MAX_ITERATIONS`。

### 测试策略

1. **单元测试**：验证每个修改后的组件在创建时使用正确的默认值
2. **集成测试**：验证从 GUI 到 Agent 的完整数据流中 `max_iterations` 保持一致
3. **手动测试**：
   - 打开覆盖率配置对话框，修改最大迭代次数
   - 运行测试生成，观察 progress 显示是否与配置一致
   - 验证不同代码路径（正常生成、恢复状态等）都使用正确的值

## 实施检查清单

- [ ] 在 `config.py` 中添加 `DEFAULT_MAX_ITERATIONS` 常量
- [ ] 修改 `working_memory.py` 使用常量
- [ ] 修改 `test_generator.py` 使用常量
- [ ] 修改 `autonomous_loop.py` 使用常量
- [ ] 修改 `base_agent.py` 使用常量
- [ ] 修改 `tool_enabled_agent.py` 使用常量
- [ ] 修改 `state_validator.py` 使用常量
- [ ] 修改 `tools/tool_use.py` 使用常量
- [ ] 运行单元测试确保通过
- [ ] 手动测试 GUI 显示与配置同步

## 风险与注意事项

1. **向后兼容性**：修改默认值可能影响依赖旧默认值的代码，需要确保所有调用点都正确传递参数
2. **状态恢复**：`load_state()` 方法恢复状态时，如果旧状态没有保存 `max_iterations`，会使用新的默认值
3. **测试覆盖**：需要确保修改后的默认值不会破坏现有测试
