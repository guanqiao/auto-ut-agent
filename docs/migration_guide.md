# PyUT Agent 统一接口迁移指南

本文档指导开发者从旧接口迁移到新的统一接口。

## 概述

PyUT Agent 已完成代码整合和优化，将多个重复功能统一为单一实现。旧接口仍然可用但已标记为废弃，建议尽快迁移。

## 迁移对照表

### 1. 消息总线 (Message Bus)

| 旧接口 | 新接口 | 状态 |
|--------|--------|------|
| `pyutagent.core.event_bus.EventBus` | `pyutagent.core.messaging.UnifiedMessageBus` | 废弃 |
| `pyutagent.core.event_bus.AsyncEventBus` | `pyutagent.core.messaging.UnifiedMessageBus` | 废弃 |
| `pyutagent.core.message_bus.MessageBus` | `pyutagent.core.messaging.UnifiedMessageBus` | 废弃 |
| `pyutagent.agent.multi_agent.message_bus.MessageBus` | `pyutagent.core.messaging.UnifiedMessageBus` | 废弃 |

**迁移示例:**

```python
# 旧代码
from pyutagent.core.event_bus import EventBus
event_bus = EventBus()
event_bus.subscribe(MyEvent, handler)
event_bus.publish(event)

# 新代码
from pyutagent.core.messaging import UnifiedMessageBus, Message, MessageType
bus = UnifiedMessageBus()
await bus.register("my_entity", handler)
message = Message.create(
    sender="sender_id",
    recipient="recipient_id",
    message_type=MessageType.COMPONENT_REQUEST,
    payload={"data": event_data}
)
await bus.send(message)
```

### 2. Agent基类

| 旧接口 | 新接口 | 状态 |
|--------|--------|------|
| `pyutagent.agent.base_agent.BaseAgent` | `pyutagent.agent.unified_agent_base.UnifiedAgentBase` | 废弃 |
| `pyutagent.agent.subagent_base.SubAgent` | `pyutagent.agent.unified_agent_base.UnifiedAgentBase` | 废弃 |

**迁移示例:**

```python
# 旧代码
from pyutagent.agent.base_agent import BaseAgent
class MyAgent(BaseAgent):
    async def generate_tests(self, target_file: str) -> AgentResult:
        # ...
        pass

# 新代码
from pyutagent.agent.unified_agent_base import (
    UnifiedAgentBase, AgentConfig, AgentCapability, AgentResult
)

class MyAgent(UnifiedAgentBase):
    def __init__(self):
        config = AgentConfig(
            name="MyAgent",
            agent_type="test_generator",
            capabilities=[AgentCapability.TEST_GENERATION]
        )
        super().__init__(config)
    
    async def execute(self, task: Any) -> AgentResult:
        # 统一使用 execute 方法
        # ...
        return AgentResult(success=True, output=result)
```

### 3. 自主循环

| 旧接口 | 新接口 | 状态 |
|--------|--------|------|
| `pyutagent.agent.autonomous_loop.AutonomousLoop` | `pyutagent.agent.unified_autonomous_loop.UnifiedAutonomousLoop` | 废弃 |
| `pyutagent.agent.enhanced_autonomous_loop.EnhancedAutonomousLoop` | `pyutagent.agent.unified_autonomous_loop.UnifiedAutonomousLoop` | 废弃 |
| `pyutagent.agent.llm_driven_autonomous_loop.LLMDrivenAutonomousLoop` | `pyutagent.agent.unified_autonomous_loop.UnifiedAutonomousLoop` | 废弃 |
| `pyutagent.agent.delegating_autonomous_loop.DelegatingAutonomousLoop` | `pyutagent.agent.unified_autonomous_loop.UnifiedAutonomousLoop` | 废弃 |

**迁移示例:**

```python
# 旧代码
from pyutagent.agent.autonomous_loop import AutonomousLoop
loop = AutonomousLoop(tool_service, llm_client)
result = await loop.run("Generate tests for UserService.java")

# 新代码
from pyutagent.agent.unified_autonomous_loop import (
    UnifiedAutonomousLoop, LoopConfig, LoopFeature
)

config = LoopConfig(
    max_iterations=10,
    features={
        LoopFeature.LLM_REASONING,
        LoopFeature.SELF_CORRECTION,
        LoopFeature.LEARNING,
    }
)

class MyLoop(UnifiedAutonomousLoop):
    async def _observe(self, task: str, context: Dict[str, Any]) -> Observation:
        # 实现观察逻辑
        pass
    
    async def _think(self, task: str, observation: Observation, context: Dict[str, Any]) -> Thought:
        # 实现思考逻辑
        pass
    
    async def _act(self, action: Action, context: Dict[str, Any]) -> Any:
        # 实现执行逻辑
        pass
    
    async def _verify(self, result: Any, expected: str, context: Dict[str, Any]) -> Verification:
        # 实现验证逻辑
        pass

loop = MyLoop(config)
result = await loop.run("Generate tests for UserService.java")
```

### 4. 配置管理

| 旧接口 | 新接口 | 状态 |
|--------|--------|------|
| `pyutagent.llm.config.LLMConfig` | `pyutagent.core.config.LLMConfig` | 已重定向 |
| `pyutagent.config.project_config.ProjectConfigManager` | `pyutagent.core.project_config.ProjectConfigManager` | 废弃 |

**迁移示例:**

```python
# 旧代码
from pyutagent.config.project_config import ProjectConfigManager
manager = ProjectConfigManager()

# 新代码
from pyutagent.core.project_config import ProjectConfigManager
manager = ProjectConfigManager()
```

### 5. 执行器

| 旧接口 | 新接口 | 状态 |
|--------|--------|------|
| `pyutagent.agent.batch_executor.BatchExecutor` | `pyutagent.agent.execution.executor.StepExecutor` | 废弃 |
| `pyutagent.agent.parallel_executor.ParallelExecutor` | `pyutagent.agent.execution.executor.StepExecutor` | 废弃 |

**迁移示例:**

```python
# 旧代码
from pyutagent.agent.batch_executor import BatchExecutor
executor = BatchExecutor()

# 新代码
from pyutagent.agent.execution.executor import StepExecutor
from pyutagent.core.agent_state import StateManager
from pyutagent.core.agent_context import AgentContext

state_manager = StateManager()
context = AgentContext()
executor = StepExecutor(state_manager, context)

# 注册步骤处理器
executor.register_handler("my_step_type", handler_func)

# 执行计划
results = await executor.execute_plan(plan, parallel=True)
```

### 6. Manager类

| 旧接口 | 新接口 | 状态 |
|--------|--------|------|
| `pyutagent.agent.utils.state_manager.StateManager` | `pyutagent.agent.core.agent_state.StateManager` | 废弃 |
| `pyutagent.agent.context_manager.ContextManager` | `pyutagent.agent.unified_context_manager.UnifiedContextManager` | 废弃 |

**迁移示例:**

```python
# 旧代码
from pyutagent.agent.utils.state_manager import StateManager
manager = StateManager()

# 新代码
from pyutagent.agent.core.agent_state import StateManager
manager = StateManager()
```

## 废弃警告处理

在迁移期间，你会看到类似以下的警告：

```
DeprecationWarning: pyutagent.core.event_bus is deprecated. 
Use pyutagent.core.messaging.UnifiedMessageBus instead.
```

这些警告是正常的，用于提醒你迁移到新的接口。

### 临时禁用警告（不推荐长期使用）

```python
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
```

## 向后兼容适配器

对于需要暂时保持旧接口的情况，可以使用适配器：

```python
from pyutagent.core.messaging import EventBusAdapter

# 创建适配器（会显示废弃警告）
adapter = EventBusAdapter(unified_bus)

# 使用旧接口
adapter.subscribe(MyEvent, handler)
adapter.publish(event)
```

## 完整测试

迁移后，请运行测试确保功能正常：

```bash
# 运行单元测试
python -m pytest tests/unit/ -v --tb=short

# 运行集成测试
python -m pytest tests/integration/ -v --tb=short

# 检查覆盖率
python -m pytest tests/ --cov=pyutagent --cov-report=term-missing
```

## 常见问题

### Q: 为什么需要迁移？
A: 统一接口减少了代码重复，提高了可维护性，并提供了更一致的功能集。

### Q: 旧接口什么时候会被移除？
A: 目前没有时间表，但建议尽快迁移。移除前会提前通知。

### Q: 迁移过程中遇到问题怎么办？
A: 
1. 检查本迁移指南的对照表
2. 查看新接口的文档字符串
3. 参考 `tests/unit/` 中的测试用例
4. 在项目中提交 Issue

## 相关文档

- [统一架构设计](architecture.md)
- [API参考](api_reference.md)
- [开发指南](development_guide.md)
