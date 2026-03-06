# PyUT Agent 统一接口推荐指南

本文档介绍 PyUT Agent 的统一接口，推荐在新开发中使用这些接口。

## 概述

为了提高代码可维护性和减少重复，PyUT Agent 已将多个重复功能整合为统一接口。本文档介绍推荐的统一接口及其使用方法。

## 推荐的统一接口

### 1. 消息总线 - UnifiedMessageBus

**推荐导入:**
```python
from pyutagent.core.messaging import UnifiedMessageBus, Message, MessageType, MessagePriority
```

**特性:**
- 点对点消息传递
- 广播消息
- 请求-响应模式
- 优先级支持
- 消息历史
- 统计信息

**基本使用:**
```python
# 创建消息总线
bus = UnifiedMessageBus(max_queue_size=1000)

# 注册实体
await bus.register("my_agent")

# 发送消息
message = Message.create(
    sender="my_agent",
    recipient="other_agent",
    message_type=MessageType.AGENT_TASK,
    payload={"task": "generate_tests"}
)
await bus.send(message)

# 接收消息
received = await bus.receive("my_agent", timeout=30.0)

# 请求-响应模式
response = await bus.request(
    sender="my_agent",
    recipient="other_agent",
    payload={"query": "get_status"},
    timeout=30.0
)
```

### 2. Agent基类 - UnifiedAgentBase

**推荐导入:**
```python
from pyutagent.agent.unified_agent_base import (
    UnifiedAgentBase,
    AgentConfig,
    AgentCapability,
    AgentResult,
    AgentState
)
```

**特性:**
- 统一状态管理
- 生命周期控制（启动、停止、暂停、恢复）
- 进度报告
- 错误处理
- 配置管理
- 能力系统

**基本使用:**
```python
class MyAgent(UnifiedAgentBase):
    def __init__(self):
        config = AgentConfig(
            name="TestGenerator",
            agent_type="test_generator",
            description="Generates unit tests for Java classes",
            capabilities=[
                AgentCapability.TEST_GENERATION,
                AgentCapability.CODE_ANALYSIS,
            ],
            max_iterations=10,
            timeout=300,
            max_retries=3
        )
        super().__init__(config)
    
    async def execute(self, task: Any) -> AgentResult:
        """执行任务"""
        try:
            # 更新进度
            self._update_progress(0.1, "analyzing", "Analyzing target class...")
            
            # 检查是否应该继续
            if not self._should_continue():
                return AgentResult(
                    success=False,
                    error="Execution stopped",
                    state=AgentState.STOPPED
                )
            
            # 检查暂停
            await self._check_pause()
            
            # 执行任务逻辑
            result = await self._generate_tests(task)
            
            return AgentResult(
                success=True,
                output=result,
                iterations=self._current_iteration,
                state=AgentState.COMPLETED
            )
        except Exception as e:
            return AgentResult(
                success=False,
                error=str(e),
                state=AgentState.FAILED
            )

# 使用Agent
agent = MyAgent()
result = await agent.run({"target_file": "UserService.java"})
```

### 3. 自主循环 - UnifiedAutonomousLoop

**推荐导入:**
```python
from pyutagent.agent.unified_autonomous_loop import (
    UnifiedAutonomousLoop,
    LoopConfig,
    LoopFeature,
    LoopState,
    Observation,
    Thought,
    Action,
    Verification,
    LoopResult
)
```

**特性:**
- O-T-A-V-L 循环（观察-思考-行动-验证-学习）
- 特性开关机制
- 自我纠正
- 学习机制
- 进度报告

**基本使用:**
```python
class MyAutonomousLoop(UnifiedAutonomousLoop):
    def __init__(self):
        config = LoopConfig(
            max_iterations=10,
            confidence_threshold=0.8,
            features={
                LoopFeature.LLM_REASONING,
                LoopFeature.SELF_CORRECTION,
                LoopFeature.LEARNING,
                LoopFeature.PROGRESS_REPORTING,
            }
        )
        super().__init__(config)
    
    async def _observe(self, task: str, context: Dict[str, Any]) -> Observation:
        """观察当前状态"""
        return Observation(
            timestamp=datetime.now(),
            state_summary="Current state",
            relevant_data={"task": task},
            tool_results=[]
        )
    
    async def _think(self, task: str, observation: Observation, context: Dict[str, Any]) -> Thought:
        """思考下一步行动"""
        return Thought(
            timestamp=datetime.now(),
            reasoning="Based on observation...",
            decision="Generate tests",
            confidence=0.9,
            plan=[{
                "tool_name": "generate_test",
                "parameters": {"target": task},
                "expected_outcome": "Test file generated"
            }]
        )
    
    async def _act(self, action: Action, context: Dict[str, Any]) -> Any:
        """执行动作"""
        # 执行工具调用
        result = await execute_tool(action.tool_name, action.parameters)
        return result
    
    async def _verify(self, result: Any, expected: str, context: Dict[str, Any]) -> Verification:
        """验证执行结果"""
        success = validate_result(result, expected)
        return Verification(
            success=success,
            actual_outcome=str(result),
            expected_outcome=expected,
            differences=[] if success else ["Mismatch found"]
        )

# 使用循环
loop = MyAutonomousLoop()
result = await loop.run("Generate tests for UserService.java")
```

### 4. 配置管理 - core.config

**推荐导入:**
```python
from pyutagent.core.config import (
    Settings,
    LLMConfig,
    LLMConfigCollection,
    LLMProvider,
    AiderConfig,
    ProjectPaths,
    CoverageSettings,
    get_settings,
    load_llm_config,
    save_llm_config,
)
```

**基本使用:**
```python
# 获取全局设置
settings = get_settings()

# 创建LLM配置
llm_config = LLMConfig(
    name="OpenAI GPT-4",
    provider=LLMProvider.OPENAI,
    model="gpt-4",
    api_key=SecretStr("sk-..."),
    temperature=0.7
)

# 管理配置集合
config_collection = LLMConfigCollection()
config_collection.add_config(llm_config)
config_collection.set_default_config(llm_config.id)

# 保存和加载配置
save_llm_config(config_collection)
loaded_config = load_llm_config()
```

### 5. 执行器 - StepExecutor

**推荐导入:**
```python
from pyutagent.agent.execution.executor import StepExecutor, ExecutionResult
from pyutagent.agent.execution.execution_plan import ExecutionPlan, Step, StepStatus
from pyutagent.core.agent_state import StateManager
from pyutagent.core.agent_context import AgentContext
```

**基本使用:**
```python
# 创建执行器
state_manager = StateManager()
context = AgentContext()
executor = StepExecutor(state_manager, context)

# 注册步骤处理器
async def handle_generate_step(params: Dict[str, Any], context: AgentContext):
    # 执行生成逻辑
    return {"status": "success", "output": "generated_code"}

executor.register_handler("generate", handle_generate_step)

# 创建执行计划
plan = ExecutionPlan()
plan.add_step(Step(
    id="step1",
    name="Generate Test",
    step_type=StepType.GENERATION,
    params={"target": "UserService.java"}
))

# 执行计划
results = await executor.execute_plan(plan, parallel=False)
```

### 6. 状态管理 - StateManager

**推荐导入:**
```python
from pyutagent.agent.core.agent_state import StateManager, AgentState
```

**基本使用:**
```python
# 创建状态管理器
state_manager = StateManager()

# 状态转换
state_manager.start_task()
state_manager.set_status(AgentState.GENERATING)

# 检查状态
if state_manager.is_active():
    # 执行工作
    pass

# 完成或失败
state_manager.complete_task()
# 或
state_manager.fail_task()
```

### 7. 上下文管理 - UnifiedContextManager

**推荐导入:**
```python
from pyutagent.agent.unified_context_manager import (
    UnifiedContextManager,
    ContextScope,
    ContextVisibility,
    CompressionStrategy
)
```

**基本使用:**
```python
# 创建上下文管理器
context_manager = UnifiedContextManager()

# 设置上下文值
context_manager.set(
    key="target_class",
    value="UserService",
    scope=ContextScope.TASK,
    visibility=ContextVisibility.PUBLIC
)

# 获取上下文值
value = context_manager.get("target_class")

# 压缩上下文
compressed = context_manager.compress_context(
    strategy=CompressionStrategy.SMART,
    max_tokens=4000
)
```

## 接口选择指南

### 何时使用 UnifiedAgentBase？

- 创建新的Agent实现
- 需要统一的生命周期管理
- 需要进度报告和状态跟踪
- 需要能力声明系统

### 何时使用 UnifiedAutonomousLoop？

- 实现自主决策循环
- 需要 O-T-A-V-L 模式
- 需要自我纠正机制
- 需要从执行中学习

### 何时使用 StepExecutor？

- 执行步骤化任务
- 需要步骤重试机制
- 需要并行执行支持
- 需要执行计划管理

## 最佳实践

### 1. 配置管理

```python
# 推荐：使用 dataclass 配置
from dataclasses import dataclass

@dataclass
class MyAgentConfig:
    name: str
    max_iterations: int = 10
    timeout: int = 300

# 不推荐：使用字典配置
config = {"name": "MyAgent", "max_iterations": 10}  # 避免
```

### 2. 错误处理

```python
# 推荐：使用统一的 AgentResult
async def execute(self, task: Any) -> AgentResult:
    try:
        result = await do_work(task)
        return AgentResult(success=True, output=result)
    except Exception as e:
        return AgentResult(
            success=False,
            error=str(e),
            state=AgentState.FAILED
        )

# 不推荐：返回原始值或抛出异常
async def execute(self, task: Any):  # 避免
    return await do_work(task)  # 避免
```

### 3. 进度报告

```python
# 推荐：使用进度回调
class MyAgent(UnifiedAgentBase):
    async def execute(self, task: Any) -> AgentResult:
        for i, item in enumerate(items):
            progress = (i + 1) / len(items)
            self._update_progress(
                progress=progress,
                status="processing",
                message=f"Processing item {i+1}/{len(items)}"
            )
            await process(item)
```

### 4. 状态检查

```python
# 推荐：定期检查是否应该继续
async def long_running_task(self):
    while self._should_continue():
        await self._check_pause()
        await do_work()
        self._current_iteration += 1
```

## 迁移检查清单

- [ ] 替换 `EventBus` / `AsyncEventBus` → `UnifiedMessageBus`
- [ ] 替换 `BaseAgent` / `SubAgent` → `UnifiedAgentBase`
- [ ] 替换 `AutonomousLoop` / `EnhancedAutonomousLoop` → `UnifiedAutonomousLoop`
- [ ] 替换 `BatchExecutor` / `ParallelExecutor` → `StepExecutor`
- [ ] 替换分散的配置类 → `pyutagent.core.config`
- [ ] 运行测试确保功能正常
- [ ] 更新文档和注释

## 相关文档

- [迁移指南](migration_guide.md) - 详细的迁移步骤
- [API参考](api_reference.md) - 完整的API文档
- [架构设计](architecture.md) - 系统架构说明
