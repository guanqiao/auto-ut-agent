# PyUT Agent 代码整合和优化 - 详细规范 (Spec)

## 1. 项目概述

### 1.1 目标
将 PyUT Agent 代码库中的重复功能整合为统一接口，提高代码可维护性、减少冗余，并建立清晰的架构规范。

### 1.2 范围
- **完整迁移**: 将所有代码从废弃接口迁移到统一接口
- **完美整合**: 实现架构层面的深度优化

### 1.3 关键术语
| 术语 | 定义 |
|------|------|
| **统一接口** | 经整合后的标准接口，如 `UnifiedAgentBase` |
| **废弃接口** | 标记为 `@deprecated` 的旧接口 |
| **向后兼容** | 新代码支持旧接口的能力 |
| **功能回归** | 迁移后功能丢失或行为改变 |

---

## 2. 功能需求

### 2.1 完整迁移需求

#### FR-001: 核心模块迁移
- **描述**: 迁移 `core/__init__.py` 中的废弃导入
- **输入**: `EventBus`, `AsyncEventBus` 导入
- **输出**: 使用 `UnifiedMessageBus` 替代
- **验收标准**: 
  - `scripts/check_migration.py` 不报告 `core/__init__.py` 有废弃导入
  - 所有使用 `core.EventBus` 的代码正常工作

#### FR-002: Agent组件迁移
- **描述**: 迁移 `agent/components/` 中的废弃导入
- **输入**: `StepResult`, `ContextManager` 导入
- **输出**: 使用 `AgentResult`, `UnifiedContextManager` 替代
- **验收标准**:
  - `execution_steps.py`, `core_agent.py`, `agent_initialization.py` 无废弃导入
  - 组件功能等价

#### FR-003: Capability Registry迁移
- **描述**: 迁移 `capability_registry.py` 中的 `SubAgent` 使用
- **输入**: `SubAgent` 类型注解
- **输出**: 使用 `UnifiedAgentBase` 替代
- **验收标准**:
  - 类型检查通过
  - 功能等价

#### FR-004: Agent模块导出清理
- **描述**: 清理 `agent/__init__.py` 中的废弃导出
- **输入**: `BaseAgent`, `AutonomousLoop`, `ContextManager` 导出
- **输出**: 保留导出但添加废弃警告
- **验收标准**:
  - 导入时显示废弃警告
  - 向后兼容保持

### 2.2 完美整合需求

#### FR-005: 统一依赖注入
- **描述**: 创建统一的依赖注入容器
- **输入**: 分散的 Manager 类实例
- **输出**: 统一的 `DIContainer` 管理所有依赖
- **验收标准**:
  - 所有 Manager 通过 DIContainer 注册和解析
  - 单例模式正确

#### FR-006: 统一事件系统
- **描述**: 整合所有事件/消息机制
- **输入**: `UnifiedMessageBus`, `EventBus`, 回调机制
- **输出**: 统一的 `EventBus` 接口
- **验收标准**:
  - 支持同步/异步事件
  - 支持请求-响应模式
  - 向后兼容

#### FR-007: 统一配置系统
- **描述**: 所有配置集中管理
- **输入**: 分散的配置类
- **输出**: 统一的 `AppConfig` 管理所有配置
- **验收标准**:
  - 配置加载/保存功能正常
  - 配置验证通过

#### FR-008: 消除重复代码
- **描述**: 识别并消除重复代码模式
- **输入**: 多个 StateManager, ContextManager, ErrorHandler 实现
- **输出**: 单一实现
- **验收标准**:
  - 代码重复率 < 5%

---

## 3. 非功能需求

### 3.1 性能需求

| ID | 需求 | 目标 | 验证方法 |
|----|------|------|----------|
| NFR-001 | 消息总线吞吐量 | > 1000 msg/sec | 性能测试 |
| NFR-002 | Agent执行延迟 | < 100ms | 性能测试 |
| NFR-003 | 内存使用 | 无内存泄漏 | 内存分析 |
| NFR-004 | 启动时间 | < 5s | 启动测试 |

### 3.2 质量需求

| ID | 需求 | 目标 | 验证方法 |
|----|------|------|----------|
| NFR-005 | 代码覆盖率 | > 90% | 覆盖率报告 |
| NFR-006 | 代码重复率 | < 5% | 静态分析 |
| NFR-007 | 类型检查 | 无错误 | mypy |
| NFR-008 | 代码风格 | 符合PEP8 | ruff/flake8 |

### 3.3 兼容性需求

| ID | 需求 | 目标 | 验证方法 |
|----|------|------|----------|
| NFR-009 | 向后兼容 | 现有代码可运行 | 集成测试 |
| NFR-010 | 废弃警告 | 正确显示 | 测试验证 |
| NFR-011 | 适配器功能 | 正常工作 | 单元测试 |

---

## 4. 技术规范

### 4.1 统一接口规范

#### 4.1.1 UnifiedAgentBase
```python
class UnifiedAgentBase(ABC):
    """统一Agent基类规范"""
    
    def __init__(self, config: AgentConfig) -> None
    
    @abstractmethod
    async def execute(self, task: Any) -> AgentResult
    
    def get_capabilities(self) -> List[AgentCapability]
    
    def _update_progress(self, progress: float, status: str, message: str) -> None
    
    def _should_continue(self) -> bool
    
    async def _check_pause(self) -> None
```

#### 4.1.2 UnifiedMessageBus
```python
class UnifiedMessageBus:
    """统一消息总线规范"""
    
    async def register(self, entity_id: str, handler: Optional[MessageHandler] = None) -> bool
    
    async def send(self, message: Message) -> bool
    
    async def receive(self, entity_id: str, timeout: Optional[float] = None) -> Optional[Message]
    
    async def request(self, sender: str, recipient: str, payload: Dict[str, Any], 
                     timeout: float = 30.0) -> Optional[Message]
    
    async def broadcast(self, sender: str, message_type: MessageType, 
                       payload: Dict[str, Any]) -> int
```

#### 4.1.3 UnifiedAutonomousLoop
```python
class UnifiedAutonomousLoop(ABC):
    """统一自主循环规范"""
    
    def __init__(self, config: LoopConfig) -> None
    
    async def run(self, task: str, context: Optional[Dict[str, Any]] = None) -> LoopResult
    
    @abstractmethod
    async def _observe(self, task: str, context: Dict[str, Any]) -> Observation
    
    @abstractmethod
    async def _think(self, task: str, observation: Observation, context: Dict[str, Any]) -> Thought
    
    @abstractmethod
    async def _act(self, action: Action, context: Dict[str, Any]) -> Any
    
    @abstractmethod
    async def _verify(self, result: Any, expected: str, context: Dict[str, Any]) -> Verification
```

### 4.2 废弃标记规范

```python
import warnings

def deprecated_function():
    """已废弃函数示例"""
    warnings.warn(
        "deprecated_function is deprecated, use new_function instead. "
        "See: https://docs.pyutagent.org/migration",
        DeprecationWarning,
        stacklevel=2
    )
    # 实现...

# 模块级废弃警告
warnings.warn(
    "pyutagent.old_module is deprecated. "
    "Use pyutagent.new_module instead.",
    DeprecationWarning,
    stacklevel=2
)
```

### 4.3 适配器规范

```python
class LegacyAdapter:
    """向后兼容适配器规范"""
    
    def __init__(self, unified_impl: UnifiedInterface):
        warnings.warn(
            "LegacyAdapter is deprecated. Use UnifiedInterface directly.",
            DeprecationWarning,
            stacklevel=2
        )
        self._impl = unified_impl
    
    def legacy_method(self, arg: Any) -> Any:
        """适配旧接口到新接口"""
        # 转换参数
        new_arg = self._convert_arg(arg)
        # 调用新实现
        result = self._impl.new_method(new_arg)
        # 转换结果
        return self._convert_result(result)
```

---

## 5. 接口映射

### 5.1 完整迁移映射表

| 废弃接口 | 统一接口 | 迁移复杂度 | 备注 |
|----------|----------|------------|------|
| `core.event_bus.EventBus` | `core.messaging.UnifiedMessageBus` | 中 | 使用适配器 |
| `core.event_bus.AsyncEventBus` | `core.messaging.UnifiedMessageBus` | 中 | 统一接口 |
| `agent.base_agent.BaseAgent` | `agent.unified_agent_base.UnifiedAgentBase` | 高 | 需要重构 |
| `agent.base_agent.StepResult` | `agent.unified_agent_base.AgentResult` | 低 | 别名兼容 |
| `agent.subagent_base.SubAgent` | `agent.unified_agent_base.UnifiedAgentBase` | 高 | 需要重构 |
| `agent.autonomous_loop.AutonomousLoop` | `agent.unified_autonomous_loop.UnifiedAutonomousLoop` | 高 | 需要重构 |
| `agent.context_manager.ContextManager` | `agent.unified_context_manager.UnifiedContextManager` | 中 | API变化 |
| `agent.batch_executor.BatchExecutor` | `agent.execution.executor.StepExecutor` | 中 | 配置变化 |
| `agent.parallel_executor.ParallelExecutor` | `agent.execution.executor.StepExecutor` | 中 | 使用parallel参数 |

### 5.2 配置映射表

| 废弃配置 | 统一配置 | 变化说明 |
|----------|----------|----------|
| `config.project_config.ProjectConfigManager` | `core.project_config.ProjectConfigManager` | 路径变化 |
| `agent.utils.state_manager.StateManager` | `agent.core.agent_state.StateManager` | 路径变化 |
| `llm.config.LLMConfig` | `core.config.LLMConfig` | 已重定向 |

---

## 6. 数据模型

### 6.1 AgentResult 模型

```python
@dataclass
class AgentResult:
    """Agent执行结果"""
    success: bool
    output: Optional[Any] = None
    error: Optional[str] = None
    state: AgentState = AgentState.PENDING
    iterations: int = 0
    duration_ms: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
```

### 6.2 LoopConfig 模型

```python
@dataclass
class LoopConfig:
    """自主循环配置"""
    max_iterations: int = 10
    confidence_threshold: float = 0.8
    features: Set[LoopFeature] = field(default_factory=lambda: {
        LoopFeature.LLM_REASONING,
        LoopFeature.SELF_CORRECTION,
    })
    timeout: int = 300
    learning_enabled: bool = True
```

### 6.3 Message 模型

```python
@dataclass
class Message:
    """统一消息模型"""
    id: str
    sender: str
    recipient: str
    message_type: MessageType
    payload: Dict[str, Any]
    priority: MessagePriority = MessagePriority.NORMAL
    timestamp: datetime = field(default_factory=datetime.now)
    correlation_id: Optional[str] = None
```

---

## 7. 错误处理

### 7.1 错误类型

```python
class MigrationError(Exception):
    """迁移错误"""
    pass

class CompatibilityError(Exception):
    """兼容性错误"""
    pass

class IntegrationError(Exception):
    """整合错误"""
    pass
```

### 7.2 错误处理策略

1. **迁移错误**: 记录详细日志，提供迁移建议
2. **兼容性错误**: 提供适配器或降级方案
3. **整合错误**: 回滚到稳定版本，分析问题

---

## 8. 安全考虑

### 8.1 代码安全
- 所有废弃代码保留，不删除（向后兼容）
- 废弃警告包含迁移文档链接
- 适配器进行输入验证

### 8.2 配置安全
- 敏感配置使用 `SecretStr`
- 配置验证防止注入攻击
- 配置文件权限控制

---

## 9. 依赖关系

### 9.1 模块依赖图

```
core/messaging/
    └── UnifiedMessageBus (基础)
        └── agent/unified_agent_base/
            └── UnifiedAgentBase
                └── agent/unified_autonomous_loop/
                    └── UnifiedAutonomousLoop
                        └── agent/execution/
                            └── StepExecutor
```

### 9.2 迁移依赖顺序

1. 先迁移底层模块（core/messaging）
2. 再迁移中间层（agent/base）
3. 最后迁移上层（agent/loop, agent/execution）
4. 清理导出和文档

---

## 10. 附录

### 10.1 参考文档
- [迁移指南](../docs/migration_guide.md)
- [统一接口指南](../docs/unified_interfaces.md)
- [架构设计](../docs/architecture.md)

### 10.2 工具参考
- 迁移检查: `python scripts/check_migration.py`
- 测试运行: `python -m pytest tests/ -v`
- 覆盖率: `python -m pytest tests/ --cov=pyutagent`

### 10.3 变更历史
| 版本 | 日期 | 变更 |
|------|------|------|
| 1.0 | 2026-03-06 | 初始版本 |
