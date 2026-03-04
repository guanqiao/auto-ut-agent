# PyUT Agent TDD 开发计划

本文档基于 TDD（Test-Driven Development）方法，按照"Red-Green-Refactor"循环，分阶段实施架构优化。

---

## 🎯 TDD 核心原则

1. **测试先行**：在编写功能代码之前先写测试
2. **小步快跑**：每个测试只验证一个小功能
3. **快速反馈**：测试应该快速执行，提供即时反馈
4. **持续重构**：测试通过后立即重构，保持代码质量

---

## 📋 开发阶段总览

### 阶段 1：基础架构与事件总线（P0）- 2 周
### 阶段 2：统一状态管理（P0）- 1 周
### 阶段 3：增量式修复（P1）- 1 周
### 阶段 4：性能优化（P2）- 1 周
### 阶段 5：质量提升（P3）- 1 周

---

## 📅 阶段 1：基础架构与事件总线（P0）

### 迭代 1.1：事件总线基础

#### 测试 1.1.1：事件总线基本功能

**测试文件**: `tests/unit/agent/test_event_bus.py`

```python
"""测试事件总线基础功能"""
import pytest
from pyutagent.core.event_bus import EventBus, Event

class TestEventBusBasic:
    """事件总线基础测试"""
    
    def test_create_event_bus(self):
        """测试创建事件总线"""
        bus = EventBus()
        assert bus is not None
    
    def test_subscribe_and_publish(self):
        """测试订阅和发布"""
        bus = EventBus()
        received_events = []
        
        # 定义事件处理器
        def handler(event: Event):
            received_events.append(event)
        
        # 订阅事件
        bus.subscribe(str, handler)
        
        # 发布事件
        test_event = "test_message"
        bus.publish(test_event)
        
        # 验证
        assert len(received_events) == 1
        assert received_events[0] == test_event
    
    def test_multiple_subscribers(self):
        """测试多个订阅者"""
        bus = EventBus()
        received_by_handler1 = []
        received_by_handler2 = []
        
        def handler1(event: Event):
            received_by_handler1.append(event)
        
        def handler2(event: Event):
            received_by_handler2.append(event)
        
        bus.subscribe(str, handler1)
        bus.subscribe(str, handler2)
        
        bus.publish("test")
        
        assert len(received_by_handler1) == 1
        assert len(received_by_handler2) == 1
    
    def test_unsubscribe(self):
        """测试取消订阅"""
        bus = EventBus()
        received = []
        
        def handler(event: Event):
            received.append(event)
        
        subscription = bus.subscribe(str, handler)
        bus.unsubscribe(subscription)
        bus.publish("test")
        
        assert len(received) == 0
    
    def test_publish_without_subscribers(self):
        """测试没有订阅者时发布事件"""
        bus = EventBus()
        # 不应该抛出异常
        bus.publish("test")
```

**实现代码**: `pyutagent/core/event_bus.py`

```python
"""事件总线 - 实现组件解耦"""
from typing import Any, Callable, Dict, List, Type, TypeVar
from dataclasses import dataclass, field
import uuid

T = TypeVar('T')

@dataclass
class Subscription:
    """订阅"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    event_type: Type[Any] = None
    handler: Callable[[Any], None] = None

class EventBus:
    """事件总线"""
    
    def __init__(self):
        self._subscriptions: Dict[Type[Any], List[Subscription]] = {}
    
    def subscribe(
        self,
        event_type: Type[T],
        handler: Callable[[T], None]
    ) -> Subscription:
        """订阅事件"""
        subscription = Subscription(event_type=event_type, handler=handler)
        
        if event_type not in self._subscriptions:
            self._subscriptions[event_type] = []
        
        self._subscriptions[event_type].append(subscription)
        return subscription
    
    def unsubscribe(self, subscription: Subscription):
        """取消订阅"""
        event_type = subscription.event_type
        if event_type in self._subscriptions:
            self._subscriptions[event_type] = [
                s for s in self._subscriptions[event_type]
                if s.id != subscription.id
            ]
    
    def publish(self, event: Any):
        """发布事件"""
        event_type = type(event)
        
        if event_type in self._subscriptions:
            for subscription in self._subscriptions[event_type]:
                try:
                    subscription.handler(event)
                except Exception as e:
                    # 记录错误但不影响其他处理器
                    import logging
                    logging.error(f"Event handler error: {e}")
```

**运行测试**:
```bash
pytest tests/unit/agent/test_event_bus.py::TestEventBusBasic -v
```

---

#### 测试 1.1.2：异步事件总线

**测试文件**: `tests/unit/agent/test_event_bus.py`

```python
"""测试异步事件总线"""
import pytest
import asyncio
from pyutagent.core.event_bus import AsyncEventBus

class TestAsyncEventBus:
    """异步事件总线测试"""
    
    @pytest.mark.asyncio
    async def test_async_subscribe_and_publish(self):
        """测试异步订阅和发布"""
        bus = AsyncEventBus()
        received_events = []
        
        async def handler(event: str):
            received_events.append(event)
        
        bus.subscribe(str, handler)
        await bus.publish("test")
        
        assert len(received_events) == 1
    
    @pytest.mark.asyncio
    async def test_concurrent_publish(self):
        """测试并发发布"""
        bus = AsyncEventBus()
        received = []
        
        async def handler(event: str):
            await asyncio.sleep(0.01)
            received.append(event)
        
        bus.subscribe(str, handler)
        
        # 并发发布 10 个事件
        tasks = [bus.publish(f"event_{i}") for i in range(10)]
        await asyncio.gather(*tasks)
        
        assert len(received) == 10
    
    @pytest.mark.asyncio
    async def test_handler_error_isolation(self):
        """测试处理器错误隔离"""
        bus = AsyncEventBus()
        successful_handler_called = False
        
        async def failing_handler(event: str):
            raise ValueError("Test error")
        
        async def successful_handler(event: str):
            nonlocal successful_handler_called
            successful_handler_called = True
        
        bus.subscribe(str, failing_handler)
        bus.subscribe(str, successful_handler)
        
        # 不应该因为一个处理器失败而影响其他处理器
        await bus.publish("test")
        assert successful_handler_called
```

**实现代码**: `pyutagent/core/event_bus.py`

```python
class AsyncEventBus:
    """异步事件总线"""
    
    def __init__(self):
        self._subscriptions: Dict[Type[Any], List[Subscription]] = {}
    
    def subscribe(
        self,
        event_type: Type[T],
        handler: Callable[[T], Any]  # 可以是同步或异步函数
    ) -> Subscription:
        """订阅事件"""
        subscription = Subscription(event_type=event_type, handler=handler)
        
        if event_type not in self._subscriptions:
            self._subscriptions[event_type] = []
        
        self._subscriptions[event_type].append(subscription)
        return subscription
    
    async def publish(self, event: Any):
        """异步发布事件"""
        event_type = type(event)
        
        if event_type in self._subscriptions:
            tasks = []
            for subscription in self._subscriptions[event_type]:
                task = self._invoke_handler(subscription.handler, event)
                tasks.append(task)
            
            # 并发执行所有处理器
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # 记录错误
            for result in results:
                if isinstance(result, Exception):
                    import logging
                    logging.error(f"Async event handler error: {result}")
    
    async def _invoke_handler(
        self,
        handler: Callable,
        event: Any
    ) -> Any:
        """调用处理器（支持同步和异步）"""
        if asyncio.iscoroutinefunction(handler):
            return await handler(event)
        else:
            return handler(event)
```

**运行测试**:
```bash
pytest tests/unit/agent/test_event_bus.py::TestAsyncEventBus -v
```

---

### 迭代 1.2：组件接口标准化

#### 测试 1.2.1：组件协议定义

**测试文件**: `tests/unit/agent/test_component_protocol.py`

```python
"""测试组件协议"""
import pytest
from typing import List
from pyutagent.core.component_protocol import IAgentComponent, ComponentBase

class TestComponentProtocol:
    """组件协议测试"""
    
    def test_component_base_implementation(self):
        """测试组件基类实现"""
        component = ConcreteComponent("test_component")
        
        assert component.name == "test_component"
        assert component.is_initialized is False
        
        # 测试初始化
        component.initialize()
        assert component.is_initialized is True
        
        # 测试关闭
        component.shutdown()
        assert component.is_initialized is False
    
    def test_component_capabilities(self):
        """测试组件能力声明"""
        component = ConcreteComponent("test_component")
        capabilities = component.get_capabilities()
        
        assert isinstance(capabilities, list)
        assert "test_capability" in capabilities
    
    def test_component_lifecycle(self):
        """测试组件生命周期"""
        component = ConcreteComponent("test_component")
        
        # 初始状态
        assert component.lifecycle_state == "CREATED"
        
        # 初始化
        component.initialize()
        assert component.lifecycle_state == "INITIALIZED"
        
        # 运行
        component.start()
        assert component.lifecycle_state == "RUNNING"
        
        # 停止
        component.stop()
        assert component.lifecycle_state == "STOPPED"
        
        # 关闭
        component.shutdown()
        assert component.lifecycle_state == "SHUTDOWN"

class ConcreteComponent(ComponentBase):
    """具体组件实现（用于测试）"""
    
    def __init__(self, name: str):
        super().__init__(name)
    
    def get_capabilities(self) -> List[str]:
        return ["test_capability"]
    
    def _do_initialize(self):
        pass
    
    def _do_shutdown(self):
        pass
    
    def _do_start(self):
        pass
    
    def _do_stop(self):
        pass
```

**实现代码**: `pyutagent/core/component_protocol.py`

```python
"""组件协议 - 标准化组件接口"""
from abc import ABC, abstractmethod
from typing import List
from enum import Enum

class ComponentLifecycle(Enum):
    """组件生命周期状态"""
    CREATED = "created"
    INITIALIZED = "initialized"
    RUNNING = "running"
    STOPPED = "stopped"
    SHUTDOWN = "shutdown"

class IAgentComponent(ABC):
    """组件协议"""
    
    @property
    @abstractmethod
    def name(self) -> str:
        """组件名称"""
        pass
    
    @abstractmethod
    async def initialize(self) -> None:
        """初始化组件"""
        pass
    
    @abstractmethod
    async def shutdown(self) -> None:
        """关闭组件"""
        pass
    
    @abstractmethod
    def get_capabilities(self) -> List[str]:
        """获取组件能力列表"""
        pass
    
    @property
    @abstractmethod
    def lifecycle_state(self) -> ComponentLifecycle:
        """当前生命周期状态"""
        pass

class ComponentBase(IAgentComponent):
    """组件基类，提供通用实现"""
    
    def __init__(self, name: str):
        self._name = name
        self._lifecycle_state = ComponentLifecycle.CREATED
        self._is_initialized = False
    
    @property
    def name(self) -> str:
        return self._name
    
    @property
    def lifecycle_state(self) -> ComponentLifecycle:
        return self._lifecycle_state
    
    @property
    def is_initialized(self) -> bool:
        return self._is_initialized
    
    async def initialize(self) -> None:
        """初始化组件"""
        if self._is_initialized:
            return
        
        await self._do_initialize()
        self._is_initialized = True
        self._lifecycle_state = ComponentLifecycle.INITIALIZED
    
    async def shutdown(self) -> None:
        """关闭组件"""
        if not self._is_initialized:
            return
        
        await self._do_shutdown()
        self._is_initialized = False
        self._lifecycle_state = ComponentLifecycle.SHUTDOWN
    
    async def start(self) -> None:
        """启动组件"""
        if not self._is_initialized:
            raise RuntimeError("Component not initialized")
        
        await self._do_start()
        self._lifecycle_state = ComponentLifecycle.RUNNING
    
    async def stop(self) -> None:
        """停止组件"""
        await self._do_stop()
        self._lifecycle_state = ComponentLifecycle.STOPPED
    
    @abstractmethod
    def get_capabilities(self) -> List[str]:
        """获取组件能力"""
        pass
    
    @abstractmethod
    def _do_initialize(self) -> None:
        """执行初始化"""
        pass
    
    @abstractmethod
    def _do_shutdown(self) -> None:
        """执行关闭"""
        pass
    
    @abstractmethod
    def _do_start(self) -> None:
        """执行启动"""
        pass
    
    @abstractmethod
    def _do_stop(self) -> None:
        """执行停止"""
        pass
```

**运行测试**:
```bash
pytest tests/unit/agent/test_component_protocol.py -v
```

---

### 迭代 1.3：分层依赖注入

#### 测试 1.3.1：分层容器

**测试文件**: `tests/unit/core/test_layered_container.py`

```python
"""测试分层依赖注入容器"""
import pytest
from pyutagent.core.layered_container import LayeredContainer, Layer

class TestLayeredContainer:
    """分层容器测试"""
    
    def test_create_layered_container(self):
        """测试创建分层容器"""
        container = LayeredContainer()
        assert container is not None
    
    def test_register_in_layer(self):
        """测试在指定层注册服务"""
        container = LayeredContainer()
        
        # 在 P0 层注册
        container.register(
            layer=Layer.P0,
            service="service_a",
            implementation=ServiceA
        )
        
        # 从 P0 层解析
        service = container.resolve("service_a", Layer.P0)
        assert service is not None
        assert isinstance(service, ServiceA)
    
    def test_cross_layer_resolution(self):
        """测试跨层解析"""
        container = LayeredContainer()
        
        # 在 P0 层注册基础服务
        container.register(
            layer=Layer.P0,
            service="base_service",
            implementation=BaseService
        )
        
        # 在 P1 层注册依赖基础服务的服务
        container.register(
            layer=Layer.P1,
            service="advanced_service",
            implementation=lambda: AdvancedService(
                container.resolve("base_service", Layer.P0)
            )
        )
        
        # 解析 P1 层服务（应该自动解析 P0 层依赖）
        service = container.resolve("advanced_service", Layer.P1)
        assert service is not None
        assert isinstance(service, AdvancedService)
    
    def test_layer_isolation(self):
        """测试层隔离"""
        container = LayeredContainer()
        
        # 在 P0 层注册
        container.register(
            layer=Layer.P0,
            service="p0_service",
            implementation=ServiceA
        )
        
        # 尝试从 P1 层解析（应该失败或返回 None）
        with pytest.raises(KeyError):
            container.resolve("p0_service", Layer.P1)
    
    def test_singleton_in_layer(self):
        """测试层内单例"""
        container = LayeredContainer()
        
        container.register(
            layer=Layer.P0,
            service="singleton_service",
            implementation=ServiceA,
            singleton=True
        )
        
        # 两次解析应该返回同一个实例
        service1 = container.resolve("singleton_service", Layer.P0)
        service2 = container.resolve("singleton_service", Layer.P0)
        
        assert service1 is service2

class ServiceA:
    pass

class BaseService:
    pass

class AdvancedService:
    def __init__(self, base_service: BaseService):
        self.base_service = base_service
```

**实现代码**: `pyutagent/core/layered_container.py`

```python
"""分层依赖注入容器"""
from enum import Enum
from typing import Any, Callable, Dict, Optional, Type

class Layer(Enum):
    """层定义"""
    P0 = "p0"  # 核心层
    P1 = "p1"  # 增强层
    P2 = "p2"  # 协作层
    P3 = "p3"  # 高级层

class LayeredContainer:
    """分层依赖注入容器"""
    
    def __init__(self):
        self._layers: Dict[Layer, Dict[str, Any]] = {}
        self._singletons: Dict[str, Any] = {}
        
        # 初始化各层
        for layer in Layer:
            self._layers[layer] = {}
    
    def register(
        self,
        layer: Layer,
        service: str,
        implementation: Any,
        singleton: bool = False
    ):
        """注册服务"""
        layer_dict = self._layers[layer]
        
        if singleton:
            # 立即创建单例
            instance = implementation() if callable(implementation) else implementation
            self._singletons[f"{layer.value}:{service}"] = instance
            layer_dict[service] = implementation
        else:
            layer_dict[service] = implementation
    
    def resolve(
        self,
        service: str,
        layer: Layer,
        dependencies: Optional[Dict[str, Any]] = None
    ) -> Any:
        """解析服务"""
        layer_dict = self._layers[layer]
        
        if service not in layer_dict:
            raise KeyError(f"Service '{service}' not found in layer {layer.value}")
        
        # 检查是否是单例
        singleton_key = f"{layer.value}:{service}"
        if singleton_key in self._singletons:
            return self._singletons[singleton_key]
        
        implementation = layer_dict[service]
        
        # 如果是可调用对象，创建实例
        if callable(implementation):
            # 注入依赖
            if dependencies:
                return implementation(**dependencies)
            else:
                return implementation()
        
        return implementation
```

**运行测试**:
```bash
pytest tests/unit/core/test_layered_container.py -v
```

---

## 📅 阶段 2：统一状态管理（P0）

### 迭代 2.1：状态存储

#### 测试 2.1.1：状态存储基础

**测试文件**: `tests/unit/core/test_state_store.py`

```python
"""测试状态存储"""
import pytest
from pyutagent.core.state_store import StateStore, AgentState, Action

class TestStateStore:
    """状态存储测试"""
    
    def test_create_state_store(self):
        """测试创建状态存储"""
        store = StateStore()
        assert store is not None
    
    def test_get_initial_state(self):
        """测试获取初始状态"""
        store = StateStore()
        state = store.get_state()
        
        assert state.lifecycle_state == "IDLE"
        assert state.current_iteration == 0
    
    def test_dispatch_action(self):
        """测试分发 action"""
        store = StateStore()
        
        # 创建 action
        action = UpdateIterationAction(iteration=5)
        
        # 分发 action
        store.dispatch(action)
        
        # 验证状态更新
        state = store.get_state()
        assert state.current_iteration == 5
    
    def test_state_subscriber(self):
        """测试状态订阅者"""
        store = StateStore()
        received_states = []
        
        def subscriber(state: AgentState):
            received_states.append(state)
        
        store.subscribe(subscriber)
        
        # 分发 action
        store.dispatch(UpdateIterationAction(iteration=3))
        
        # 验证订阅者收到通知（初始状态 + 更新后状态）
        assert len(received_states) >= 1

class UpdateIterationAction(Action):
    """更新迭代次数的 action"""
    
    def __init__(self, iteration: int):
        self.iteration = iteration
    
    def reduce(self, state: AgentState) -> AgentState:
        return AgentState(
            **{**state.__dict__, 'current_iteration': self.iteration}
        )
```

**实现代码**: `pyutagent/core/state_store.py`

```python
"""统一状态管理"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Callable, List, Dict, Any
from enum import Enum

class LifecycleState(Enum):
    """生命周期状态"""
    IDLE = "idle"
    RUNNING = "running"
    PAUSED = "paused"
    TERMINATED = "terminated"

@dataclass
class AgentState:
    """Agent 状态"""
    lifecycle_state: LifecycleState = LifecycleState.IDLE
    current_phase: str = "IDLE"
    current_iteration: int = 0
    target_coverage: float = 0.8
    current_coverage: float = 0.0
    working_memory: Dict[str, Any] = field(default_factory=dict)
    error_state: Dict[str, Any] = field(default_factory=dict)
    metrics: Dict[str, Any] = field(default_factory=dict)
    state_history: List[Dict[str, Any]] = field(default_factory=list)

class Action(ABC):
    """Action 基类"""
    
    @abstractmethod
    def reduce(self, state: AgentState) -> AgentState:
        """将 action 应用到状态上"""
        pass

class StateStore:
    """状态存储"""
    
    def __init__(self, initial_state: AgentState = None):
        self._state = initial_state or AgentState()
        self._listeners: List[Callable[[AgentState], None]] = []
    
    def get_state(self) -> AgentState:
        """获取当前状态"""
        return self._state
    
    def dispatch(self, action: Action):
        """分发 action"""
        old_state = self._state
        self._state = action.reduce(self._state)
        
        # 记录状态历史
        self._state.state_history.append({
            'from': old_state,
            'to': self._state,
            'action': action.__class__.__name__
        })
        
        # 通知订阅者
        self._notify_listeners()
    
    def subscribe(self, listener: Callable[[AgentState], None]):
        """订阅状态变化"""
        self._listeners.append(listener)
        # 立即通知当前状态
        listener(self._state)
    
    def _notify_listeners(self):
        """通知所有订阅者"""
        for listener in self._listeners:
            try:
                listener(self._state)
            except Exception as e:
                import logging
                logging.error(f"State listener error: {e}")
```

**运行测试**:
```bash
pytest tests/unit/core/test_state_store.py -v
```

---

## 📅 阶段 3：增量式修复（P1）

### 迭代 3.1：测试失败分析

#### 测试 3.1.1：失败聚类

**测试文件**: `tests/unit/agent/test_incremental_fixer.py`

```python
"""测试增量式修复"""
import pytest
from pyutagent.agent.incremental_fixer import IncrementalFixer, TestFailureCluster

class TestIncrementalFixer:
    """增量式修复测试"""
    
    def test_group_failures_by_type(self):
        """测试按失败类型分组"""
        fixer = IncrementalFixer()
        
        test_results = TestResults(
            failures=[
                TestFailure("test1", "AssertionError", "Expected 1 but got 2"),
                TestFailure("test2", "AssertionError", "Expected 3 but got 4"),
                TestFailure("test3", "NullPointerException", "obj is null"),
            ]
        )
        
        groups = fixer.group_failures_by_type(test_results)
        
        assert len(groups) == 2
        assert len(groups["AssertionError"]) == 2
        assert len(groups["NullPointerException"]) == 1
    
    def test_cluster_by_root_cause(self):
        """测试按根本原因聚类"""
        fixer = IncrementalFixer()
        
        failures = [
            TestFailure("test1", "AssertionError", "Expected 1 but got 2"),
            TestFailure("test2", "AssertionError", "Expected 1 but got 3"),
            TestFailure("test3", "AssertionError", "obj is null"),
        ]
        
        clusters = fixer.cluster_by_root_cause(failures)
        
        # 应该聚类成 2 组（相似的错误消息）
        assert len(clusters) == 2
    
    def test_generate_targeted_fix(self):
        """测试生成针对性修复"""
        fixer = IncrementalFixer(llm_client=MockLLMClient())
        
        cluster = TestFailureCluster(
            failures=[TestFailure("test1", "AssertionError", "Expected 1 but got 2")],
            root_cause="Incorrect return value"
        )
        
        current_code = """
public int add(int a, int b) {
    return a - b;  // Bug: should be a + b
}
"""
        
        fixed_code = fixer.generate_targeted_fix(cluster, current_code)
        
        assert "a + b" in fixed_code
        assert "a - b" not in fixed_code
```

**实现代码**: `pyutagent/agent/incremental_fixer.py`

```python
"""增量式修复器"""
from typing import List, Dict
from pyutagent.llm.client import LLMClient

class TestFailure:
    """测试失败"""
    
    def __init__(self, test_name: str, error_type: str, message: str):
        self.test_name = test_name
        self.error_type = error_type
        self.message = message

class TestFailureCluster:
    """测试失败聚类"""
    
    def __init__(
        self,
        failures: List[TestFailure],
        root_cause: str
    ):
        self.failures = failures
        self.root_cause = root_cause

class IncrementalFixer:
    """增量式修复器"""
    
    def __init__(self, llm_client: LLMClient = None):
        self.llm_client = llm_client
    
    def group_failures_by_type(
        self,
        test_results: 'TestResults'
    ) -> Dict[str, List[TestFailure]]:
        """按失败类型分组"""
        groups: Dict[str, List[TestFailure]] = {}
        
        for failure in test_results.failures:
            error_type = failure.error_type
            if error_type not in groups:
                groups[error_type] = []
            groups[error_type].append(failure)
        
        return groups
    
    def cluster_by_root_cause(
        self,
        failures: List[TestFailure]
    ) -> List[TestFailureCluster]:
        """按根本原因聚类"""
        # 使用简单的消息相似度聚类
        clusters = []
        
        for failure in failures:
            # 查找相似的聚类
            similar_cluster = None
            for cluster in clusters:
                if self._is_similar(failure, cluster.failures[0]):
                    similar_cluster = cluster
                    break
            
            if similar_cluster:
                similar_cluster.failures.append(failure)
            else:
                clusters.append(TestFailureCluster(
                    failures=[failure],
                    root_cause=failure.message
                ))
        
        return clusters
    
    def _is_similar(
        self,
        failure1: TestFailure,
        failure2: TestFailure
    ) -> bool:
        """判断两个失败是否相似"""
        # 简单的相似度：错误类型相同且消息相似度>0.7
        if failure1.error_type != failure2.error_type:
            return False
        
        # 使用简单的字符串相似度
        similarity = self._string_similarity(
            failure1.message,
            failure2.message
        )
        return similarity > 0.7
    
    def _string_similarity(self, s1: str, s2: str) -> float:
        """计算字符串相似度"""
        # 使用 Levenshtein 距离
        from difflib import SequenceMatcher
        return SequenceMatcher(None, s1, s2).ratio()
    
    async def generate_targeted_fix(
        self,
        cluster: TestFailureCluster,
        current_code: str
    ) -> str:
        """生成针对性修复"""
        prompt = f"""Fix the following code to address these test failures:

Failures:
{self._format_failures(cluster.failures)}

Root Cause: {cluster.root_cause}

Current Code:
```java
{current_code}
```

Output only the fixed code:"""
        
        response = await self.llm_client.agenerate(prompt)
        return self._extract_code(response)
    
    def _format_failures(self, failures: List[TestFailure]) -> str:
        """格式化失败信息"""
        lines = []
        for f in failures:
            lines.append(f"- {f.test_name}: {f.error_type} - {f.message}")
        return "\n".join(lines)
    
    def _extract_code(self, response: str) -> str:
        """从响应中提取代码"""
        import re
        match = re.search(r'```java\s*(.*?)\s*```', response, re.DOTALL)
        if match:
            return match.group(1)
        return response
```

**运行测试**:
```bash
pytest tests/unit/agent/test_incremental_fixer.py -v
```

---

## 📅 阶段 4：性能优化（P2）

### 迭代 4.1：LLM 调用优化

#### 测试 4.1.1：Prompt 缓存

**测试文件**: `tests/unit/llm/test_prompt_cache.py`

```python
"""测试 Prompt 缓存"""
import pytest
import asyncio
from pyutagent.llm.prompt_cache import PromptCache

class TestPromptCache:
    """Prompt 缓存测试"""
    
    @pytest.mark.asyncio
    async def test_cache_hit(self):
        """测试缓存命中"""
        cache = PromptCache()
        mock_llm = MockLLM()
        
        # 第一次调用（缓存未命中）
        result1 = await cache.get_or_generate(
            prompt="test prompt",
            system_prompt="test system",
            llm_client=mock_llm
        )
        
        # 第二次调用（缓存命中）
        result2 = await cache.get_or_generate(
            prompt="test prompt",
            system_prompt="test system",
            llm_client=mock_llm
        )
        
        # 两次结果应该相同
        assert result1 == result2
        # LLM 只应该被调用一次
        assert mock_llm.call_count == 1
    
    @pytest.mark.asyncio
    async def test_cache_miss(self):
        """测试缓存未命中"""
        cache = PromptCache()
        mock_llm = MockLLM()
        
        # 不同的 prompt 应该导致缓存未命中
        await cache.get_or_generate(
            prompt="prompt 1",
            system_prompt="system",
            llm_client=mock_llm
        )
        
        await cache.get_or_generate(
            prompt="prompt 2",
            system_prompt="system",
            llm_client=mock_llm
        )
        
        assert mock_llm.call_count == 2
    
    @pytest.mark.asyncio
    async def test_cache_eviction(self):
        """测试缓存淘汰"""
        cache = PromptCache(capacity=2)
        mock_llm = MockLLM()
        
        # 添加 3 个不同的 prompt（超过容量）
        await cache.get_or_generate("prompt 1", "system", mock_llm)
        await cache.get_or_generate("prompt 2", "system", mock_llm)
        await cache.get_or_generate("prompt 3", "system", mock_llm)
        
        # 最早添加的应该被淘汰
        assert len(cache._cache) <= 2

class MockLLM:
    def __init__(self):
        self.call_count = 0
    
    async def agenerate(self, prompt, system_prompt):
        self.call_count += 1
        return f"Response to {prompt}"
```

**实现代码**: `pyutagent/llm/prompt_cache.py`

```python
"""Prompt 缓存"""
from typing import Dict, Any
import hashlib
from collections import OrderedDict
from pyutagent.llm.client import LLMClient

class PromptCache:
    """Prompt 结果缓存"""
    
    def __init__(self, capacity: int = 1000):
        self._cache: OrderedDict[str, str] = OrderedDict()
        self._capacity = capacity
    
    def _generate_key(
        self,
        prompt: str,
        system_prompt: str
    ) -> str:
        """生成缓存键"""
        key_data = f"{system_prompt}|||{prompt}"
        return hashlib.sha256(key_data.encode()).hexdigest()
    
    async def get_or_generate(
        self,
        prompt: str,
        system_prompt: str,
        llm_client: LLMClient
    ) -> str:
        """获取或生成"""
        key = self._generate_key(prompt, system_prompt)
        
        # 检查缓存
        if key in self._cache:
            return self._cache[key]
        
        # 生成
        response = await llm_client.agenerate(prompt, system_prompt)
        
        # 缓存结果
        self._cache[key] = response
        self._cache.move_to_end(key)
        
        # 淘汰最旧的
        if len(self._cache) > self._capacity:
            self._cache.popitem(last=False)
        
        return response
```

**运行测试**:
```bash
pytest tests/unit/llm/test_prompt_cache.py -v
```

---

## 📅 阶段 5：质量提升（P3）

### 迭代 5.1：类型系统完善

#### 测试 5.1.1：类型注解测试

**测试文件**: `tests/unit/core/test_types.py`

```python
"""测试类型系统"""
import pytest
from typing import get_type_hints
from pyutagent.core.types import (
    FilePath,
    ClassName,
    MethodName,
    CoveragePercentage,
    CompilationResultDict
)

class TestTypes:
    """类型系统测试"""
    
    def test_new_types(self):
        """测试 NewType"""
        file_path: FilePath = FilePath("/path/to/file.java")
        assert isinstance(file_path, str)
        
        class_name: ClassName = ClassName("MyClass")
        assert isinstance(class_name, str)
    
    def test_typed_dict(self):
        """测试 TypedDict"""
        result: CompilationResultDict = {
            "success": True,
            "output": "Build successful",
            "errors": [],
            "duration": 1.5
        }
        
        # 验证类型
        hints = get_type_hints(CompilationResultDict)
        assert hints["success"] == bool
        assert hints["output"] == str
        assert hints["errors"] == list
        assert hints["duration"] == float
```

**实现代码**: `pyutagent/core/types.py`

```python
"""类型定义"""
from typing import NewType, TypedDict, List, Optional, NotRequired

# NewType 定义
FilePath = NewType('FilePath', str)
ClassName = NewType('ClassName', str)
MethodName = NewType('MethodName', str)
CoveragePercentage = NewType('CoveragePercentage', float)

# TypedDict 定义
class CompilationResultDict(TypedDict):
    """编译结果"""
    success: bool
    output: str
    errors: List[str]
    duration: float
    is_incremental: NotRequired[bool]

class TestResultDict(TypedDict):
    """测试结果"""
    total: int
    passed: int
    failed: int
    errors: int
    duration: float
```

**运行测试**:
```bash
pytest tests/unit/core/test_types.py -v
```

---

## 🧪 测试执行策略

### 测试分层

1. **单元测试** (Unit Tests)
   - 位置：`tests/unit/`
   - 目标：测试单个函数/类
   - 执行时间：< 1 秒/测试
   - 覆盖率目标：> 80%

2. **集成测试** (Integration Tests)
   - 位置：`tests/integration/`
   - 目标：测试组件间交互
   - 执行时间：< 10 秒/测试
   - 覆盖率目标：> 60%

3. **端到端测试** (E2E Tests)
   - 位置：`tests/e2e/`
   - 目标：测试完整流程
   - 执行时间：< 1 分钟/测试
   - 覆盖率目标：> 40%

### 测试执行命令

```bash
# 运行所有单元测试
pytest tests/unit/ -v --cov=pyutagent --cov-report=html

# 运行特定模块测试
pytest tests/unit/core/ -v

# 运行并生成覆盖率报告
pytest tests/unit/ --cov=pyutagent --cov-report=term-missing

# 运行快速测试（跳过慢测试）
pytest tests/unit/ -m "not slow"

# 持续测试（文件变化时自动运行）
ptw --now .
```

### CI/CD 集成

```yaml
# .github/workflows/test.yml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: [3.9, 3.10, 3.11]
    
    steps:
    - uses: actions/checkout@v2
    
    - name: Set up Python
      uses: actions/setup-python@v2
      with:
        python-version: ${{ matrix.python-version }}
    
    - name: Install dependencies
      run: |
        pip install -r requirements.txt
        pip install -r requirements-dev.txt
    
    - name: Run tests
      run: |
        pytest tests/unit/ -v --cov=pyutagent --cov-report=xml
    
    - name: Upload coverage
      uses: codecov/codecov-action@v2
      with:
        file: ./coverage.xml
```

---

## 📊 进度跟踪

### 阶段完成情况

| 阶段 | 迭代 | 测试数 | 通过数 | 覆盖率 | 状态 |
|------|------|--------|--------|--------|------|
| 阶段 1 | 1.1 | 6 | 0 | 0% | 📝 待开始 |
| 阶段 1 | 1.2 | 4 | 0 | 0% | 📝 待开始 |
| 阶段 1 | 1.3 | 5 | 0 | 0% | 📝 待开始 |
| 阶段 2 | 2.1 | 4 | 0 | 0% | 📝 待开始 |
| 阶段 3 | 3.1 | 3 | 0 | 0% | 📝 待开始 |
| 阶段 4 | 4.1 | 3 | 0 | 0% | 📝 待开始 |
| 阶段 5 | 5.1 | 2 | 0 | 0% | 📝 待开始 |

### 定义完成（DoD）

每个迭代的完成标准：
- ✅ 所有测试通过（绿色）
- ✅ 代码覆盖率 > 80%
- ✅ 代码审查通过
- ✅ 文档更新
- ✅ 无 lint 错误

---

## 🎯 下一步行动

### 立即开始

1. **创建测试文件结构**
```bash
mkdir -p tests/unit/{agent,core,llm,memory,tools}
mkdir -p tests/integration
mkdir -p tests/e2e
```

2. **安装测试依赖**
```bash
pip install pytest pytest-asyncio pytest-cov pytest-watch
```

3. **运行第一个测试**
```bash
pytest tests/unit/agent/test_event_bus.py::TestEventBusBasic::test_create_event_bus -v
```

### 开发流程

对于每个功能：
1. **写测试** (Red) - 编写失败的测试
2. **实现功能** (Green) - 编写最少的代码使测试通过
3. **重构** (Refactor) - 优化代码，保持测试通过
4. **重复** - 继续下一个功能

---

**文档版本**: 1.0  
**创建日期**: 2026-03-04  
**TDD 原则**: Red → Green → Refactor
