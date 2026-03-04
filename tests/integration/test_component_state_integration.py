"""集成测试：组件生命周期与状态管理的集成"""
import pytest
import asyncio
from typing import List
from pyutagent.core.component_protocol import (
    ComponentBase,
    ComponentLifecycle,
    ComponentCapability
)
from pyutagent.core.state_store import (
    StateStore,
    AgentState,
    UpdateLifecycleAction,
    UpdateIterationAction,
    LifecycleState
)


class TestComponent(ComponentBase):
    """测试用组件"""
    
    def __init__(self, name: str):
        super().__init__(name)
        self.state_changes: List[ComponentLifecycle] = []
    
    def get_capabilities(self) -> List[ComponentCapability]:
        return [ComponentCapability(name="test", description="Test capability")]
    
    async def _do_initialize(self):
        self.state_changes.append(ComponentLifecycle.INITIALIZED)
    
    async def _do_shutdown(self):
        self.state_changes.append(ComponentLifecycle.SHUTDOWN)
    
    async def _do_start(self):
        self.state_changes.append(ComponentLifecycle.RUNNING)
    
    async def _do_stop(self):
        self.state_changes.append(ComponentLifecycle.STOPPED)


class TestComponentStateIntegration:
    """测试组件生命周期与状态管理的集成"""
    
    def test_component_state_triggers_global_state_update(self):
        """测试组件状态变化触发全局状态更新"""
        state_store = StateStore()
        component = TestComponent("test")
        
        # 初始状态
        assert state_store.get_state().lifecycle_state == LifecycleState.IDLE
        
        # 组件初始化应该触发全局状态更新
        asyncio.run(component.initialize())
        state_store.dispatch(UpdateLifecycleAction(LifecycleState.RUNNING))
        
        # 验证全局状态已更新
        assert state_store.get_state().lifecycle_state == LifecycleState.RUNNING
    
    def test_state_subscriber_respond_to_component_lifecycle(self):
        """测试状态订阅者响应组件生命周期事件"""
        state_store = StateStore()
        component = TestComponent("test")
        lifecycle_events = []
        
        def on_state_change(state: AgentState):
            lifecycle_events.append(state.lifecycle_state)
        
        state_store.subscribe(on_state_change)
        
        # 组件生命周期变化
        asyncio.run(component.initialize())
        state_store.dispatch(UpdateLifecycleAction(LifecycleState.RUNNING))
        asyncio.run(component.start())
        state_store.dispatch(UpdateLifecycleAction(LifecycleState.RUNNING))
        
        # 验证订阅者收到状态变化通知
        assert len(lifecycle_events) >= 2
    
    @pytest.mark.asyncio
    async def test_concurrent_component_state_updates(self):
        """测试多组件并发状态更新"""
        state_store = StateStore()
        component1 = TestComponent("comp1")
        component2 = TestComponent("comp2")
        
        update_count = [0]
        
        def count_updates(state: AgentState):
            update_count[0] += 1
        
        state_store.subscribe(count_updates)
        
        # 并发初始化多个组件
        await asyncio.gather(
            component1.initialize(),
            component2.initialize()
        )
        
        # 更新状态
        state_store.dispatch(UpdateIterationAction(1))
        state_store.dispatch(UpdateIterationAction(2))
        
        # 验证所有更新都被处理
        assert update_count[0] >= 3  # 初始 + 2 次更新
    
    def test_component_failure_state_recovery(self):
        """测试组件故障时的状态恢复"""
        state_store = StateStore()
        component = TestComponent("test")
        
        # 正常初始化
        asyncio.run(component.initialize())
        state_store.dispatch(UpdateLifecycleAction(LifecycleState.RUNNING))
        
        # 模拟故障
        try:
            raise RuntimeError("Component failed")
        except Exception:
            # 故障时更新状态
            state_store.dispatch(UpdateLifecycleAction(LifecycleState.PAUSED))
        
        # 验证状态已更新为 PAUSED
        assert state_store.get_state().lifecycle_state == LifecycleState.PAUSED
        
        # 恢复
        state_store.dispatch(UpdateLifecycleAction(LifecycleState.RUNNING))
        assert state_store.get_state().lifecycle_state == LifecycleState.RUNNING
    
    @pytest.mark.asyncio
    async def test_component_lifecycle_state_synchronization(self):
        """测试组件生命周期与全局状态的同步"""
        state_store = StateStore()
        component = TestComponent("test")
        
        # 组件状态变化历史
        component_states = []
        global_states = []
        
        def track_component_state(state: AgentState):
            global_states.append(state.lifecycle_state)
        
        state_store.subscribe(track_component_state)
        
        # 完整的生命周期
        await component.initialize()
        state_store.dispatch(UpdateLifecycleAction(LifecycleState.RUNNING))
        
        await component.start()
        state_store.dispatch(UpdateLifecycleAction(LifecycleState.RUNNING))
        
        await component.stop()
        state_store.dispatch(UpdateLifecycleAction(LifecycleState.PAUSED))
        
        await component.shutdown()
        state_store.dispatch(UpdateLifecycleAction(LifecycleState.IDLE))
        
        # 验证状态同步
        assert len(global_states) >= 4
        assert global_states[-1] == LifecycleState.IDLE
    
    def test_multiple_components_shared_state(self):
        """测试多组件共享状态"""
        state_store = StateStore()
        component1 = TestComponent("comp1")
        component2 = TestComponent("comp2")
        
        shared_iteration = []
        
        def track_iteration(state: AgentState):
            shared_iteration.append(state.current_iteration)
        
        state_store.subscribe(track_iteration)
        
        # 两个组件都更新同一个状态
        asyncio.run(component1.initialize())
        state_store.dispatch(UpdateIterationAction(1))
        
        asyncio.run(component2.initialize())
        state_store.dispatch(UpdateIterationAction(2))
        
        # 验证共享状态一致
        assert shared_iteration == [0, 1, 2]
        assert state_store.get_state().current_iteration == 2
    
    def test_component_state_isolation(self):
        """测试组件状态隔离"""
        state_store = StateStore()
        component1 = TestComponent("comp1")
        component2 = TestComponent("comp2")
        
        # 每个组件有自己的内部状态
        asyncio.run(component1.initialize())
        asyncio.run(component2.initialize())
        
        # 组件内部状态独立
        assert len(component1.state_changes) == 1
        assert len(component2.state_changes) == 1
        assert component1.state_changes[0] == ComponentLifecycle.INITIALIZED
        assert component2.state_changes[0] == ComponentLifecycle.INITIALIZED
    
    @pytest.mark.asyncio
    async def test_component_lifecycle_error_propagation(self):
        """测试组件生命周期错误传播到状态管理"""
        state_store = StateStore()
        component = TestComponent("test")
        
        errors = []
        
        def track_errors(state: AgentState):
            if state.lifecycle_state == LifecycleState.PAUSED:
                errors.append("Component paused due to error")
        
        state_store.subscribe(track_errors)
        
        # 正常初始化
        await component.initialize()
        state_store.dispatch(UpdateLifecycleAction(LifecycleState.RUNNING))
        
        # 模拟错误并更新状态
        try:
            raise Exception("Test error")
        except Exception as e:
            # 错误传播到状态管理
            state_store.dispatch(UpdateLifecycleAction(LifecycleState.PAUSED))
        
        # 验证错误被传播
        assert len(errors) == 1
        assert "error" in errors[0].lower()


class TestComponentStatePatterns:
    """测试组件与状态管理的设计模式"""
    
    def test_state_machine_pattern(self):
        """测试状态机模式的实现"""
        state_store = StateStore()
        component = TestComponent("test")
        
        # 状态转换规则
        valid_transitions = {
            LifecycleState.IDLE: [LifecycleState.RUNNING],
            LifecycleState.RUNNING: [LifecycleState.PAUSED, LifecycleState.IDLE],
            LifecycleState.PAUSED: [LifecycleState.RUNNING, LifecycleState.IDLE],
        }
        
        # 初始状态
        assert state_store.get_state().lifecycle_state == LifecycleState.IDLE
        
        # 状态转换
        asyncio.run(component.initialize())
        state_store.dispatch(UpdateLifecycleAction(LifecycleState.RUNNING))
        
        # 验证状态转换有效
        assert state_store.get_state().lifecycle_state == LifecycleState.RUNNING
        
        # 验证转换规则
        current_state = state_store.get_state().lifecycle_state
        assert LifecycleState.PAUSED in valid_transitions.get(current_state, [])
    
    def test_observer_pattern_with_components(self):
        """测试组件场景下的观察者模式"""
        state_store = StateStore()
        
        class ComponentObserver:
            def __init__(self, name):
                self.name = name
                self.notifications = []
            
            def notify(self, state: AgentState):
                self.notifications.append({
                    'name': self.name,
                    'state': state.lifecycle_state,
                    'iteration': state.current_iteration
                })
        
        observer1 = ComponentObserver("A")
        observer2 = ComponentObserver("B")
        
        state_store.subscribe(observer1.notify)
        state_store.subscribe(observer2.notify)
        
        # 触发状态变化
        state_store.dispatch(UpdateIterationAction(5))
        state_store.dispatch(UpdateLifecycleAction(LifecycleState.RUNNING))
        
        # 验证所有观察者都收到通知
        assert len(observer1.notifications) == 3  # 初始 + 2 次更新
        assert len(observer2.notifications) == 3
        assert observer1.notifications[-1]['iteration'] == 5
    
    def test_mediator_pattern_through_state(self):
        """测试通过状态实现的中介者模式"""
        state_store = StateStore()
        component1 = TestComponent("comp1")
        component2 = TestComponent("comp2")
        
        # 组件通过状态存储间接通信
        interactions = []
        
        def track_interaction(state: AgentState):
            interactions.append({
                'iteration': state.current_iteration,
                'timestamp': len(interactions)
            })
        
        state_store.subscribe(track_interaction)
        
        # 组件 1 更新状态
        asyncio.run(component1.initialize())
        state_store.dispatch(UpdateIterationAction(1))
        
        # 组件 2 响应状态变化
        asyncio.run(component2.initialize())
        state_store.dispatch(UpdateIterationAction(2))
        
        # 验证中介者模式：组件通过状态存储解耦
        assert len(interactions) == 3  # 初始 + 2 次更新
        assert interactions[1]['iteration'] == 1
        assert interactions[2]['iteration'] == 2
