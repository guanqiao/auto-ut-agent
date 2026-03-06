"""集成测试：事件总线与状态存储的协同工作"""
import pytest
import asyncio
from pyutagent.core.event_bus import EventBus
from pyutagent.core.state_store import (
    StateStore,
    AgentState,
    UpdateIterationAction,
    UpdateCoverageAction,
    UpdateLifecycleAction,
    LifecycleState
)


class TestEventBusStateStoreIntegration:
    """测试事件总线与状态存储的集成"""
    
    def test_state_change_via_event_bus(self):
        """测试通过事件总线触发状态变化"""
        event_bus = EventBus()
        state_store = StateStore()
        received_states = []
        
        # 定义事件处理器：当状态变化事件发布时，更新状态存储
        def on_state_update(event: dict):
            if event.get('type') == 'update_iteration':
                action = UpdateIterationAction(event['iteration'])
                state_store.dispatch(action)
            elif event.get('type') == 'update_coverage':
                action = UpdateCoverageAction(event['coverage'])
                state_store.dispatch(action)
        
        # 订阅状态更新事件
        event_bus.subscribe(dict, on_state_update)
        
        # 发布状态更新事件
        event_bus.publish({'type': 'update_iteration', 'iteration': 5})
        event_bus.publish({'type': 'update_coverage', 'coverage': 0.75})
        
        # 验证状态已更新
        state = state_store.get_state()
        assert state.current_iteration == 5
        assert state.current_coverage == 0.75
    
    def test_state_store_publishes_to_event_bus(self):
        """测试状态存储通过事件总线发布状态变化"""
        event_bus = EventBus()
        state_store = StateStore()
        received_events = []
        
        # 订阅状态变化事件
        def on_state_change(state: AgentState):
            received_events.append(state)
        
        state_store.subscribe(on_state_change)
        
        # 更新状态
        state_store.dispatch(UpdateIterationAction(3))
        state_store.dispatch(UpdateIterationAction(6))
        
        # 验证订阅者收到通知（初始 + 2 次更新）
        assert len(received_events) >= 2
        assert received_events[-1].current_iteration == 6
    
    def test_multiple_components_subscribe_same_state(self):
        """测试多个组件订阅同一状态变化"""
        state_store = StateStore()
        component1_states = []
        component2_states = []
        
        def component1_listener(state: AgentState):
            component1_states.append(state.current_iteration)
        
        def component2_listener(state: AgentState):
            component2_states.append(state.current_iteration)
        
        state_store.subscribe(component1_listener)
        state_store.subscribe(component2_listener)
        
        # 更新状态
        state_store.dispatch(UpdateIterationAction(1))
        state_store.dispatch(UpdateIterationAction(2))
        state_store.dispatch(UpdateIterationAction(3))
        
        # 验证两个组件都收到所有状态变化
        assert component1_states == [0, 1, 2, 3]  # 包含初始状态
        assert component2_states == [0, 1, 2, 3]
    
    @pytest.mark.asyncio
    async def test_async_event_bus_with_state_store(self):
        """测试异步事件总线与状态存储的时序"""
        event_bus = AsyncEventBus()
        state_store = StateStore()
        execution_order = []
        
        async def slow_handler(event: dict):
            await asyncio.sleep(0.01)
            if event.get('type') == 'update_iteration':
                action = UpdateIterationAction(event['iteration'])
                state_store.dispatch(action)
            execution_order.append(f"handler_{event.get('iteration', 'unknown')}")
        
        event_bus.subscribe(dict, slow_handler)
        
        # 并发发布多个事件
        tasks = [
            event_bus.publish({'type': 'update_iteration', 'iteration': i})
            for i in range(1, 4)
        ]
        await asyncio.gather(*tasks)
        
        # 验证所有事件都被处理
        assert len(execution_order) == 3
        # 验证最终状态
        state = state_store.get_state()
        # 由于并发执行，最终状态可能是任意一个值（1, 2, 或 3）
        assert state.current_iteration in [1, 2, 3]
    
    def test_state_update_error_handling(self):
        """测试状态更新失败时的事件通知"""
        event_bus = EventBus()
        state_store = StateStore()
        error_events = []
        
        def error_handler(event: dict):
            if event.get('type') == 'error':
                error_events.append(event)
        
        event_bus.subscribe(dict, error_handler)
        
        # 模拟一个会导致错误的场景
        try:
            # 这里我们手动触发一个错误
            raise ValueError("State update failed")
        except Exception as e:
            event_bus.publish({
                'type': 'error',
                'error': str(e),
                'context': 'state_update'
            })
        
        # 验证错误事件被发布
        assert len(error_events) == 1
        assert error_events[0]['error'] == "State update failed"
    
    def test_state_history_with_events(self):
        """测试状态历史与事件记录的对应关系"""
        state_store = StateStore()
        
        # 执行多次状态更新
        state_store.dispatch(UpdateIterationAction(1))
        state_store.dispatch(UpdateCoverageAction(0.5))
        state_store.dispatch(UpdateIterationAction(2))
        state_store.dispatch(UpdateLifecycleAction(LifecycleState.RUNNING))
        
        # 验证状态历史
        state = state_store.get_state()
        assert len(state.state_history) == 4
        
        # 验证历史记录包含正确的 action 信息
        assert state.state_history[0]['action'] == 'UpdateIterationAction'
        assert state.state_history[1]['action'] == 'UpdateCoverageAction'
        assert state.state_history[2]['action'] == 'UpdateIterationAction'
        assert state.state_history[3]['action'] == 'UpdateLifecycleAction'
    
    def test_concurrent_state_updates(self):
        """测试并发状态更新的正确性"""
        state_store = StateStore()
        update_count = [0]  # 使用列表以便在闭包中修改
        
        def count_updates(state: AgentState):
            update_count[0] += 1
        
        state_store.subscribe(count_updates)
        
        # 模拟多次并发更新（实际场景中可能来自不同组件）
        for i in range(10):
            state_store.dispatch(UpdateIterationAction(i))
        
        # 验证所有更新都被处理
        assert update_count[0] == 11  # 初始订阅通知 + 10 次更新
        assert state_store.get_state().current_iteration == 9
    
    def test_state_subscription_lifecycle(self):
        """测试状态订阅的生命周期管理"""
        state_store = StateStore()
        received = []
        
        def listener(state: AgentState):
            received.append(state.current_iteration)
        
        # 订阅
        state_store.subscribe(listener)
        
        # 产生一些状态变化
        state_store.dispatch(UpdateIterationAction(1))
        state_store.dispatch(UpdateIterationAction(2))
        
        # 取消订阅
        state_store.unsubscribe(listener)
        
        # 再次产生状态变化
        state_store.dispatch(UpdateIterationAction(3))
        
        # 验证取消订阅后不再收到通知
        assert received == [0, 1, 2]  # 不包含 3


class TestEventBusStateStorePatterns:
    """测试事件总线与状态存储的设计模式"""
    
    def test_observer_pattern(self):
        """测试观察者模式的实现"""
        # StateStore 作为被观察者（Subject）
        # 监听器作为观察者（Observer）
        state_store = StateStore()
        observers = []
        
        class Observer:
            def __init__(self, name):
                self.name = name
                self.states = []
            
            def update(self, state: AgentState):
                self.states.append(state)
                observers.append((self.name, state.current_iteration))
        
        observer1 = Observer("A")
        observer2 = Observer("B")
        
        state_store.subscribe(observer1.update)
        state_store.subscribe(observer2.update)
        
        state_store.dispatch(UpdateIterationAction(5))
        
        # 验证所有观察者都收到通知
        assert len(observer1.states) == 2  # 初始 + 更新
        assert len(observer2.states) == 2
        assert observer1.states[-1].current_iteration == 5
        assert observer2.states[-1].current_iteration == 5
    
    def test_publish_subscribe_decoupling(self):
        """测试发布/订阅模式的解耦效果"""
        event_bus = EventBus()
        state_store = StateStore()
        
        # 生产者：只负责发布事件，不关心谁处理
        def producer():
            event_bus.publish({'type': 'iteration', 'value': 10})
        
        # 消费者：只关心事件，不关心谁发布
        consumers_called = []
        def consumer1(event: dict):
            if event.get('type') == 'iteration':
                state_store.dispatch(UpdateIterationAction(event['value']))
                consumers_called.append('consumer1')
        
        def consumer2(event: dict):
            if event.get('type') == 'iteration':
                consumers_called.append('consumer2')
        
        event_bus.subscribe(dict, consumer1)
        event_bus.subscribe(dict, consumer2)
        
        # 生产者发布事件
        producer()
        
        # 验证：生产者和消费者完全解耦
        assert state_store.get_state().current_iteration == 10
        assert 'consumer1' in consumers_called
        assert 'consumer2' in consumers_called
