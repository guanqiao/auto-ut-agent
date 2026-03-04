"""测试统一状态管理"""
import pytest
from pyutagent.core.state_store import (
    StateStore,
    AgentState,
    Action,
    UpdateIterationAction,
    UpdateCoverageAction,
    UpdateLifecycleAction,
    LifecycleState
)


class TestAgentState:
    """测试 Agent 状态"""
    
    def test_create_default_state(self):
        """测试创建默认状态"""
        state = AgentState()
        
        assert state.lifecycle_state == LifecycleState.IDLE
        assert state.current_phase == "IDLE"
        assert state.current_iteration == 0
        assert state.target_coverage == 0.8
        assert state.current_coverage == 0.0
    
    def test_create_custom_state(self):
        """测试创建自定义状态"""
        state = AgentState(
            lifecycle_state=LifecycleState.RUNNING,
            current_phase="GENERATING",
            current_iteration=5,
            target_coverage=0.85,
            current_coverage=0.6
        )
        
        assert state.lifecycle_state == LifecycleState.RUNNING
        assert state.current_phase == "GENERATING"
        assert state.current_iteration == 5
        assert state.target_coverage == 0.85
        assert state.current_coverage == 0.6


class TestStateStore:
    """测试状态存储"""
    
    def test_create_state_store(self):
        """测试创建状态存储"""
        store = StateStore()
        assert store is not None
    
    def test_get_initial_state(self):
        """测试获取初始状态"""
        store = StateStore()
        state = store.get_state()
        
        assert state.lifecycle_state == LifecycleState.IDLE
        assert state.current_iteration == 0
    
    def test_dispatch_update_iteration(self):
        """测试分发更新迭代次数的 action"""
        store = StateStore()
        
        # 分发 action
        action = UpdateIterationAction(iteration=5)
        store.dispatch(action)
        
        # 验证状态更新
        state = store.get_state()
        assert state.current_iteration == 5
    
    def test_dispatch_update_coverage(self):
        """测试分发更新覆盖率的 action"""
        store = StateStore()
        
        # 分发 action
        action = UpdateCoverageAction(coverage=0.75)
        store.dispatch(action)
        
        # 验证状态更新
        state = store.get_state()
        assert state.current_coverage == 0.75
    
    def test_dispatch_update_lifecycle(self):
        """测试分发更新生命周期的 action"""
        store = StateStore()
        
        # 分发 action
        action = UpdateLifecycleAction(state=LifecycleState.RUNNING)
        store.dispatch(action)
        
        # 验证状态更新
        state = store.get_state()
        assert state.lifecycle_state == LifecycleState.RUNNING
    
    def test_state_subscriber(self):
        """测试状态订阅者"""
        store = StateStore()
        received_states = []
        
        def subscriber(state: AgentState):
            received_states.append(state)
        
        store.subscribe(subscriber)
        
        # 分发 action
        action = UpdateIterationAction(iteration=3)
        store.dispatch(action)
        
        # 验证订阅者收到通知（初始状态 + 更新后状态）
        assert len(received_states) >= 1
        assert received_states[-1].current_iteration == 3
    
    def test_state_history(self):
        """测试状态历史"""
        store = StateStore()
        
        # 分发多个 action
        store.dispatch(UpdateIterationAction(iteration=1))
        store.dispatch(UpdateIterationAction(iteration=2))
        store.dispatch(UpdateIterationAction(iteration=3))
        
        # 验证状态历史
        state = store.get_state()
        assert len(state.state_history) == 3
    
    def test_multiple_subscribers(self):
        """测试多个订阅者"""
        store = StateStore()
        received_by_sub1 = []
        received_by_sub2 = []
        
        def subscriber1(state: AgentState):
            received_by_sub1.append(state)
        
        def subscriber2(state: AgentState):
            received_by_sub2.append(state)
        
        store.subscribe(subscriber1)
        store.subscribe(subscriber2)
        
        # 分发 action
        action = UpdateCoverageAction(coverage=0.5)
        store.dispatch(action)
        
        # 验证两个订阅者都收到通知
        assert len(received_by_sub1) >= 1
        assert len(received_by_sub2) >= 1
        assert received_by_sub1[-1].current_coverage == 0.5
        assert received_by_sub2[-1].current_coverage == 0.5


class TestActions:
    """测试 Action"""
    
    def test_update_iteration_action(self):
        """测试更新迭代次数 action"""
        state = AgentState(current_iteration=0)
        action = UpdateIterationAction(iteration=5)
        
        new_state = action.reduce(state)
        
        assert new_state.current_iteration == 5
    
    def test_update_coverage_action(self):
        """测试更新覆盖率 action"""
        state = AgentState(current_coverage=0.0)
        action = UpdateCoverageAction(coverage=0.8)
        
        new_state = action.reduce(state)
        
        assert new_state.current_coverage == 0.8
    
    def test_update_lifecycle_action(self):
        """测试更新生命周期 action"""
        state = AgentState(lifecycle_state=LifecycleState.IDLE)
        action = UpdateLifecycleAction(state=LifecycleState.RUNNING)
        
        new_state = action.reduce(state)
        
        assert new_state.lifecycle_state == LifecycleState.RUNNING
