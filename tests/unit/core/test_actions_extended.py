"""Action 系统扩展测试"""
import pytest
from typing import List, Callable
from pyutagent.core.actions import (
    BatchAction,
    TransactionalAction,
    ConditionalAction,
    ActionSequence,
    ActionWithRollback
)
from pyutagent.core.state_store import AgentState, LifecycleState, Action


class TestBatchAction:
    """批量动作测试"""
    
    def test_create_batch_action(self):
        """测试创建批量动作"""
        batch = BatchAction()
        assert batch is not None
    
    def test_batch_action_add_actions(self):
        """测试批量动作添加动作"""
        batch = BatchAction()
        
        from pyutagent.core.state_store import UpdateIterationAction
        batch.add_action(UpdateIterationAction(5))
        batch.add_action(UpdateIterationAction(10))
        
        assert len(batch.actions) == 2
    
    def test_batch_action_execute(self):
        """测试批量动作执行"""
        from pyutagent.core.state_store import UpdateIterationAction, UpdateCoverageAction
        
        batch = BatchAction()
        batch.add_action(UpdateIterationAction(5))
        batch.add_action(UpdateCoverageAction(0.75))
        
        state = AgentState(current_iteration=0, current_coverage=0.0)
        new_state = batch.reduce(state)
        
        assert new_state.current_iteration == 5
        assert new_state.current_coverage == 0.75
    
    def test_batch_action_empty(self):
        """测试空批量动作"""
        batch = BatchAction()
        state = AgentState(current_iteration=0)
        
        new_state = batch.reduce(state)
        
        # 空批量动作应该返回原状态
        assert new_state.current_iteration == 0
    
    def test_batch_action_order_matters(self):
        """测试批量动作顺序"""
        from pyutagent.core.state_store import UpdateIterationAction
        
        batch1 = BatchAction()
        batch1.add_action(UpdateIterationAction(5))
        batch1.add_action(UpdateIterationAction(10))
        
        batch2 = BatchAction()
        batch2.add_action(UpdateIterationAction(10))
        batch2.add_action(UpdateIterationAction(5))
        
        state = AgentState(current_iteration=0)
        
        state1 = batch1.reduce(state)
        state2 = batch2.reduce(state)
        
        # 最后一个动作生效
        assert state1.current_iteration == 10
        assert state2.current_iteration == 5


class TestTransactionalAction:
    """事务性动作测试"""
    
    def test_create_transactional_action(self):
        """测试创建事务性动作"""
        transactional = TransactionalAction()
        assert transactional is not None
    
    def test_transactional_action_success(self):
        """测试事务性动作成功"""
        from pyutagent.core.state_store import UpdateIterationAction, UpdateCoverageAction
        
        transactional = TransactionalAction()
        transactional.add_action(UpdateIterationAction(5))
        transactional.add_action(UpdateCoverageAction(0.75))
        
        state = AgentState(current_iteration=0, current_coverage=0.0)
        new_state = transactional.reduce(state)
        
        assert new_state.current_iteration == 5
        assert new_state.current_coverage == 0.75
    
    def test_transactional_action_rollback_on_failure(self):
        """测试事务性动作失败时回滚"""
        from pyutagent.core.state_store import UpdateIterationAction
        
        class FailingAction(Action):
            def reduce(self, state: AgentState) -> AgentState:
                raise ValueError("Simulated failure")
        
        transactional = TransactionalAction()
        transactional.add_action(UpdateIterationAction(5))
        transactional.add_action(FailingAction())
        transactional.add_action(UpdateIterationAction(10))
        
        # 创建新状态用于测试
        state = AgentState(current_iteration=0)
        
        # 执行应该失败并回滚
        with pytest.raises(ValueError):
            transactional.reduce(state)
        
        # 由于 Python 参数传递是引用，状态应该已被回滚
        # 检查状态是否被恢复到初始值
        assert state.current_iteration == 0  # 应该回滚到初始值
    
    def test_transactional_action_with_rollback_actions(self):
        """测试带 rollback 动作的事务"""
        from pyutagent.core.state_store import UpdateIterationAction
        
        transactional = TransactionalAction()
        transactional.add_action(UpdateIterationAction(5))
        
        # 添加 rollback 动作
        rollback_action = UpdateIterationAction(0)
        transactional.add_rollback_action(rollback_action)
        
        state = AgentState(current_iteration=10)
        new_state = transactional.reduce(state)
        
        assert new_state.current_iteration == 5
    
    def test_transactional_action_preserves_history(self):
        """测试事务性动作保留历史"""
        from pyutagent.core.state_store import UpdateIterationAction
        
        transactional = TransactionalAction()
        transactional.add_action(UpdateIterationAction(5))
        
        state = AgentState(current_iteration=0)
        new_state = transactional.reduce(state)
        
        # 状态历史应该被保留
        assert new_state.state_history is not None


class TestConditionalAction:
    """条件动作测试"""
    
    def test_create_conditional_action(self):
        """测试创建条件动作"""
        condition = lambda s: True
        action = ConditionalAction(condition, None)
        assert action is not None
    
    def test_conditional_action_when_true(self):
        """测试条件为真时执行"""
        from pyutagent.core.state_store import UpdateIterationAction
        
        condition = lambda s: s.current_iteration < 10
        update_action = UpdateIterationAction(15)
        
        conditional = ConditionalAction(condition, update_action)
        
        state = AgentState(current_iteration=5)
        new_state = conditional.reduce(state)
        
        assert new_state.current_iteration == 15
    
    def test_conditional_action_when_false(self):
        """测试条件为假时不执行"""
        from pyutagent.core.state_store import UpdateIterationAction
        
        condition = lambda s: s.current_iteration < 10
        update_action = UpdateIterationAction(15)
        
        conditional = ConditionalAction(condition, update_action)
        
        state = AgentState(current_iteration=20)
        new_state = conditional.reduce(state)
        
        # 条件不满足，状态不变
        assert new_state.current_iteration == 20
    
    def test_conditional_action_with_else_action(self):
        """测试带 else 动作的条件动作"""
        from pyutagent.core.state_store import UpdateIterationAction
        
        condition = lambda s: s.current_iteration < 10
        then_action = UpdateIterationAction(15)
        else_action = UpdateIterationAction(25)
        
        conditional = ConditionalAction(condition, then_action, else_action)
        
        # 条件为假
        state = AgentState(current_iteration=20)
        new_state = conditional.reduce(state)
        
        assert new_state.current_iteration == 25
    
    def test_conditional_action_complex_condition(self):
        """测试复杂条件"""
        from pyutagent.core.state_store import UpdateCoverageAction
        
        condition = lambda s: s.current_coverage < 0.5 and s.lifecycle_state == LifecycleState.RUNNING
        update_action = UpdateCoverageAction(0.8)
        
        conditional = ConditionalAction(condition, update_action)
        
        # 条件满足
        state1 = AgentState(current_coverage=0.3, lifecycle_state=LifecycleState.RUNNING)
        new_state1 = conditional.reduce(state1)
        assert new_state1.current_coverage == 0.8
        
        # 条件不满足（覆盖率已达标）
        state2 = AgentState(current_coverage=0.7, lifecycle_state=LifecycleState.RUNNING)
        new_state2 = conditional.reduce(state2)
        assert new_state2.current_coverage == 0.7
        
        # 条件不满足（状态不对）
        state3 = AgentState(current_coverage=0.3, lifecycle_state=LifecycleState.IDLE)
        new_state3 = conditional.reduce(state3)
        assert new_state3.current_coverage == 0.3


class TestActionSequence:
    """动作序列测试"""
    
    def test_create_action_sequence(self):
        """测试创建动作序列"""
        sequence = ActionSequence()
        assert sequence is not None
    
    def test_action_sequence_chain(self):
        """测试动作序列链"""
        from pyutagent.core.state_store import UpdateIterationAction, UpdateCoverageAction
        
        sequence = ActionSequence()
        sequence.add(UpdateIterationAction(5))
        sequence.add(UpdateCoverageAction(0.75))
        
        state = AgentState(current_iteration=0, current_coverage=0.0)
        new_state = sequence.execute(state)
        
        assert new_state.current_iteration == 5
        assert new_state.current_coverage == 0.75
    
    def test_action_sequence_with_conditions(self):
        """测试带条件的动作序列"""
        from pyutagent.core.state_store import UpdateIterationAction
        
        sequence = ActionSequence()
        sequence.add(UpdateIterationAction(5))
        
        # 条件执行
        condition = lambda s: s.current_iteration >= 5
        sequence.add_conditional(
            condition,
            UpdateIterationAction(10)
        )
        
        state = AgentState(current_iteration=0)
        new_state = sequence.execute(state)
        
        # 第一个动作执行后 iteration=5，满足条件，第二个动作也执行
        assert new_state.current_iteration == 10
    
    def test_action_sequence_stop_on_failure(self):
        """测试动作序列失败时停止"""
        from pyutagent.core.state_store import UpdateIterationAction
        
        class FailingAction(Action):
            def reduce(self, state: AgentState) -> AgentState:
                raise ValueError("Action failed")
        
        sequence = ActionSequence(stop_on_failure=True)
        sequence.add(UpdateIterationAction(5))
        sequence.add(FailingAction())
        sequence.add(UpdateIterationAction(10))
        
        state = AgentState(current_iteration=0)
        
        # 应该在失败时停止
        with pytest.raises(ValueError):
            sequence.execute(state)
        
        # 状态应该是失败前的状态
        assert state.current_iteration == 0


class TestActionWithRollback:
    """可回滚动作测试"""
    
    def test_create_action_with_rollback(self):
        """测试创建可回滚动作"""
        from pyutagent.core.state_store import UpdateIterationAction
        
        action = UpdateIterationAction(5)
        rollback = UpdateIterationAction(0)
        
        rollback_action = ActionWithRollback(action, rollback)
        assert rollback_action is not None
    
    def test_action_with_rollback_execute(self):
        """测试可回滚动作执行"""
        from pyutagent.core.state_store import UpdateIterationAction
        
        action = UpdateIterationAction(5)
        rollback = UpdateIterationAction(0)
        
        rollback_action = ActionWithRollback(action, rollback)
        
        state = AgentState(current_iteration=10)
        new_state = rollback_action.reduce(state)
        
        assert new_state.current_iteration == 5
    
    def test_action_with_rollback_execute_rollback(self):
        """测试可回滚动作回滚"""
        from pyutagent.core.state_store import UpdateIterationAction
        
        action = UpdateIterationAction(5)
        # rollback action 在这里实际上不会被使用，因为 rollback 方法使用保存的 before_state
        rollback = UpdateIterationAction(0)
        
        rollback_action = ActionWithRollback(action, rollback)
        
        state = AgentState(current_iteration=10)
        
        # 执行
        new_state = rollback_action.reduce(state)
        assert new_state.current_iteration == 5
        
        # 回滚 - 应该恢复到执行前的状态 (iteration=10)
        original_state = rollback_action.rollback(new_state)
        assert original_state.current_iteration == 10  # 改为 10，恢复到执行前的状态
    
    def test_action_with_rollback_complex(self):
        """测试复杂可回滚动作"""
        from pyutagent.core.state_store import UpdateCoverageAction
        
        action = UpdateCoverageAction(0.8)
        rollback = UpdateCoverageAction(0.0)
        
        rollback_action = ActionWithRollback(action, rollback)
        
        state = AgentState(current_coverage=0.5)
        new_state = rollback_action.reduce(state)
        
        assert new_state.current_coverage == 0.8
        
        # 回滚 - 应该恢复到执行前的状态 (coverage=0.5)
        original_state = rollback_action.rollback(new_state)
        assert original_state.current_coverage == 0.5  # 改为 0.5，恢复到执行前的状态
