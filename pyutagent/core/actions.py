"""Action 系统扩展 - 批量动作、事务性动作、条件动作"""
import copy
from typing import Any, Callable, List, Optional
from abc import ABC, abstractmethod
import logging

from pyutagent.core.state_store import Action, AgentState

logger = logging.getLogger(__name__)


class BatchAction(Action):
    """批量动作 - 一次性执行多个动作"""
    
    def __init__(self):
        self.actions: List[Action] = []
    
    def add_action(self, action: Action) -> 'BatchAction':
        """添加动作"""
        self.actions.append(action)
        return self
    
    def reduce(self, state: AgentState) -> AgentState:
        """执行所有动作"""
        if not self.actions:
            return state
        
        current_state = state
        
        for action in self.actions:
            try:
                current_state = action.reduce(current_state)
                logger.debug(f"Batch action executed: {action.__class__.__name__}")
            except Exception as e:
                logger.error(f"Batch action failed: {e}")
                raise
        
        return current_state


class TransactionalAction(Action):
    """事务性动作 - 支持回滚的动作"""
    
    def __init__(self):
        self.actions: List[Action] = []
        self.rollback_actions: List[Action] = []
    
    def add_action(self, action: Action) -> 'TransactionalAction':
        """添加动作"""
        self.actions.append(action)
        return self
    
    def add_rollback_action(self, action: Action) -> 'TransactionalAction':
        """添加回滚动作"""
        self.rollback_actions.append(action)
        return self
    
    def reduce(self, state: AgentState) -> AgentState:
        """执行事务"""
        if not self.actions:
            return state
        
        # 保存原始状态的深拷贝
        original_state = self._deep_copy_state(state)
        
        try:
            current_state = state
            
            for action in self.actions:
                current_state = action.reduce(current_state)
                logger.debug(f"Transactional action executed: {action.__class__.__name__}")
            
            return current_state
            
        except Exception as e:
            logger.error(f"Transaction failed, rolling back: {e}")
            # 回滚到原始状态
            self._restore_state(state, original_state)
            raise
    
    def _deep_copy_state(self, state: AgentState) -> AgentState:
        """深拷贝状态"""
        return AgentState(
            lifecycle_state=state.lifecycle_state,
            current_phase=state.current_phase,
            current_iteration=state.current_iteration,
            target_coverage=state.target_coverage,
            current_coverage=state.current_coverage,
            working_memory=copy.deepcopy(state.working_memory),
            error_state=copy.deepcopy(state.error_state),
            metrics=copy.deepcopy(state.metrics),
            state_history=copy.deepcopy(state.state_history)
        )
    
    def _restore_state(self, target: AgentState, source: AgentState):
        """恢复状态"""
        target.lifecycle_state = source.lifecycle_state
        target.current_phase = source.current_phase
        target.current_iteration = source.current_iteration
        target.target_coverage = source.target_coverage
        target.current_coverage = source.current_coverage
        target.working_memory = source.working_memory.copy()
        target.error_state = source.error_state.copy()
        target.metrics = source.metrics.copy()
        target.state_history = source.state_history.copy()


class ConditionalAction(Action):
    """条件动作 - 基于条件执行动作"""
    
    def __init__(
        self,
        condition: Callable[[AgentState], bool],
        action: Optional[Action],
        else_action: Optional[Action] = None
    ):
        self.condition = condition
        self.action = action
        self.else_action = else_action
    
    def reduce(self, state: AgentState) -> AgentState:
        """根据条件执行动作"""
        try:
            should_execute = self.condition(state)
        except Exception as e:
            logger.error(f"Condition evaluation failed: {e}")
            should_execute = False
        
        if should_execute:
            if self.action:
                logger.debug("Condition met, executing action")
                return self.action.reduce(state)
            else:
                logger.debug("Condition met, but no action defined")
                return state
        else:
            if self.else_action:
                logger.debug("Condition not met, executing else action")
                return self.else_action.reduce(state)
            else:
                logger.debug("Condition not met, returning original state")
                return state


class ActionSequence:
    """动作序列 - 按顺序执行动作"""
    
    def __init__(self, stop_on_failure: bool = False):
        self.actions: List[Action] = []
        self.stop_on_failure = stop_on_failure
    
    def add(self, action: Action) -> 'ActionSequence':
        """添加动作"""
        self.actions.append(action)
        return self
    
    def add_conditional(
        self,
        condition: Callable[[AgentState], bool],
        action: Action,
        else_action: Optional[Action] = None
    ) -> 'ActionSequence':
        """添加条件动作"""
        conditional = ConditionalAction(condition, action, else_action)
        self.actions.append(conditional)
        return self
    
    def execute(self, state: AgentState) -> AgentState:
        """执行序列"""
        if not self.actions:
            return state
        
        current_state = state
        
        for action in self.actions:
            try:
                current_state = action.reduce(current_state)
                logger.debug(f"Sequence action executed: {action.__class__.__name__}")
            except Exception as e:
                logger.error(f"Sequence action failed: {e}")
                if self.stop_on_failure:
                    raise
                # 如果不停止，继续执行下一个动作
        
        return current_state


class ActionWithRollback(Action):
    """可回滚动作 - 执行后可以回滚"""
    
    def __init__(self, action: Action, rollback_action: Action):
        self.action = action
        self.rollback_action = rollback_action
        self._executed = False
        self._before_state: Optional[AgentState] = None
    
    def reduce(self, state: AgentState) -> AgentState:
        """执行动作"""
        # 保存执行前的状态
        self._before_state = self._copy_state(state)
        
        # 执行主动作
        result = self.action.reduce(state)
        self._executed = True
        
        logger.debug(f"ActionWithRollback executed: {self.action.__class__.__name__}")
        return result
    
    def rollback(self, state: AgentState) -> AgentState:
        """回滚到执行前的状态"""
        if not self._executed:
            logger.warning("Action not executed yet, nothing to rollback")
            return state
        
        if self._before_state is None:
            logger.warning("No before state saved, cannot rollback")
            return state
        
        # 恢复状态
        result = self._restore_state(state, self._before_state)
        
        logger.debug(f"ActionWithRollback rolled back: {self.rollback_action.__class__.__name__}")
        return result
    
    def _copy_state(self, state: AgentState) -> AgentState:
        """拷贝状态"""
        return AgentState(
            lifecycle_state=state.lifecycle_state,
            current_phase=state.current_phase,
            current_iteration=state.current_iteration,
            target_coverage=state.target_coverage,
            current_coverage=state.current_coverage,
            working_memory=copy.deepcopy(state.working_memory),
            error_state=copy.deepcopy(state.error_state),
            metrics=copy.deepcopy(state.metrics),
            state_history=copy.deepcopy(state.state_history)
        )
    
    def _restore_state(
        self,
        target: AgentState,
        source: AgentState
    ) -> AgentState:
        """恢复状态"""
        target.lifecycle_state = source.lifecycle_state
        target.current_phase = source.current_phase
        target.current_iteration = source.current_iteration
        target.target_coverage = source.target_coverage
        target.current_coverage = source.current_coverage
        target.working_memory = source.working_memory.copy()
        target.error_state = source.error_state.copy()
        target.metrics = source.metrics.copy()
        target.state_history = source.state_history.copy()
        return target
