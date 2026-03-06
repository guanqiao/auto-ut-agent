"""统一状态管理"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Callable, List, Dict, Any
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class LifecycleState(Enum):
    """生命周期状态"""
    IDLE = "idle"
    RUNNING = "running"
    PAUSED = "paused"
    TERMINATED = "terminated"


@dataclass
class AgentStateData:
    """Agent 状态数据
    
    注意：此类名已从 AgentState 重命名为 AgentStateData，
    以避免与 protocols.py 中的 AgentState 枚举冲突。
    """
    lifecycle_state: LifecycleState = LifecycleState.IDLE
    current_phase: str = "IDLE"
    current_iteration: int = 0
    target_coverage: float = 0.8
    current_coverage: float = 0.0
    working_memory: Dict[str, Any] = field(default_factory=dict)
    error_state: Dict[str, Any] = field(default_factory=dict)
    metrics: Dict[str, Any] = field(default_factory=dict)
    state_history: List[Dict[str, Any]] = field(default_factory=list)


# 向后兼容别名
AgentState = AgentStateData


class Action(ABC):
    """Action 基类"""
    
    @abstractmethod
    def reduce(self, state: AgentState) -> AgentState:
        """将 action 应用到状态上"""
        pass


class UpdateIterationAction(Action):
    """更新迭代次数"""
    
    def __init__(self, iteration: int):
        self.iteration = iteration
    
    def reduce(self, state: AgentState) -> AgentState:
        return AgentState(
            lifecycle_state=state.lifecycle_state,
            current_phase=state.current_phase,
            current_iteration=self.iteration,
            target_coverage=state.target_coverage,
            current_coverage=state.current_coverage,
            working_memory=state.working_memory.copy(),
            error_state=state.error_state.copy(),
            metrics=state.metrics.copy(),
            state_history=state.state_history.copy()
        )


class UpdateCoverageAction(Action):
    """更新覆盖率"""
    
    def __init__(self, coverage: float):
        self.coverage = coverage
    
    def reduce(self, state: AgentState) -> AgentState:
        return AgentState(
            lifecycle_state=state.lifecycle_state,
            current_phase=state.current_phase,
            current_iteration=state.current_iteration,
            target_coverage=state.target_coverage,
            current_coverage=self.coverage,
            working_memory=state.working_memory.copy(),
            error_state=state.error_state.copy(),
            metrics=state.metrics.copy(),
            state_history=state.state_history.copy()
        )


class UpdateLifecycleAction(Action):
    """更新生命周期状态"""
    
    def __init__(self, state: LifecycleState):
        self.lifecycle_state = state
    
    def reduce(self, state: AgentState) -> AgentState:
        return AgentState(
            lifecycle_state=self.lifecycle_state,
            current_phase=state.current_phase,
            current_iteration=state.current_iteration,
            target_coverage=state.target_coverage,
            current_coverage=state.current_coverage,
            working_memory=state.working_memory.copy(),
            error_state=state.error_state.copy(),
            metrics=state.metrics.copy(),
            state_history=state.state_history.copy()
        )


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
        
        # 应用 action
        self._state = action.reduce(self._state)
        
        # 记录状态历史
        self._state.state_history.append({
            'from': old_state,
            'to': self._state,
            'action': action.__class__.__name__
        })
        
        logger.debug(f"State changed: {old_state.lifecycle_state} -> {self._state.lifecycle_state}")
        
        # 通知订阅者
        self._notify_listeners()
    
    def subscribe(self, listener: Callable[[AgentState], None]):
        """订阅状态变化"""
        self._listeners.append(listener)
        # 立即通知当前状态
        listener(self._state)
    
    def unsubscribe(self, listener: Callable[[AgentState], None]):
        """取消订阅"""
        if listener in self._listeners:
            self._listeners.remove(listener)
    
    def _notify_listeners(self):
        """通知所有订阅者"""
        for listener in self._listeners:
            try:
                listener(self._state)
            except Exception as e:
                logger.error(f"State listener error: {e}")
