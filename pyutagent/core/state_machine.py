"""State machine for agent state management.

This module provides a state machine for managing agent state transitions
with validation, history tracking, and observer pattern support.
"""

import logging
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Set

logger = logging.getLogger(__name__)


class AgentState(Enum):
    """Agent states for the state machine."""
    IDLE = auto()
    PARSING = auto()
    GENERATING = auto()
    COMPILING = auto()
    TESTING = auto()
    ANALYZING = auto()
    FIXING = auto()
    OPTIMIZING = auto()
    PAUSED = auto()
    COMPLETED = auto()
    FAILED = auto()


@dataclass
class StateTransition:
    """Represents a state transition."""
    from_state: AgentState
    to_state: AgentState
    message: str = ""
    timestamp: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


class StateMachine:
    """State machine for managing agent state transitions.
    
    Features:
    - Valid transition checking
    - Transition history
    - Observer pattern for state changes
    - Debounced state updates
    - State duration tracking
    
    Attributes:
        VALID_TRANSITIONS: Dictionary mapping states to valid target states
    
    Example:
        >>> sm = StateMachine()
        >>> sm.transition(AgentState.PARSING, "Starting parse")
        True
        >>> sm.current_state
        <AgentState.PARSING: 2>
    """
    
    VALID_TRANSITIONS: Dict[AgentState, Set[AgentState]] = {
        AgentState.IDLE: {AgentState.PARSING, AgentState.PAUSED},
        AgentState.PARSING: {AgentState.GENERATING, AgentState.FAILED, AgentState.PAUSED},
        AgentState.GENERATING: {AgentState.COMPILING, AgentState.FIXING, AgentState.FAILED, AgentState.PAUSED},
        AgentState.COMPILING: {AgentState.TESTING, AgentState.FIXING, AgentState.FAILED, AgentState.PAUSED},
        AgentState.TESTING: {AgentState.ANALYZING, AgentState.FIXING, AgentState.FAILED, AgentState.PAUSED},
        AgentState.ANALYZING: {AgentState.OPTIMIZING, AgentState.COMPLETED, AgentState.FAILED, AgentState.PAUSED},
        AgentState.OPTIMIZING: {AgentState.COMPILING, AgentState.COMPLETED, AgentState.FAILED, AgentState.PAUSED},
        AgentState.FIXING: {AgentState.COMPILING, AgentState.TESTING, AgentState.GENERATING, AgentState.FAILED, AgentState.PAUSED},
        AgentState.PAUSED: {AgentState.IDLE, AgentState.PARSING, AgentState.GENERATING, AgentState.COMPILING, 
                           AgentState.TESTING, AgentState.ANALYZING, AgentState.OPTIMIZING, AgentState.FIXING},
        AgentState.COMPLETED: {AgentState.IDLE},
        AgentState.FAILED: {AgentState.IDLE, AgentState.PARSING},
    }
    
    def __init__(self, initial_state: AgentState = AgentState.IDLE):
        """Initialize state machine.
        
        Args:
            initial_state: Starting state
        """
        self._state = initial_state
        self._transition_history: List[StateTransition] = []
        self._observers: List[Callable[[AgentState, str], None]] = []
        self._last_update_time: float = 0.0
        self._debounce_seconds: float = 0.1
        self._state_enter_time: float = time.time()
    
    @property
    def current_state(self) -> AgentState:
        """Get current state."""
        return self._state
    
    @property
    def state_duration(self) -> float:
        """Get duration in current state (seconds)."""
        return time.time() - self._state_enter_time
    
    def can_transition_to(self, new_state: AgentState) -> bool:
        """Check if transition to new state is valid.
        
        Args:
            new_state: Target state
            
        Returns:
            True if transition is valid
        """
        valid_targets = self.VALID_TRANSITIONS.get(self._state, set())
        return new_state in valid_targets
    
    def transition(self, new_state: AgentState, message: str = "", metadata: Optional[Dict[str, Any]] = None) -> bool:
        """Attempt to transition to a new state.
        
        Args:
            new_state: Target state
            message: Optional message for the transition
            metadata: Optional metadata for the transition
            
        Returns:
            True if transition was successful
        """
        current_time = time.time()
        
        if new_state == self._state:
            if current_time - self._last_update_time < self._debounce_seconds:
                logger.debug(f"[StateMachine] Debouncing state update - State: {new_state.name}")
                return True
        
        if not self.can_transition_to(new_state):
            logger.warning(f"[StateMachine] Invalid transition - From: {self._state.name}, To: {new_state.name}")
            return False
        
        old_state = self._state
        self._state = new_state
        self._last_update_time = current_time
        self._state_enter_time = current_time
        
        transition = StateTransition(
            from_state=old_state,
            to_state=new_state,
            message=message,
            timestamp=current_time,
            metadata=metadata or {}
        )
        self._transition_history.append(transition)
        
        logger.info(f"[StateMachine] State transition - {old_state.name} → {new_state.name}: {message}")
        
        self._notify_observers(new_state, message)
        
        return True
    
    def force_transition(self, new_state: AgentState, message: str = ""):
        """Force a transition without validation (use with caution).
        
        Args:
            new_state: Target state
            message: Message for the transition
        """
        current_time = time.time()
        
        old_state = self._state
        self._state = new_state
        self._last_update_time = current_time
        self._state_enter_time = current_time
        
        logger.warning(f"[StateMachine] Forced transition - {old_state.name} → {new_state.name}: {message}")
        self._notify_observers(new_state, message)
    
    def add_observer(self, observer: Callable[[AgentState, str], None]):
        """Add an observer for state changes.
        
        Args:
            observer: Function to call on state change
        """
        self._observers.append(observer)
    
    def remove_observer(self, observer: Callable[[AgentState, str], None]):
        """Remove an observer.
        
        Args:
            observer: Function to remove
        """
        if observer in self._observers:
            self._observers.remove(observer)
    
    def _notify_observers(self, new_state: AgentState, message: str):
        """Notify all observers of state change.
        
        Args:
            new_state: New state
            message: Transition message
        """
        for observer in self._observers:
            try:
                observer(new_state, message)
            except Exception as e:
                logger.warning(f"[StateMachine] Observer error: {e}")
    
    def get_history(self, limit: int = 10) -> List[StateTransition]:
        """Get recent transition history.
        
        Args:
            limit: Maximum number of transitions to return
            
        Returns:
            List of recent transitions
        """
        return self._transition_history[-limit:]
    
    def get_full_history(self) -> List[StateTransition]:
        """Get full transition history.
        
        Returns:
            List of all transitions
        """
        return self._transition_history.copy()
    
    def get_state_statistics(self) -> Dict[str, Any]:
        """Get statistics about state transitions.
        
        Returns:
            Dictionary with state statistics
        """
        state_counts: Dict[AgentState, int] = {}
        state_durations: Dict[AgentState, List[float]] = {}
        
        for i, transition in enumerate(self._transition_history):
            state_counts[transition.to_state] = state_counts.get(transition.to_state, 0) + 1
            
            if i < len(self._transition_history) - 1:
                next_transition = self._transition_history[i + 1]
                duration = next_transition.timestamp - transition.timestamp
                
                if transition.to_state not in state_durations:
                    state_durations[transition.to_state] = []
                state_durations[transition.to_state].append(duration)
        
        avg_durations = {}
        for state, durations in state_durations.items():
            if durations:
                avg_durations[state.name] = sum(durations) / len(durations)
        
        return {
            "current_state": self._state.name,
            "total_transitions": len(self._transition_history),
            "state_counts": {s.name: c for s, c in state_counts.items()},
            "average_durations": avg_durations,
            "current_state_duration": self.state_duration,
        }
    
    def reset(self, reset_to: AgentState = AgentState.IDLE):
        """Reset to specified state.
        
        Args:
            reset_to: State to reset to
        """
        self._state = reset_to
        self._transition_history.clear()
        self._last_update_time = 0.0
        self._state_enter_time = time.time()
        logger.debug(f"[StateMachine] Reset to {reset_to.name}")
    
    def is_terminal_state(self) -> bool:
        """Check if current state is terminal.
        
        Returns:
            True if in a terminal state (COMPLETED, FAILED)
        """
        return self._state in (AgentState.COMPLETED, AgentState.FAILED)
    
    def is_active_state(self) -> bool:
        """Check if current state is active (processing).
        
        Returns:
            True if in an active state
        """
        active_states = {
            AgentState.PARSING,
            AgentState.GENERATING,
            AgentState.COMPILING,
            AgentState.TESTING,
            AgentState.ANALYZING,
            AgentState.OPTIMIZING,
            AgentState.FIXING,
        }
        return self._state in active_states


def create_state_machine(initial_state: AgentState = AgentState.IDLE) -> StateMachine:
    """Create a state machine with specified initial state.
    
    Args:
        initial_state: Starting state
        
    Returns:
        Configured StateMachine
    """
    return StateMachine(initial_state=initial_state)
