"""Agent State Management.

This module provides enhanced state management for agents with:
- Clear state definitions
- State transition tracking
- State validation
- History management
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import Dict, List, Optional, Any, Callable
import logging

logger = logging.getLogger(__name__)


class AgentState(Enum):
    """Agent execution states with clear semantics.
    
    State Transitions:
    IDLE -> INITIALIZING -> PARSING -> GENERATING -> COMPILING -> TESTING -> ANALYZING -> OPTIMIZING -> COMPLETED
                                                                          |
                                                                          v
                                                                       FIXING -> (back to COMPILING)
    
    Any state can transition to:
    - PAUSED: User requested pause
    - FAILED: Unrecoverable error
    """
    
    IDLE = auto()
    INITIALIZING = auto()
    PLANNING = auto()
    PARSING = auto()
    GENERATING = auto()
    COMPILING = auto()
    TESTING = auto()
    ANALYZING = auto()
    FIXING = auto()
    OPTIMIZING = auto()
    WAITING = auto()
    COMPLETED = auto()
    FAILED = auto()
    PAUSED = auto()
    TERMINATED = auto()
    
    @property
    def is_terminal(self) -> bool:
        """Check if this is a terminal state."""
        return self in (AgentState.COMPLETED, AgentState.FAILED, AgentState.TERMINATED)
    
    @property
    def is_active(self) -> bool:
        """Check if agent is actively working."""
        return self in (
            AgentState.INITIALIZING,
            AgentState.PLANNING,
            AgentState.PARSING,
            AgentState.GENERATING,
            AgentState.COMPILING,
            AgentState.TESTING,
            AgentState.ANALYZING,
            AgentState.FIXING,
            AgentState.OPTIMIZING,
        )
    
    @property
    def can_pause(self) -> bool:
        """Check if this state can be paused."""
        return self.is_active
    
    @property
    def can_resume(self) -> bool:
        """Check if this state can be resumed."""
        return self == AgentState.PAUSED


@dataclass
class AgentStateTransition:
    """Record of a state transition."""
    from_state: AgentState
    to_state: AgentState
    timestamp: datetime = field(default_factory=datetime.now)
    message: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "from_state": self.from_state.name,
            "to_state": self.to_state.name,
            "timestamp": self.timestamp.isoformat(),
            "message": self.message,
            "metadata": self.metadata,
        }


class StateManager:
    """Manages agent state transitions with validation and tracking.
    
    Features:
    - State transition validation
    - Transition history tracking
    - Callback support for state changes
    - Thread-safe state updates
    """
    
    VALID_TRANSITIONS: Dict[AgentState, set] = {
        AgentState.IDLE: {
            AgentState.INITIALIZING,
            AgentState.PLANNING,
            AgentState.PAUSED,
            AgentState.TERMINATED,
        },
        AgentState.INITIALIZING: {
            AgentState.PARSING,
            AgentState.PLANNING,
            AgentState.FAILED,
            AgentState.PAUSED,
            AgentState.TERMINATED,
        },
        AgentState.PLANNING: {
            AgentState.PARSING,
            AgentState.GENERATING,
            AgentState.WAITING,
            AgentState.FAILED,
            AgentState.PAUSED,
            AgentState.TERMINATED,
        },
        AgentState.PARSING: {
            AgentState.GENERATING,
            AgentState.FAILED,
            AgentState.PAUSED,
            AgentState.TERMINATED,
        },
        AgentState.GENERATING: {
            AgentState.COMPILING,
            AgentState.FAILED,
            AgentState.PAUSED,
            AgentState.TERMINATED,
        },
        AgentState.COMPILING: {
            AgentState.TESTING,
            AgentState.FIXING,
            AgentState.FAILED,
            AgentState.PAUSED,
            AgentState.TERMINATED,
        },
        AgentState.TESTING: {
            AgentState.ANALYZING,
            AgentState.FIXING,
            AgentState.FAILED,
            AgentState.PAUSED,
            AgentState.TERMINATED,
        },
        AgentState.ANALYZING: {
            AgentState.OPTIMIZING,
            AgentState.COMPLETED,
            AgentState.FAILED,
            AgentState.PAUSED,
            AgentState.TERMINATED,
        },
        AgentState.FIXING: {
            AgentState.COMPILING,
            AgentState.GENERATING,
            AgentState.FAILED,
            AgentState.PAUSED,
            AgentState.TERMINATED,
        },
        AgentState.OPTIMIZING: {
            AgentState.GENERATING,
            AgentState.COMPLETED,
            AgentState.FAILED,
            AgentState.PAUSED,
            AgentState.TERMINATED,
        },
        AgentState.WAITING: {
            AgentState.PLANNING,
            AgentState.GENERATING,
            AgentState.PAUSED,
            AgentState.TERMINATED,
        },
        AgentState.PAUSED: {
            AgentState.IDLE,
            AgentState.INITIALIZING,
            AgentState.PLANNING,
            AgentState.PARSING,
            AgentState.GENERATING,
            AgentState.COMPILING,
            AgentState.TESTING,
            AgentState.ANALYZING,
            AgentState.FIXING,
            AgentState.OPTIMIZING,
            AgentState.TERMINATED,
        },
        AgentState.FAILED: {
            AgentState.IDLE,
            AgentState.TERMINATED,
        },
        AgentState.COMPLETED: {
            AgentState.IDLE,
            AgentState.TERMINATED,
        },
        AgentState.TERMINATED: set(),
    }
    
    def __init__(
        self,
        initial_state: AgentState = AgentState.IDLE,
        on_state_change: Optional[Callable[[AgentStateTransition], None]] = None,
    ):
        """Initialize state manager.
        
        Args:
            initial_state: Starting state
            on_state_change: Optional callback for state changes
        """
        self._state = initial_state
        self._on_state_change = on_state_change
        self._history: List[AgentStateTransition] = []
        self._previous_state: Optional[AgentState] = None
        self._state_data: Dict[AgentState, Dict[str, Any]] = {}
    
    @property
    def state(self) -> AgentState:
        """Get current state."""
        return self._state
    
    @property
    def previous_state(self) -> Optional[AgentState]:
        """Get previous state."""
        return self._previous_state
    
    @property
    def history(self) -> List[AgentStateTransition]:
        """Get state transition history."""
        return self._history.copy()
    
    def can_transition_to(self, new_state: AgentState) -> bool:
        """Check if transition to new state is valid.
        
        Args:
            new_state: Target state
            
        Returns:
            True if transition is valid
        """
        valid_targets = self.VALID_TRANSITIONS.get(self._state, set())
        return new_state in valid_targets
    
    def transition(
        self,
        new_state: AgentState,
        message: str = "",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Transition to a new state.
        
        Args:
            new_state: Target state
            message: Optional message for the transition
            metadata: Optional metadata for the transition
            
        Returns:
            True if transition was successful
        """
        if not self.can_transition_to(new_state):
            logger.warning(
                f"Invalid state transition: {self._state.name} -> {new_state.name}"
            )
            return False
        
        old_state = self._state
        transition = AgentStateTransition(
            from_state=old_state,
            to_state=new_state,
            message=message,
            metadata=metadata or {},
        )
        
        self._previous_state = old_state
        self._state = new_state
        self._history.append(transition)
        
        logger.info(f"State transition: {old_state.name} -> {new_state.name}: {message}")
        
        if self._on_state_change:
            try:
                self._on_state_change(transition)
            except Exception as e:
                logger.error(f"Error in state change callback: {e}")
        
        return True
    
    def force_transition(
        self,
        new_state: AgentState,
        message: str = "",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Force a state transition without validation.
        
        Use with caution - primarily for error recovery.
        
        Args:
            new_state: Target state
            message: Optional message
            metadata: Optional metadata
        """
        old_state = self._state
        transition = AgentStateTransition(
            from_state=old_state,
            to_state=new_state,
            message=message,
            metadata=metadata or {},
        )
        
        self._previous_state = old_state
        self._state = new_state
        self._history.append(transition)
        
        logger.warning(f"Forced state transition: {old_state.name} -> {new_state.name}: {message}")
        
        if self._on_state_change:
            try:
                self._on_state_change(transition)
            except Exception as e:
                logger.error(f"Error in state change callback: {e}")
    
    def set_state_data(self, state: AgentState, data: Dict[str, Any]) -> None:
        """Store data associated with a state.
        
        Args:
            state: State to associate data with
            data: Data to store
        """
        self._state_data[state] = data
    
    def get_state_data(self, state: AgentState) -> Optional[Dict[str, Any]]:
        """Get data associated with a state.
        
        Args:
            state: State to get data for
            
        Returns:
            Stored data or None
        """
        return self._state_data.get(state)
    
    def reset(self) -> None:
        """Reset to initial state."""
        self._state = AgentState.IDLE
        self._previous_state = None
        self._history.clear()
        self._state_data.clear()
        logger.info("State manager reset to IDLE")
    
    def get_state_summary(self) -> Dict[str, Any]:
        """Get a summary of the state history.
        
        Returns:
            Summary dictionary
        """
        state_counts: Dict[AgentState, int] = {}
        for transition in self._history:
            state_counts[transition.to_state] = state_counts.get(transition.to_state, 0) + 1
        
        return {
            "current_state": self._state.name,
            "previous_state": self._previous_state.name if self._previous_state else None,
            "total_transitions": len(self._history),
            "state_counts": {s.name: c for s, c in state_counts.items()},
        }
    
    def get_last_transition_of_type(self, state: AgentState) -> Optional[AgentStateTransition]:
        """Get the last transition to a specific state.
        
        Args:
            state: State to find transition for
            
        Returns:
            Last transition to that state or None
        """
        for transition in reversed(self._history):
            if transition.to_state == state:
                return transition
        return None
