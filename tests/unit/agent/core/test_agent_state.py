"""Tests for Agent State Management."""

import pytest
from datetime import datetime

from pyutagent.agent.core.agent_state import (
    AgentState,
    AgentStateTransition,
    StateManager,
)


class TestAgentState:
    """Tests for AgentState enum."""
    
    def test_is_terminal(self):
        """Test terminal state detection."""
        assert AgentState.COMPLETED.is_terminal
        assert AgentState.FAILED.is_terminal
        assert AgentState.TERMINATED.is_terminal
        assert not AgentState.IDLE.is_terminal
        assert not AgentState.GENERATING.is_terminal
    
    def test_is_active(self):
        """Test active state detection."""
        assert AgentState.GENERATING.is_active
        assert AgentState.COMPILING.is_active
        assert AgentState.TESTING.is_active
        assert not AgentState.IDLE.is_active
        assert not AgentState.PAUSED.is_active
    
    def test_can_pause(self):
        """Test pausable state detection."""
        assert AgentState.GENERATING.can_pause
        assert AgentState.TESTING.can_pause
        assert not AgentState.IDLE.can_pause
        assert not AgentState.COMPLETED.can_pause
    
    def test_can_resume(self):
        """Test resumable state detection."""
        assert AgentState.PAUSED.can_resume
        assert not AgentState.IDLE.can_resume
        assert not AgentState.GENERATING.can_resume


class TestAgentStateTransition:
    """Tests for AgentStateTransition."""
    
    def test_creation(self):
        """Test transition creation."""
        transition = AgentStateTransition(
            from_state=AgentState.IDLE,
            to_state=AgentState.GENERATING,
            message="Starting generation",
        )
        
        assert transition.from_state == AgentState.IDLE
        assert transition.to_state == AgentState.GENERATING
        assert transition.message == "Starting generation"
        assert isinstance(transition.timestamp, datetime)
    
    def test_to_dict(self):
        """Test transition serialization."""
        transition = AgentStateTransition(
            from_state=AgentState.IDLE,
            to_state=AgentState.GENERATING,
            message="Test",
            metadata={"key": "value"},
        )
        
        result = transition.to_dict()
        
        assert result["from_state"] == "IDLE"
        assert result["to_state"] == "GENERATING"
        assert result["message"] == "Test"
        assert result["metadata"]["key"] == "value"


class TestStateManager:
    """Tests for StateManager."""
    
    def test_initial_state(self):
        """Test initial state is IDLE."""
        manager = StateManager()
        assert manager.state == AgentState.IDLE
    
    def test_valid_transition(self):
        """Test valid state transition."""
        manager = StateManager()
        
        result = manager.transition(AgentState.INITIALIZING, "Starting")
        
        assert result is True
        assert manager.state == AgentState.INITIALIZING
        assert manager.previous_state == AgentState.IDLE
        assert len(manager.history) == 1
    
    def test_invalid_transition(self):
        """Test invalid state transition is rejected."""
        manager = StateManager()
        
        result = manager.transition(AgentState.COMPLETED, "Invalid")
        
        assert result is False
        assert manager.state == AgentState.IDLE
    
    def test_force_transition(self):
        """Test forced transition bypasses validation."""
        manager = StateManager()
        
        manager.force_transition(AgentState.COMPLETED, "Forced")
        
        assert manager.state == AgentState.COMPLETED
    
    def test_can_transition_to(self):
        """Test transition validation."""
        manager = StateManager()
        
        assert manager.can_transition_to(AgentState.INITIALIZING)
        assert not manager.can_transition_to(AgentState.COMPLETED)
    
    def test_state_data(self):
        """Test storing data with state."""
        manager = StateManager()
        
        manager.set_state_data(AgentState.IDLE, {"key": "value"})
        
        assert manager.get_state_data(AgentState.IDLE) == {"key": "value"}
        assert manager.get_state_data(AgentState.GENERATING) is None
    
    def test_reset(self):
        """Test state manager reset."""
        manager = StateManager()
        manager.transition(AgentState.INITIALIZING, "Start")
        manager.set_state_data(AgentState.INITIALIZING, {"data": "test"})
        
        manager.reset()
        
        assert manager.state == AgentState.IDLE
        assert manager.previous_state is None
        assert len(manager.history) == 0
        assert manager.get_state_data(AgentState.INITIALIZING) is None
    
    def test_state_summary(self):
        """Test state summary generation."""
        manager = StateManager()
        manager.transition(AgentState.INITIALIZING, "Start")
        manager.transition(AgentState.PARSING, "Parse")
        manager.transition(AgentState.GENERATING, "Generate")
        
        summary = manager.get_state_summary()
        
        assert summary["current_state"] == "GENERATING"
        assert summary["previous_state"] == "PARSING"
        assert summary["total_transitions"] == 3
    
    def test_callback(self):
        """Test state change callback."""
        callback_calls = []
        
        def on_change(transition: AgentStateTransition):
            callback_calls.append(transition)
        
        manager = StateManager(on_state_change=on_change)
        manager.transition(AgentState.INITIALIZING, "Start")
        
        assert len(callback_calls) == 1
        assert callback_calls[0].to_state == AgentState.INITIALIZING
    
    def test_get_last_transition_of_type(self):
        """Test finding last transition of a type."""
        manager = StateManager()
        manager.transition(AgentState.INITIALIZING, "Start")
        manager.transition(AgentState.PARSING, "Parse")
        manager.transition(AgentState.GENERATING, "Generate")
        manager.transition(AgentState.COMPILING, "Compile")
        
        result = manager.get_last_transition_of_type(AgentState.GENERATING)
        
        assert result is not None
        assert result.from_state == AgentState.PARSING
        assert result.to_state == AgentState.GENERATING
