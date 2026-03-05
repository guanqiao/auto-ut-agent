"""Tests for UnifiedAgentBase."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
import asyncio

from pyutagent.agent.unified_agent_base import (
    UnifiedAgentBase,
    AgentConfig,
    AgentState,
    AgentCapability,
    AgentResult,
    ProgressUpdate,
    create_agent_config,
)


class ConcreteAgent(UnifiedAgentBase):
    """Concrete implementation for testing."""

    async def execute(self, task):
        self._state = AgentState.RUNNING
        await asyncio.sleep(0.01)
        self._state = AgentState.COMPLETED
        return AgentResult(
            success=True,
            output="Task completed",
            metadata={"duration_ms": 10}
        )


class TestAgentConfig:
    """Test AgentConfig."""

    def test_default_config(self):
        config = AgentConfig()
        assert config.name == "Agent"
        assert config.max_iterations == 10
        assert config.timeout == 300

    def test_custom_config(self):
        config = AgentConfig(
            name="test_agent",
            max_iterations=20,
            timeout=600,
            auto_restart=False
        )
        assert config.name == "test_agent"
        assert config.max_iterations == 20
        assert config.timeout == 600
        assert config.auto_restart is False

    def test_to_dict(self):
        config = AgentConfig(name="test")
        d = config.to_dict()
        assert d["name"] == "test"
        assert "max_iterations" in d

    def test_from_dict(self):
        data = {
            "name": "custom",
            "max_iterations": 50,
            "timeout": 1000
        }
        config = AgentConfig.from_dict(data)
        assert config.name == "custom"
        assert config.max_iterations == 50
        assert config.timeout == 1000

    def test_capabilities_in_config(self):
        config = AgentConfig(
            name="test",
            capabilities=[AgentCapability.TEST_GENERATION, AgentCapability.DEBUGGING]
        )
        assert len(config.capabilities) == 2
        assert AgentCapability.TEST_GENERATION in config.capabilities


class TestAgentState:
    """Test AgentState enum."""

    def test_states_exist(self):
        assert AgentState.IDLE is not None
        assert AgentState.RUNNING is not None
        assert AgentState.PAUSED is not None
        assert AgentState.COMPLETED is not None
        assert AgentState.FAILED is not None
        assert AgentState.STOPPED is not None
        assert AgentState.INITIALIZING is not None
        assert AgentState.STOPPING is not None


class TestAgentCapability:
    """Test AgentCapability enum."""

    def test_capabilities(self):
        assert AgentCapability.CODE_GENERATION.value == "code_generation"
        assert AgentCapability.TEST_GENERATION.value == "test_generation"
        assert AgentCapability.CODE_REVIEW.value == "code_review"
        assert AgentCapability.DEBUGGING.value == "debugging"


class TestAgentResult:
    """Test AgentResult."""

    def test_success_result(self):
        result = AgentResult(success=True, output="done")
        assert result.success is True
        assert result.output == "done"

    def test_error_result(self):
        result = AgentResult(success=False, error="Something went wrong")
        assert result.success is False
        assert result.error == "Something went wrong"

    def test_to_dict(self):
        result = AgentResult(
            success=True,
            output="test",
            iterations=5
        )
        d = result.to_dict()
        assert d["success"] is True
        assert d["output"] == "test"
        assert d["iterations"] == 5


class TestProgressUpdate:
    """Test ProgressUpdate."""

    def test_progress_update(self):
        update = ProgressUpdate(
            agent_id="agent-123",
            progress=0.5,
            status="running",
            message="Half done"
        )
        assert update.agent_id == "agent-123"
        assert update.progress == 0.5
        assert update.message == "Half done"


class TestUnifiedAgentBase:
    """Test UnifiedAgentBase."""

    def test_init(self):
        config = AgentConfig(name="test")
        agent = ConcreteAgent(config)
        assert agent.config.name == "test"
        assert agent.state == AgentState.IDLE

    def test_start(self):
        config = AgentConfig(name="test")
        agent = ConcreteAgent(config)
        assert agent.start() is True
        assert agent.state == AgentState.INITIALIZING

    def test_start_from_non_idle(self):
        config = AgentConfig(name="test")
        agent = ConcreteAgent(config)
        agent._state = AgentState.RUNNING
        assert agent.start() is False

    def test_stop(self):
        config = AgentConfig(name="test")
        agent = ConcreteAgent(config)
        agent._state = AgentState.RUNNING
        assert agent.stop() is True
        assert agent.state == AgentState.STOPPING

    def test_pause(self):
        config = AgentConfig(name="test")
        agent = ConcreteAgent(config)
        agent._state = AgentState.RUNNING
        assert agent.pause() is True
        assert agent._pause_requested is True

    def test_pause_from_idle(self):
        config = AgentConfig(name="test")
        agent = ConcreteAgent(config)
        assert agent.pause() is False

    def test_resume(self):
        config = AgentConfig(name="test")
        agent = ConcreteAgent(config)
        agent._state = AgentState.PAUSED
        assert agent.resume() is True
        assert agent.state == AgentState.RUNNING

    def test_resume_from_non_paused(self):
        config = AgentConfig(name="test")
        agent = ConcreteAgent(config)
        agent._state = AgentState.RUNNING
        assert agent.resume() is False

    def test_terminate(self):
        config = AgentConfig(name="test")
        agent = ConcreteAgent(config)
        assert agent.terminate() is True
        assert agent.state == AgentState.STOPPED

    def test_reset(self):
        config = AgentConfig(name="test")
        agent = ConcreteAgent(config)
        agent._state = AgentState.COMPLETED
        agent._current_iteration = 5
        agent.reset()
        assert agent.state == AgentState.IDLE
        assert agent._current_iteration == 0

    @pytest.mark.asyncio
    async def test_run(self):
        config = AgentConfig(name="test")
        agent = ConcreteAgent(config)
        result = await agent.run("test task")
        assert result.success is True
        assert result.output == "Task completed"

    def test_has_capability(self):
        config = AgentConfig(
            name="test",
            capabilities=[AgentCapability.TEST_GENERATION]
        )
        agent = ConcreteAgent(config)
        assert agent.has_capability(AgentCapability.TEST_GENERATION)
        assert not agent.has_capability(AgentCapability.CODE_REVIEW)

    def test_state_transitions(self):
        config = AgentConfig(name="test")
        agent = ConcreteAgent(config)

        assert agent.state == AgentState.IDLE

        agent._state = AgentState.RUNNING
        assert agent.state == AgentState.RUNNING

        agent._state = AgentState.COMPLETED
        assert agent.state == AgentState.COMPLETED

    def test_get_stats(self):
        config = AgentConfig(name="test")
        agent = ConcreteAgent(config)
        stats = agent.get_stats()
        assert stats["name"] == "test"
        assert stats["state"] == "IDLE"

    def test_get_state_history(self):
        config = AgentConfig(name="test")
        agent = ConcreteAgent(config)
        agent.state = AgentState.RUNNING
        history = agent.get_state_history()
        assert len(history) > 0


class TestCreateAgentConfig:
    """Test factory function."""

    def test_create_with_name(self):
        config = create_agent_config(name="custom_agent", agent_type="test")
        assert config.name == "custom_agent"
        assert config.agent_type == "test"

    def test_create_with_options(self):
        config = create_agent_config(
            name="test",
            agent_type="custom",
            max_iterations=50,
            timeout=1000
        )
        assert config.name == "test"
        assert config.agent_type == "custom"
        assert config.max_iterations == 50
        assert config.timeout == 1000
