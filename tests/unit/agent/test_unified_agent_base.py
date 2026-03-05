"""Tests for UnifiedAgentBase."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
import asyncio
from dataclasses import asdict

from pyutagent.agent.unified_agent_base import (
    UnifiedAgentBase,
    AgentConfig,
    AgentState,
    AgentCapability,
    AgentResult,
    AgentMixin,
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
            metrics={"duration_ms": 10}
        )


class TestAgentConfig:
    """Test AgentConfig."""

    def test_default_config(self):
        config = AgentConfig()
        assert config.name == "unnamed_agent"
        assert config.max_iterations == 10
        assert config.timeout_seconds == 300
        assert config.enable_learning is True

    def test_custom_config(self):
        config = AgentConfig(
            name="test_agent",
            max_iterations=20,
            timeout_seconds=600,
            enable_learning=False
        )
        assert config.name == "test_agent"
        assert config.max_iterations == 20
        assert config.timeout_seconds == 600
        assert config.enable_learning is False

    def test_to_dict(self):
        config = AgentConfig(name="test")
        d = config.to_dict()
        assert d["name"] == "test"
        assert "max_iterations" in d


class TestAgentState:
    """Test AgentState enum."""

    def test_states(self):
        assert AgentState.IDLE.value == "idle"
        assert AgentState.RUNNING.value == "running"
        assert AgentState.PAUSED.value == "paused"
        assert AgentState.COMPLETED.value == "completed"
        assert AgentState.FAILED.value == "failed"


class TestAgentCapability:
    """Test AgentCapability enum."""

    def test_capabilities(self):
        assert AgentCapability.CODE_GENERATION.value == "code_generation"
        assert AgentCapability.TEST_GENERATION.value == "test_generation"
        assert AgentCapability.CODE_REVIEW.value == "code_review"
        assert AgentCapability.DEBUGGING.value == "debugging"


class TestAgentResult:
    """Test AgentResult."""

    def test_default_result(self):
        result = AgentResult()
        assert result.success is False
        assert result.output is None
        assert result.error is None

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
            metrics={"time": 100}
        )
        d = result.to_dict()
        assert d["success"] is True
        assert d["output"] == "test"
        assert d["metrics"]["time"] == 100


class TestProgressUpdate:
    """Test ProgressUpdate."""

    def test_progress_update(self):
        update = ProgressUpdate(
            agent_name="test_agent",
            progress=0.5,
            message="Half done",
            step="processing"
        )
        assert update.agent_name == "test_agent"
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
        agent = ConcreteAgent()
        assert agent.start() is True
        assert agent.state == AgentState.IDLE

    def test_stop(self):
        agent = ConcreteAgent()
        agent._state = AgentState.RUNNING
        assert agent.stop() is True
        assert agent.state == AgentState.STOPPED

    def test_pause_resume(self):
        agent = ConcreteAgent()
        agent._state = AgentState.RUNNING
        assert agent.pause() is True
        assert agent.state == AgentState.PAUSED

        assert agent.resume() is True
        assert agent.state == AgentState.RUNNING

    def test_pause_from_idle(self):
        agent = ConcreteAgent()
        assert agent.pause() is False

    @pytest.mark.asyncio
    async def test_run(self):
        agent = ConcreteAgent()
        result = await agent.run("test task")
        assert result.success is True
        assert result.output == "Task completed"

    def test_add_mixin(self):
        agent = ConcreteAgent()

        class TestMixin(AgentMixin):
            def __init__(self):
                self.called = False

            def do_something(self):
                self.called = True

        mixin = TestMixin()
        agent.add_mixin("test", mixin)

        assert agent.has_mixin("test")
        assert agent.get_mixin("test") is mixin

    def test_capabilities(self):
        agent = ConcreteAgent()
        agent.add_capability(AgentCapability.TEST_GENERATION)

        assert agent.has_capability(AgentCapability.TEST_GENERATION)
        assert not agent.has_capability(AgentCapability.CODE_REVIEW)

    def test_state_transitions(self):
        agent = ConcreteAgent()

        assert agent.state == AgentState.IDLE

        agent._state = AgentState.RUNNING
        assert agent.state == AgentState.RUNNING

        agent._state = AgentState.COMPLETED
        assert agent.state == AgentState.COMPLETED


class TestCreateAgentConfig:
    """Test factory function."""

    def test_create_default(self):
        config = create_agent_config()
        assert isinstance(config, AgentConfig)

    def test_create_with_name(self):
        config = create_agent_config(name="custom_agent")
        assert config.name == "custom_agent"

    def test_create_with_options(self):
        config = create_agent_config(
            name="test",
            max_iterations=50,
            timeout_seconds=1000
        )
        assert config.name == "test"
        assert config.max_iterations == 50
        assert config.timeout_seconds == 1000
