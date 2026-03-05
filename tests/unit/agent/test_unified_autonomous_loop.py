"""Tests for UnifiedAutonomousLoop."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
import asyncio

from pyutagent.agent.unified_autonomous_loop import (
    UnifiedAutonomousLoop,
    LoopConfig,
    LoopState,
    LoopFeature,
    DecisionStrategy,
    Observation,
    Thought,
    Action,
    Verification,
    LearningEntry,
    LoopResult,
    create_loop_config,
)


class ConcreteLoop(UnifiedAutonomousLoop):
    """Concrete implementation for testing."""

    async def _observe(self, task: str, context: dict) -> Observation:
        return Observation(
            data={"task": task},
            insights=["Test observation"]
        )

    async def _think(self, task: str, observation: Observation, context: dict) -> Thought:
        return Thought(
            reasoning="Test reasoning",
            plan=["step1", "step2"],
            confidence=0.9
        )

    async def _act(self, action: Action, context: dict):
        return {"result": "action completed"}

    async def _verify(self, result, expected: str, context: dict) -> Verification:
        return Verification(
            success=True,
            feedback="All good",
            gaps=[]
        )


class TestLoopConfig:
    """Test LoopConfig."""

    def test_default_config(self):
        config = LoopConfig()
        assert config.max_iterations == 10
        assert config.timeout_seconds == 300
        assert LoopFeature.OBSERVE in config.enabled_features

    def test_custom_config(self):
        config = LoopConfig(
            max_iterations=20,
            timeout_seconds=600,
            decision_strategy=DecisionStrategy.EXPLORATION
        )
        assert config.max_iterations == 20
        assert config.timeout_seconds == 600
        assert config.decision_strategy == DecisionStrategy.EXPLORATION

    def test_feature_enable_disable(self):
        config = LoopConfig()
        config.enable_feature(LoopFeature.LEARN)
        assert LoopFeature.LEARN in config.enabled_features

        config.disable_feature(LoopFeature.LEARN)
        assert LoopFeature.LEARN not in config.enabled_features


class TestLoopState:
    """Test LoopState enum."""

    def test_states(self):
        assert LoopState.IDLE.value == "idle"
        assert LoopState.OBSERVING.value == "observing"
        assert LoopState.THINKING.value == "thinking"
        assert LoopState.ACTING.value == "acting"
        assert LoopState.VERIFYING.value == "verifying"
        assert LoopState.LEARNING.value == "learning"
        assert LoopState.COMPLETED.value == "completed"
        assert LoopState.FAILED.value == "failed"


class TestLoopFeature:
    """Test LoopFeature enum."""

    def test_features(self):
        assert LoopFeature.OBSERVE.value == "observe"
        assert LoopFeature.THINK.value == "think"
        assert LoopFeature.ACT.value == "act"
        assert LoopFeature.VERIFY.value == "verify"
        assert LoopFeature.LEARN.value == "learn"


class TestDecisionStrategy:
    """Test DecisionStrategy enum."""

    def test_strategies(self):
        assert DecisionStrategy.GREEDY.value == "greedy"
        assert DecisionStrategy.EXPLORATION.value == "exploration"
        assert DecisionStrategy.BALANCED.value == "balanced"


class TestObservation:
    """Test Observation."""

    def test_observation(self):
        obs = Observation(
            data={"key": "value"},
            insights=["insight1", "insight2"]
        )
        assert obs.data == {"key": "value"}
        assert len(obs.insights) == 2

    def test_empty_observation(self):
        obs = Observation()
        assert obs.data == {}
        assert obs.insights == []


class TestThought:
    """Test Thought."""

    def test_thought(self):
        thought = Thought(
            reasoning="Test reasoning",
            plan=["step1"],
            confidence=0.8
        )
        assert thought.reasoning == "Test reasoning"
        assert thought.confidence == 0.8

    def test_default_confidence(self):
        thought = Thought(reasoning="test")
        assert thought.confidence == 1.0


class TestAction:
    """Test Action."""

    def test_action(self):
        action = Action(
            type="execute",
            target="file.py",
            parameters={"mode": "write"}
        )
        assert action.type == "execute"
        assert action.target == "file.py"

    def test_default_parameters(self):
        action = Action(type="read")
        assert action.parameters == {}


class TestVerification:
    """Test Verification."""

    def test_success_verification(self):
        verification = Verification(
            success=True,
            feedback="All tests passed",
            gaps=[]
        )
        assert verification.success is True
        assert verification.gaps == []

    def test_failure_verification(self):
        verification = Verification(
            success=False,
            feedback="Some tests failed",
            gaps=["gap1", "gap2"]
        )
        assert verification.success is False
        assert len(verification.gaps) == 2


class TestLearningEntry:
    """Test LearningEntry."""

    def test_learning_entry(self):
        entry = LearningEntry(
            iteration=1,
            observation="obs",
            thought="thought",
            action="action",
            result="result",
            success=True
        )
        assert entry.iteration == 1
        assert entry.success is True


class TestLoopResult:
    """Test LoopResult."""

    def test_success_result(self):
        result = LoopResult(
            success=True,
            output="Task completed",
            iterations=5,
            learnings=[]
        )
        assert result.success is True
        assert result.iterations == 5

    def test_failure_result(self):
        result = LoopResult(
            success=False,
            error="Max iterations reached",
            iterations=10
        )
        assert result.success is False
        assert result.error == "Max iterations reached"


class TestUnifiedAutonomousLoop:
    """Test UnifiedAutonomousLoop."""

    def test_init(self):
        config = LoopConfig()
        loop = ConcreteLoop(config)
        assert loop.config.max_iterations == 10
        assert loop.state == LoopState.IDLE

    def test_init_with_config(self):
        config = LoopConfig(max_iterations=20)
        loop = ConcreteLoop(config)
        assert loop.config.max_iterations == 20

    @pytest.mark.asyncio
    async def test_run(self):
        loop = ConcreteLoop()
        result = await loop.run("test task")
        assert result.success is True

    @pytest.mark.asyncio
    async def test_run_with_context(self):
        loop = ConcreteLoop()
        result = await loop.run("test task", context={"key": "value"})
        assert result.success is True

    def test_state_transitions(self):
        loop = ConcreteLoop()
        assert loop.state == LoopState.IDLE

        loop._state = LoopState.OBSERVING
        assert loop.state == LoopState.OBSERVING

    @pytest.mark.asyncio
    async def test_observe_think_act_verify_cycle(self):
        loop = ConcreteLoop()

        observation = await loop._observe("task", {})
        assert observation.insights == ["Test observation"]

        thought = await loop._think("task", observation, {})
        assert thought.confidence == 0.9

        action = Action(type="test")
        result = await loop._act(action, {})
        assert result == {"result": "action completed"}

        verification = await loop._verify(result, "expected", {})
        assert verification.success is True


class TestCreateLoopConfig:
    """Test factory function."""

    def test_create_default(self):
        config = create_loop_config()
        assert isinstance(config, LoopConfig)
        assert config.max_iterations == 10

    def test_create_with_options(self):
        config = create_loop_config(
            max_iterations=50,
            timeout_seconds=1000,
            decision_strategy=DecisionStrategy.EXPLORATION
        )
        assert config.max_iterations == 50
        assert config.timeout_seconds == 1000
        assert config.decision_strategy == DecisionStrategy.EXPLORATION
