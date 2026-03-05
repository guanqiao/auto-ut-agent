"""Tests for UnifiedAutonomousLoop."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
import asyncio
from datetime import datetime

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
            timestamp=datetime.now(),
            state_summary="Test observation",
            relevant_data={"task": task}
        )

    async def _think(self, task: str, observation: Observation, context: dict) -> Thought:
        return Thought(
            timestamp=datetime.now(),
            reasoning="Test reasoning",
            decision="proceed",
            confidence=0.9,
            plan=[{"step": 1}]
        )

    async def _act(self, action: Action, context: dict):
        return {"result": "action completed"}

    async def _verify(self, result, expected: str, context: dict) -> Verification:
        return Verification(
            success=True,
            actual_outcome="completed",
            expected_outcome=expected
        )


class TestLoopConfig:
    """Test LoopConfig."""

    def test_default_config(self):
        config = LoopConfig()
        assert config.max_iterations == 10
        assert config.timeout == 300
        assert config.confidence_threshold == 0.8

    def test_custom_config(self):
        config = LoopConfig(
            max_iterations=20,
            timeout=600,
            decision_strategy=DecisionStrategy.LLM_BASED
        )
        assert config.max_iterations == 20
        assert config.timeout == 600
        assert config.decision_strategy == DecisionStrategy.LLM_BASED

    def test_feature_methods(self):
        config = LoopConfig()
        config.enable_feature(LoopFeature.DELEGATION)
        assert config.has_feature(LoopFeature.DELEGATION)

        config.disable_feature(LoopFeature.DELEGATION)
        assert not config.has_feature(LoopFeature.DELEGATION)


class TestLoopState:
    """Test LoopState enum."""

    def test_states_exist(self):
        assert LoopState.IDLE is not None
        assert LoopState.OBSERVING is not None
        assert LoopState.THINKING is not None
        assert LoopState.ACTING is not None
        assert LoopState.VERIFYING is not None
        assert LoopState.COMPLETED is not None
        assert LoopState.FAILED is not None


class TestLoopFeature:
    """Test LoopFeature enum."""

    def test_features(self):
        assert LoopFeature.LLM_REASONING.value == "llm_reasoning"
        assert LoopFeature.SELF_CORRECTION.value == "self_correction"
        assert LoopFeature.DELEGATION.value == "delegation"
        assert LoopFeature.LEARNING.value == "learning"


class TestDecisionStrategy:
    """Test DecisionStrategy enum."""

    def test_strategies(self):
        assert DecisionStrategy.RULE_BASED is not None
        assert DecisionStrategy.LLM_BASED is not None
        assert DecisionStrategy.HYBRID is not None
        assert DecisionStrategy.ADAPTIVE is not None


class TestObservation:
    """Test Observation."""

    def test_observation(self):
        obs = Observation(
            timestamp=datetime.now(),
            state_summary="test state",
            relevant_data={"key": "value"}
        )
        assert obs.state_summary == "test state"
        assert obs.relevant_data == {"key": "value"}

    def test_default_values(self):
        obs = Observation(
            timestamp=datetime.now(),
            state_summary="test",
            relevant_data={}
        )
        assert obs.tool_results == []
        assert obs.metadata == {}


class TestThought:
    """Test Thought."""

    def test_thought(self):
        thought = Thought(
            timestamp=datetime.now(),
            reasoning="Test reasoning",
            decision="proceed",
            confidence=0.8
        )
        assert thought.reasoning == "Test reasoning"
        assert thought.confidence == 0.8

    def test_default_values(self):
        thought = Thought(
            timestamp=datetime.now(),
            reasoning="test",
            decision="test",
            confidence=1.0
        )
        assert thought.plan == []
        assert thought.tool_recommendations == []


class TestAction:
    """Test Action."""

    def test_action(self):
        action = Action(
            tool_name="test_tool",
            parameters={"arg": "value"},
            expected_outcome="success"
        )
        assert action.tool_name == "test_tool"
        assert action.parameters == {"arg": "value"}

    def test_default_values(self):
        action = Action(
            tool_name="read",
            parameters={},
            expected_outcome="data"
        )
        assert action.is_delegation is False
        assert action.metadata == {}


class TestVerification:
    """Test Verification."""

    def test_success_verification(self):
        verification = Verification(
            success=True,
            actual_outcome="All tests passed",
            expected_outcome="All tests pass"
        )
        assert verification.success is True
        assert verification.differences == []

    def test_failure_verification(self):
        verification = Verification(
            success=False,
            actual_outcome="Some tests failed",
            expected_outcome="All tests pass",
            differences=["gap1", "gap2"]
        )
        assert verification.success is False
        assert len(verification.differences) == 2


class TestLearningEntry:
    """Test LearningEntry."""

    def test_learning_entry(self):
        entry = LearningEntry(
            timestamp=datetime.now(),
            situation="test situation",
            action_taken="test action",
            outcome="success",
            lesson="test lesson",
            success=True
        )
        assert entry.success is True
        assert entry.lesson == "test lesson"


class TestLoopResult:
    """Test LoopResult."""

    def test_success_result(self):
        result = LoopResult(
            success=True,
            iterations=5,
            final_state=LoopState.COMPLETED
        )
        assert result.success is True
        assert result.iterations == 5

    def test_failure_result(self):
        result = LoopResult(
            success=False,
            iterations=10,
            final_state=LoopState.FAILED,
            error="Max iterations reached"
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
    async def test_observe_think_act_verify(self):
        loop = ConcreteLoop()

        observation = await loop._observe("task", {})
        assert observation.state_summary == "Test observation"

        thought = await loop._think("task", observation, {})
        assert thought.confidence == 0.9

        action = Action(
            tool_name="test",
            parameters={},
            expected_outcome="result"
        )
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
            timeout=1000,
            decision_strategy=DecisionStrategy.LLM_BASED
        )
        assert config.max_iterations == 50
        assert config.timeout == 1000
        assert config.decision_strategy == DecisionStrategy.LLM_BASED
