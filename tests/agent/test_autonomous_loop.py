"""Unit tests for AutonomousLoop.

This module contains comprehensive tests for the autonomous loop implementation,
covering all phases: Observe-Think-Act-Verify-Learn.
"""

import pytest
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch, Mock
import asyncio

from pyutagent.agent.autonomous_loop import (
    # Enums
    LoopPhase,
    RiskLevel,
    ActionPriority,
    
    # Data classes
    LoopState,
    Observation,
    Thought,
    Action,
    Verification,
    LearningEntry,
    RiskAssessment,
    LoopMetrics,
    LoopResult,
    LoopConfig,
    
    # Classes
    AutonomousLoop,
    DefaultAutonomousLoop,
    create_autonomous_loop,
)
from pyutagent.tools.tool import ToolResult


class TestLoopPhase:
    """Tests for LoopPhase enum."""
    
    def test_loop_phase_values(self):
        """Test that all expected phases exist."""
        phases = [
            LoopPhase.IDLE,
            LoopPhase.OBSERVING,
            LoopPhase.THINKING,
            LoopPhase.ACTING,
            LoopPhase.VERIFYING,
            LoopPhase.LEARNING,
            LoopPhase.CORRECTING,
            LoopPhase.COMPLETED,
            LoopPhase.FAILED,
            LoopPhase.PAUSED,
            LoopPhase.INTERRUPTED,
        ]
        
        assert len(phases) == 11
        assert all(isinstance(p, LoopPhase) for p in phases)


class TestRiskLevel:
    """Tests for RiskLevel enum."""
    
    def test_risk_level_values(self):
        """Test risk level values."""
        assert RiskLevel.NONE.value == "none"
        assert RiskLevel.LOW.value == "low"
        assert RiskLevel.MEDIUM.value == "medium"
        assert RiskLevel.HIGH.value == "high"
        assert RiskLevel.CRITICAL.value == "critical"


class TestActionPriority:
    """Tests for ActionPriority enum."""
    
    def test_priority_ordering(self):
        """Test that priorities have correct numeric values."""
        assert ActionPriority.LOW.value == 1
        assert ActionPriority.MEDIUM.value == 2
        assert ActionPriority.HIGH.value == 3
        assert ActionPriority.CRITICAL.value == 4


class TestLoopState:
    """Tests for LoopState dataclass."""
    
    def test_create_loop_state(self):
        """Test creating a LoopState instance."""
        state = LoopState(
            phase=LoopPhase.OBSERVING,
            iteration=1,
            task="Test task",
            context={"key": "value"},
            metadata={"test": True}
        )
        
        assert state.phase == LoopPhase.OBSERVING
        assert state.iteration == 1
        assert state.task == "Test task"
        assert state.context == {"key": "value"}
        assert state.metadata == {"test": True}
        assert isinstance(state.timestamp, datetime)
    
    def test_loop_state_to_dict(self):
        """Test converting LoopState to dictionary."""
        state = LoopState(
            phase=LoopPhase.THINKING,
            iteration=2,
            task="Test",
            context={"data": 123}
        )
        
        result = state.to_dict()
        
        assert result["phase"] == "THINKING"
        assert result["iteration"] == 2
        assert result["task"] == "Test"
        assert result["context"] == {"data": 123}
        assert "timestamp" in result


class TestObservation:
    """Tests for Observation dataclass."""
    
    def test_create_observation(self):
        """Test creating an Observation instance."""
        obs = Observation(
            timestamp=datetime.now(),
            state_summary="Test state",
            relevant_data={"files": ["test.py"]},
            tool_results=[{"tool": "read", "success": True}],
            environment_info={"os": "linux"}
        )
        
        assert obs.state_summary == "Test state"
        assert len(obs.tool_results) == 1
        assert obs.environment_info["os"] == "linux"
    
    def test_observation_to_dict(self):
        """Test converting Observation to dictionary."""
        obs = Observation(
            timestamp=datetime.now(),
            state_summary="Test",
            relevant_data={}
        )
        
        result = obs.to_dict()
        
        assert "timestamp" in result
        assert result["state_summary"] == "Test"


class TestThought:
    """Tests for Thought dataclass."""
    
    def test_create_thought(self):
        """Test creating a Thought instance."""
        thought = Thought(
            timestamp=datetime.now(),
            reasoning="Test reasoning",
            decision="Execute tool",
            confidence=0.85,
            plan=[{"tool_name": "read", "parameters": {}}],
            risk_assessment="Low risk",
            risk_level=RiskLevel.LOW
        )
        
        assert thought.confidence == 0.85
        assert thought.risk_level == RiskLevel.LOW
        assert len(thought.plan) == 1
    
    def test_thought_to_dict(self):
        """Test converting Thought to dictionary."""
        thought = Thought(
            timestamp=datetime.now(),
            reasoning="Test",
            decision="Decide",
            confidence=0.9
        )
        
        result = thought.to_dict()
        
        assert result["confidence"] == 0.9
        assert result["risk_level"] == "none"


class TestAction:
    """Tests for Action dataclass."""
    
    def test_create_action(self):
        """Test creating an Action instance."""
        action = Action(
            tool_name="read_file",
            parameters={"path": "/test.txt"},
            expected_outcome="Read file contents",
            priority=ActionPriority.HIGH,
            risk_level=RiskLevel.LOW,
            timeout=30
        )
        
        assert action.tool_name == "read_file"
        assert action.parameters["path"] == "/test.txt"
        assert action.priority == ActionPriority.HIGH
        assert action.timeout == 30


class TestVerification:
    """Tests for Verification dataclass."""
    
    def test_create_verification_success(self):
        """Test creating a successful Verification."""
        verification = Verification(
            success=True,
            actual_outcome="File read successfully",
            expected_outcome="Read file",
            confidence=1.0
        )
        
        assert verification.success is True
        assert verification.confidence == 1.0
    
    def test_create_verification_failure(self):
        """Test creating a failed Verification."""
        verification = Verification(
            success=False,
            actual_outcome="File not found",
            expected_outcome="Read file",
            differences=["File does not exist"],
            confidence=0.9
        )
        
        assert verification.success is False
        assert len(verification.differences) == 1


class TestLearningEntry:
    """Tests for LearningEntry dataclass."""
    
    def test_create_learning_entry(self):
        """Test creating a LearningEntry."""
        entry = LearningEntry(
            timestamp=datetime.now(),
            situation="Test situation",
            action_taken="read_file",
            outcome="Success",
            lesson="Reading files is useful",
            success=True,
            iteration=1
        )
        
        assert entry.success is True
        assert entry.iteration == 1
        assert entry.lesson == "Reading files is useful"


class TestRiskAssessment:
    """Tests for RiskAssessment dataclass."""
    
    def test_create_risk_assessment(self):
        """Test creating a RiskAssessment."""
        assessment = RiskAssessment(
            level=RiskLevel.HIGH,
            factors=["Deletes files", "Irreversible"],
            mitigation_strategies=["Backup first"],
            requires_approval=True
        )
        
        assert assessment.level == RiskLevel.HIGH
        assert assessment.requires_approval is True
        assert len(assessment.factors) == 2


class TestLoopMetrics:
    """Tests for LoopMetrics dataclass."""
    
    def test_create_loop_metrics(self):
        """Test creating LoopMetrics."""
        metrics = LoopMetrics(
            total_iterations=5,
            successful_actions=10,
            failed_actions=2,
            self_corrections=1
        )
        
        assert metrics.total_iterations == 5
        assert metrics.successful_actions == 10
        assert metrics.failed_actions == 2


class TestLoopResult:
    """Tests for LoopResult dataclass."""
    
    def test_create_loop_result_success(self):
        """Test creating a successful LoopResult."""
        result = LoopResult(
            success=True,
            iterations=3,
            final_state=LoopPhase.COMPLETED,
            execution_time_ms=1500
        )
        
        assert result.success is True
        assert result.iterations == 3
        assert result.execution_time_ms == 1500
    
    def test_create_loop_result_failure(self):
        """Test creating a failed LoopResult."""
        result = LoopResult(
            success=False,
            iterations=10,
            final_state=LoopPhase.FAILED,
            error="Max iterations reached"
        )
        
        assert result.success is False
        assert result.error == "Max iterations reached"
    
    def test_get_summary_success(self):
        """Test getting summary for successful result."""
        result = LoopResult(
            success=True,
            iterations=5,
            final_state=LoopPhase.COMPLETED,
            execution_time_ms=2000
        )
        
        summary = result.get_summary()
        
        assert "Success" in summary
        assert "5" in summary
        assert "2000ms" in summary
    
    def test_get_summary_failure(self):
        """Test getting summary for failed result."""
        result = LoopResult(
            success=False,
            iterations=3,
            final_state=LoopPhase.FAILED,
            error="Test error",
            execution_time_ms=1000
        )
        
        summary = result.get_summary()
        
        assert "Failed" in summary
        assert "Test error" in summary


class TestLoopConfig:
    """Tests for LoopConfig dataclass."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = LoopConfig()
        
        assert config.max_iterations == 10
        assert config.confidence_threshold == 0.8
        assert config.user_interruptible is True
        assert config.enable_self_correction is True
        assert config.enable_learning is True
    
    def test_custom_config(self):
        """Test custom configuration."""
        config = LoopConfig(
            max_iterations=20,
            confidence_threshold=0.9,
            enable_self_correction=False
        )
        
        assert config.max_iterations == 20
        assert config.confidence_threshold == 0.9
        assert config.enable_self_correction is False


class MockAutonomousLoop(AutonomousLoop):
    """Mock implementation of AutonomousLoop for testing."""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.mock_observation = Observation(
            timestamp=datetime.now(),
            state_summary="Mock state"
        )
        self.mock_thought = Thought(
            timestamp=datetime.now(),
            reasoning="Mock reasoning",
            decision="Mock decision",
            confidence=0.8,
            plan=[]
        )
        self.mock_result = ToolResult(success=True, output="Mock result")
        self.mock_verification = Verification(
            success=True,
            actual_outcome="Mock",
            expected_outcome="Mock"
        )
    
    async def _observe(self, task: str, context: dict) -> Observation:
        return self.mock_observation
    
    async def _think(self, task: str, observation: Observation, context: dict) -> Thought:
        return self.mock_thought
    
    async def _act(self, action: Action, context: dict):
        return self.mock_result
    
    async def _verify(self, result, expected: str, context: dict) -> Verification:
        return self.mock_verification


class TestAutonomousLoop:
    """Tests for AutonomousLoop base class."""
    
    @pytest.fixture
    def loop(self):
        """Create a mock autonomous loop."""
        return MockAutonomousLoop()
    
    @pytest.fixture
    def loop_with_callbacks(self):
        """Create a loop with callbacks."""
        progress_mock = MagicMock()
        result_mock = MagicMock()
        
        return MockAutonomousLoop(
            progress_callback=progress_mock,
            result_callback=result_mock
        ), progress_mock, result_mock
    
    def test_initialization(self, loop):
        """Test loop initialization."""
        assert loop.state.phase == LoopPhase.IDLE
        assert loop.current_iteration == 0
        assert loop.config.max_iterations == 10
        assert not loop.is_running
    
    def test_reset(self, loop):
        """Test loop reset."""
        loop._current_iteration = 5
        loop._observations.append(loop.mock_observation)
        
        loop.reset()
        
        assert loop.current_iteration == 0
        assert loop.state.phase == LoopPhase.IDLE
        assert len(loop._observations) == 0
    
    def test_stop(self, loop):
        """Test stop functionality."""
        loop.stop()
        
        assert loop._stop_requested is True
        assert not loop._should_continue()
    
    def test_pause_resume(self, loop):
        """Test pause and resume functionality."""
        # Test pause
        loop.pause()
        assert loop._pause_requested is True
        assert not loop._pause_event.is_set()
        
        # Test resume
        loop.resume()
        assert loop._pause_requested is False
        assert loop._pause_event.is_set()
    
    @pytest.mark.asyncio
    async def test_run_completes_successfully(self, loop):
        """Test successful loop execution."""
        # Set high confidence to complete immediately
        loop.mock_thought.confidence = 1.0
        
        result = await loop.run("Test task")
        
        assert result.success is True
        assert result.final_state == LoopPhase.COMPLETED
        assert len(result.observations) == 1
        assert len(result.thoughts) == 1
    
    @pytest.mark.asyncio
    async def test_run_with_max_iterations(self, loop):
        """Test loop stops at max iterations."""
        loop.config.max_iterations = 2
        loop.mock_thought.confidence = 0.5  # Below threshold
        
        result = await loop.run("Test task")
        
        assert result.iterations == 2
        assert result.final_state == LoopPhase.FAILED
        assert "Max iterations reached" in result.error
    
    @pytest.mark.asyncio
    async def test_run_with_stop_request(self, loop):
        """Test loop handles stop request."""
        loop.mock_thought.confidence = 0.5
        loop.config.max_iterations = 100  # High limit to test stop
        
        # Stop immediately before running
        loop.stop()
        
        result = await loop.run("Test task")
        
        assert result.success is False
        assert result.final_state == LoopPhase.INTERRUPTED
    
    def test_assess_risk_high(self, loop):
        """Test risk assessment for high-risk action."""
        action = Action(
            tool_name="delete_file",
            parameters={"force": True}
        )
        
        assessment = loop._assess_risk(action)
        
        assert assessment.level == RiskLevel.HIGH
        assert assessment.requires_approval is True
    
    def test_assess_risk_low(self, loop):
        """Test risk assessment for low-risk action."""
        action = Action(tool_name="read_file")
        
        assessment = loop._assess_risk(action)
        
        assert assessment.level == RiskLevel.LOW
        assert assessment.requires_approval is False
    
    def test_assess_risk_disabled(self, loop):
        """Test risk assessment when disabled."""
        loop.config.enable_risk_assessment = False
        action = Action(tool_name="delete_file")
        
        assessment = loop._assess_risk(action)
        
        assert assessment.level == RiskLevel.NONE
    
    @pytest.mark.asyncio
    async def test_self_correction(self, loop):
        """Test self-correction functionality."""
        loop.config.enable_self_correction = True
        
        correction = await loop._self_correct("Test error", {})
        
        assert correction is not None
        assert "Self-correction" in correction.reasoning
        assert loop._metrics.self_corrections == 1
    
    @pytest.mark.asyncio
    async def test_self_correction_disabled(self, loop):
        """Test self-correction when disabled."""
        loop.config.enable_self_correction = False
        
        correction = await loop._self_correct("Test error", {})
        
        assert correction is None
    
    @pytest.mark.asyncio
    async def test_learn(self, loop):
        """Test learning functionality."""
        loop.config.enable_learning = True
        
        entry = LearningEntry(
            timestamp=datetime.now(),
            situation="Test",
            action_taken="read",
            outcome="Success",
            lesson="Test lesson",
            success=True
        )
        
        await loop._learn(entry)
        
        assert len(loop._learnings) == 1
        assert loop._learnings[0].lesson == "Test lesson"
    
    def test_get_stats(self, loop):
        """Test getting loop statistics."""
        loop._current_iteration = 3
        loop._observations.append(loop.mock_observation)
        loop._thoughts.append(loop.mock_thought)
        
        stats = loop.get_stats()
        
        assert stats["current_iteration"] == 3
        assert stats["observations_count"] == 1
        assert stats["thoughts_count"] == 1
        assert "config" in stats


class TestDefaultAutonomousLoop:
    """Tests for DefaultAutonomousLoop."""
    
    @pytest.fixture
    def mock_tool_service(self):
        """Create a mock tool service."""
        service = MagicMock()
        service.list_available_tools = MagicMock(return_value=["read", "write", "glob"])
        service.execute_tool = AsyncMock(return_value=ToolResult(success=True, output="Test"))
        return service
    
    @pytest.fixture
    def mock_llm_client(self):
        """Create a mock LLM client."""
        client = AsyncMock()
        client.generate = AsyncMock(return_value='''
        {
            "reasoning": "Test reasoning",
            "decision": "Test decision",
            "confidence": 0.85,
            "plan": [
                {"tool_name": "read", "parameters": {}, "expected_outcome": "Read file"}
            ]
        }
        ''')
        return client
    
    @pytest.fixture
    def default_loop(self, mock_tool_service):
        """Create a DefaultAutonomousLoop."""
        return DefaultAutonomousLoop(tool_service=mock_tool_service)
    
    @pytest.fixture
    def loop_with_llm(self, mock_tool_service, mock_llm_client):
        """Create a DefaultAutonomousLoop with LLM."""
        return DefaultAutonomousLoop(
            tool_service=mock_tool_service,
            llm_client=mock_llm_client
        )
    
    @pytest.mark.asyncio
    async def test_observe(self, default_loop):
        """Test observation phase."""
        default_loop._current_iteration = 1
        
        observation = await default_loop._observe("Test task", {})
        
        assert observation.state_summary is not None
        assert "Iteration 1" in observation.state_summary
        assert "environment_info" in observation.to_dict()
    
    @pytest.mark.asyncio
    async def test_think_rule_based_test_task(self, default_loop):
        """Test rule-based thinking for test task."""
        observation = Observation(
            timestamp=datetime.now(),
            state_summary="Test"
        )
        
        thought = await default_loop._think_rule_based(
            "Generate tests for UserService",
            observation,
            {}
        )
        
        assert "test" in thought.reasoning.lower()
        assert len(thought.plan) > 0
    
    @pytest.mark.asyncio
    async def test_think_rule_based_fix_task(self, default_loop):
        """Test rule-based thinking for fix task."""
        observation = Observation(
            timestamp=datetime.now(),
            state_summary="Test"
        )
        
        thought = await default_loop._think_rule_based(
            "Fix compilation error",
            observation,
            {}
        )
        
        assert "debug" in thought.reasoning.lower() or "error" in thought.reasoning.lower()
    
    @pytest.mark.asyncio
    async def test_think_with_llm(self, loop_with_llm):
        """Test LLM-based thinking."""
        observation = Observation(
            timestamp=datetime.now(),
            state_summary="Test"
        )
        
        thought = await loop_with_llm._think_with_llm("Test task", observation, {})
        
        assert thought.confidence == 0.85
        assert len(thought.plan) == 1
        loop_with_llm.llm_client.generate.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_think_with_llm_fallback(self, loop_with_llm):
        """Test LLM fallback on error."""
        loop_with_llm.llm_client.generate = AsyncMock(side_effect=Exception("LLM error"))
        
        observation = Observation(
            timestamp=datetime.now(),
            state_summary="Test"
        )
        
        thought = await loop_with_llm._think_with_llm("Test task", observation, {})
        
        # Should fallback to rule-based
        assert thought is not None
    
    def test_parse_llm_thought_valid(self, default_loop):
        """Test parsing valid LLM response."""
        content = '''
        {
            "reasoning": "Test reasoning",
            "decision": "Test decision",
            "confidence": 0.9,
            "plan": [{"tool_name": "read", "parameters": {}}],
            "risk_assessment": "Low risk",
            "alternative_approaches": ["Option 1"]
        }
        '''
        
        thought = default_loop._parse_llm_thought(content)
        
        assert thought.confidence == 0.9
        assert thought.reasoning == "Test reasoning"
        assert len(thought.plan) == 1
    
    def test_parse_llm_thought_invalid(self, default_loop):
        """Test parsing invalid LLM response."""
        content = "Not valid JSON"
        
        thought = default_loop._parse_llm_thought(content)
        
        assert thought.confidence == 0.3  # Default low confidence
        assert "Failed to parse" in thought.reasoning
    
    @pytest.mark.asyncio
    async def test_act(self, default_loop, mock_tool_service):
        """Test action execution."""
        action = Action(
            tool_name="read",
            parameters={"path": "/test.txt"}
        )
        
        result = await default_loop._act(action, {})
        
        mock_tool_service.execute_tool.assert_called_once_with("read", {"path": "/test.txt"})
        assert result.success is True
    
    @pytest.mark.asyncio
    async def test_verify_success(self, default_loop):
        """Test verification of successful result."""
        result = ToolResult(success=True, output="Test output")
        
        verification = await default_loop._verify(result, "Test", {})
        
        assert verification.success is True
        assert verification.actual_outcome == "Test output"
    
    @pytest.mark.asyncio
    async def test_verify_failure(self, default_loop):
        """Test verification of failed result."""
        result = ToolResult(success=False, error="Test error")
        
        verification = await default_loop._verify(result, "Test", {})
        
        assert verification.success is False
        assert len(verification.differences) > 0


class TestCreateAutonomousLoop:
    """Tests for create_autonomous_loop factory function."""
    
    def test_create_with_defaults(self):
        """Test creating loop with default parameters."""
        mock_service = MagicMock()
        
        loop = create_autonomous_loop(mock_service)
        
        assert isinstance(loop, DefaultAutonomousLoop)
        assert loop.config.max_iterations == 10
        assert loop.config.confidence_threshold == 0.8
    
    def test_create_with_custom_params(self):
        """Test creating loop with custom parameters."""
        mock_service = MagicMock()
        mock_llm = MagicMock()
        
        loop = create_autonomous_loop(
            tool_service=mock_service,
            llm_client=mock_llm,
            max_iterations=20,
            confidence_threshold=0.9,
            enable_self_correction=False
        )
        
        assert loop.config.max_iterations == 20
        assert loop.config.confidence_threshold == 0.9
        assert loop.config.enable_self_correction is False
        assert loop.llm_client == mock_llm


class TestIntegration:
    """Integration tests for the full loop."""
    
    @pytest.mark.asyncio
    async def test_full_loop_execution(self):
        """Test a complete loop execution."""
        mock_service = MagicMock()
        mock_service.list_available_tools = MagicMock(return_value=["read", "glob"])
        mock_service.execute_tool = AsyncMock(return_value=ToolResult(success=True, output="Done"))
        
        loop = DefaultAutonomousLoop(
            tool_service=mock_service,
            config=LoopConfig(max_iterations=3, confidence_threshold=0.95)
        )
        
        result = await loop.run("Test task")
        
        assert isinstance(result, LoopResult)
        assert result.iterations <= 3
        assert result.execution_time_ms >= 0
    
    @pytest.mark.asyncio
    async def test_loop_with_learning(self):
        """Test loop with learning enabled."""
        mock_service = MagicMock()
        mock_service.execute_tool = AsyncMock(return_value=ToolResult(success=True, output="Done"))
        
        loop = DefaultAutonomousLoop(
            tool_service=mock_service,
            config=LoopConfig(
                max_iterations=2,
                enable_learning=True,
                confidence_threshold=0.99  # Force multiple iterations
            )
        )
        
        result = await loop.run("Test task")
        
        # Should have learned from each iteration
        assert len(result.learnings) > 0 or result.iterations < 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
