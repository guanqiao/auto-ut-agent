"""Unit tests for ThinkingEngine."""

import pytest
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

from pyutagent.agent.thinking_engine import (
    ThinkingEngine,
    ThinkingType,
    ThinkingPhase,
    ConfidenceLevel,
    ReasoningStep,
    PredictedIssue,
    ThinkingResult,
    ErrorThinkingResult,
    ThinkingSession,
)


class TestReasoningStep:
    """Tests for ReasoningStep."""

    def test_create_reasoning_step(self):
        step = ReasoningStep(
            step_id="test_001",
            phase=ThinkingPhase.PERCEPTION,
            thought="Test thought",
            evidence=["evidence1", "evidence2"],
            conclusions=["conclusion1"],
            confidence=0.8,
        )

        assert step.step_id == "test_001"
        assert step.phase == ThinkingPhase.PERCEPTION
        assert step.thought == "Test thought"
        assert len(step.evidence) == 2
        assert step.confidence == 0.8

    def test_to_dict(self):
        step = ReasoningStep(
            step_id="test_001",
            phase=ThinkingPhase.ANALYSIS,
            thought="Analyzing",
            evidence=["e1"],
            conclusions=["c1"],
            confidence=0.9,
        )

        result = step.to_dict()

        assert result["step_id"] == "test_001"
        assert result["phase"] == "ANALYSIS"
        assert result["thought"] == "Analyzing"
        assert result["confidence"] == 0.9


class TestPredictedIssue:
    """Tests for PredictedIssue."""

    def test_create_predicted_issue(self):
        issue = PredictedIssue(
            issue_type="NULL_POINTER",
            description="Mock not initialized",
            risk_level="HIGH",
            probability=0.8,
            impact="Test failure",
            prevention_suggestion="Initialize mock with when().thenReturn()",
            confidence=0.85,
        )

        assert issue.issue_type == "NULL_POINTER"
        assert issue.risk_level == "HIGH"
        assert issue.probability == 0.8

    def test_to_dict(self):
        issue = PredictedIssue(
            issue_type="COMPILATION_ERROR",
            description="Missing import",
            risk_level="MEDIUM",
            probability=0.6,
            impact="Build failure",
            prevention_suggestion="Add import statement",
        )

        result = issue.to_dict()

        assert result["issue_type"] == "COMPILATION_ERROR"
        assert result["risk_level"] == "MEDIUM"


class TestThinkingResult:
    """Tests for ThinkingResult."""

    def test_create_thinking_result(self):
        result = ThinkingResult(
            thinking_id="think_001",
            thinking_type=ThinkingType.ANALYTICAL,
            situation="Test situation",
            context={"key": "value"},
            reasoning_chain=[],
            conclusions=["conclusion1"],
            recommendations=["rec1"],
            confidence=0.85,
            duration=1.5,
            llm_calls=2,
        )

        assert result.thinking_id == "think_001"
        assert result.thinking_type == ThinkingType.ANALYTICAL
        assert result.confidence == 0.85

    def test_get_confidence_level_very_high(self):
        result = ThinkingResult(
            thinking_id="test",
            thinking_type=ThinkingType.ANALYTICAL,
            situation="test",
            context={},
            reasoning_chain=[],
            conclusions=[],
            recommendations=[],
            confidence=0.95,
            duration=0.5,
            llm_calls=0,
        )

        assert result.get_confidence_level() == ConfidenceLevel.VERY_HIGH

    def test_get_confidence_level_high(self):
        result = ThinkingResult(
            thinking_id="test",
            thinking_type=ThinkingType.ANALYTICAL,
            situation="test",
            context={},
            reasoning_chain=[],
            conclusions=[],
            recommendations=[],
            confidence=0.75,
            duration=0.5,
            llm_calls=0,
        )

        assert result.get_confidence_level() == ConfidenceLevel.HIGH

    def test_get_confidence_level_medium(self):
        result = ThinkingResult(
            thinking_id="test",
            thinking_type=ThinkingType.ANALYTICAL,
            situation="test",
            context={},
            reasoning_chain=[],
            conclusions=[],
            recommendations=[],
            confidence=0.55,
            duration=0.5,
            llm_calls=0,
        )

        assert result.get_confidence_level() == ConfidenceLevel.MEDIUM

    def test_get_confidence_level_low(self):
        result = ThinkingResult(
            thinking_id="test",
            thinking_type=ThinkingType.ANALYTICAL,
            situation="test",
            context={},
            reasoning_chain=[],
            conclusions=[],
            recommendations=[],
            confidence=0.35,
            duration=0.5,
            llm_calls=0,
        )

        assert result.get_confidence_level() == ConfidenceLevel.LOW

    def test_format_for_display(self):
        step = ReasoningStep(
            step_id="step_001",
            phase=ThinkingPhase.PERCEPTION,
            thought="Observing the situation",
            evidence=["Evidence 1"],
            conclusions=["Conclusion 1"],
            confidence=0.8,
        )

        result = ThinkingResult(
            thinking_id="think_001",
            thinking_type=ThinkingType.ANALYTICAL,
            situation="Test situation",
            context={},
            reasoning_chain=[step],
            conclusions=["Final conclusion"],
            recommendations=["Do this"],
            confidence=0.8,
            duration=1.0,
            llm_calls=1,
        )

        display = result.format_for_display()

        assert "[Thinking]" in display
        assert "Test situation" in display
        assert "ANALYTICAL" in display
        assert "80%" in display


class TestErrorThinkingResult:
    """Tests for ErrorThinkingResult."""

    def test_create_error_thinking_result(self):
        result = ErrorThinkingResult(
            thinking_id="error_001",
            thinking_type=ThinkingType.DIAGNOSTIC,
            situation="Compilation error",
            context={},
            reasoning_chain=[],
            conclusions=["Missing import"],
            recommendations=["Add import statement"],
            confidence=0.85,
            duration=1.0,
            llm_calls=1,
            error_category="COMPILATION_ERROR",
            root_cause="Missing import for ArrayList",
            recovery_strategy="ANALYZE_AND_FIX",
            fix_suggestions=["Add import java.util.ArrayList"],
        )

        assert result.error_category == "COMPILATION_ERROR"
        assert result.root_cause == "Missing import for ArrayList"
        assert result.recovery_strategy == "ANALYZE_AND_FIX"

    def test_to_dict_includes_error_fields(self):
        result = ErrorThinkingResult(
            thinking_id="error_001",
            thinking_type=ThinkingType.DIAGNOSTIC,
            situation="Test error",
            context={},
            reasoning_chain=[],
            conclusions=[],
            recommendations=[],
            confidence=0.8,
            duration=0.5,
            llm_calls=0,
            error_category="TEST_FAILURE",
            root_cause="Assertion failed",
            recovery_strategy="RETRY",
        )

        result_dict = result.to_dict()

        assert "error_category" in result_dict
        assert "root_cause" in result_dict
        assert "recovery_strategy" in result_dict


class TestThinkingSession:
    """Tests for ThinkingSession."""

    def test_create_session(self):
        session = ThinkingSession(
            session_id="session_001",
            start_time=datetime.now(),
        )

        assert session.session_id == "session_001"
        assert len(session.thinking_history) == 0

    def test_add_thinking(self):
        session = ThinkingSession(
            session_id="session_001",
            start_time=datetime.now(),
        )

        result = ThinkingResult(
            thinking_id="think_001",
            thinking_type=ThinkingType.ANALYTICAL,
            situation="Test",
            context={},
            reasoning_chain=[],
            conclusions=["conclusion1"],
            recommendations=[],
            confidence=0.8,
            duration=0.5,
            llm_calls=0,
        )

        session.add_thinking(result)

        assert len(session.thinking_history) == 1
        assert len(session.accumulated_insights) == 1

    def test_get_summary(self):
        session = ThinkingSession(
            session_id="session_001",
            start_time=datetime.now(),
        )

        result = ThinkingResult(
            thinking_id="think_001",
            thinking_type=ThinkingType.ANALYTICAL,
            situation="Test",
            context={},
            reasoning_chain=[],
            conclusions=["c1"],
            recommendations=[],
            confidence=0.8,
            duration=0.5,
            llm_calls=0,
        )

        session.add_thinking(result)
        summary = session.get_summary()

        assert summary["session_id"] == "session_001"
        assert summary["total_thinking_processes"] == 1
        assert summary["average_confidence"] == 0.8


class TestThinkingEngine:
    """Tests for ThinkingEngine."""

    @pytest.fixture
    def engine(self):
        return ThinkingEngine(
            enable_deep_thinking=True,
            max_reasoning_steps=10,
            enable_prediction=True,
        )

    @pytest.fixture
    def engine_with_llm(self):
        mock_llm = AsyncMock()
        mock_llm.generate = AsyncMock(return_value="THOUGHT: Test thought\nEVIDENCE:\n- e1\nCONCLUSIONS:\n- c1\nCONFIDENCE: 0.8")

        mock_prompt_builder = MagicMock()
        mock_prompt_builder.build_thinking_prompt = MagicMock(return_value="Test prompt")

        return ThinkingEngine(
            llm_client=mock_llm,
            prompt_builder=mock_prompt_builder,
            enable_deep_thinking=True,
            enable_prediction=True,
        )

    def test_initialization(self, engine):
        assert engine.enable_deep_thinking is True
        assert engine.max_reasoning_steps == 10
        assert engine.enable_prediction is True
        assert len(engine._thinking_history) == 0

    def test_create_session(self, engine):
        session = engine.create_session()

        assert session.session_id is not None
        assert session.session_id in engine._sessions

    @pytest.mark.asyncio
    async def test_think_basic(self, engine):
        result = await engine.think(
            situation="Test situation",
            context={"key": "value"},
            thinking_type=ThinkingType.ANALYTICAL,
        )

        assert result.thinking_id is not None
        assert result.thinking_type == ThinkingType.ANALYTICAL
        assert result.situation == "Test situation"
        assert len(result.reasoning_chain) > 0
        assert result.confidence >= 0.0

    @pytest.mark.asyncio
    async def test_think_with_session(self, engine):
        session = engine.create_session()

        result = await engine.think(
            situation="Test",
            context={},
            session=session,
        )

        assert len(session.thinking_history) == 1
        assert session.thinking_history[0].thinking_id == result.thinking_id

    @pytest.mark.asyncio
    async def test_think_about_error(self, engine):
        error = ValueError("Test error message")

        result = await engine.think_about_error(
            error=error,
            context={"step": "test_step"},
        )

        assert result.thinking_type == ThinkingType.DIAGNOSTIC
        assert result.error_category is not None
        assert result.root_cause is not None
        assert result.recovery_strategy is not None

    @pytest.mark.asyncio
    async def test_think_about_compilation_error(self, engine):
        error = Exception("cannot find symbol: class ArrayList")

        result = await engine.think_about_error(
            error=error,
            context={"compiler_output": "error: cannot find symbol"},
        )

        assert result.error_category == "COMPILATION_ERROR"

    @pytest.mark.asyncio
    async def test_predict_issues(self, engine):
        current_state = {
            "test_code": """
                @Mock
                private UserService userService;
                
                @Test
                void testSomething() {
                    // No when() setup
                }
            """,
            "coverage": 0.4,
        }

        issues = await engine.predict_issues(current_state, {})

        assert len(issues) > 0
        issue_types = [i.issue_type for i in issues]
        assert "UNSTUBBED_MOCK" in issue_types or "LOW_COVERAGE" in issue_types

    @pytest.mark.asyncio
    async def test_predict_issues_disabled(self):
        engine = ThinkingEngine(enable_prediction=False)

        issues = await engine.predict_issues({}, {})

        assert len(issues) == 0

    def test_categorize_error(self, engine):
        error = Exception("cannot find symbol: class ArrayList")
        category = engine._categorize_error(error, "cannot find symbol")

        assert category == "COMPILATION_ERROR"

    def test_categorize_network_error(self, engine):
        error = Exception("connection timeout")
        category = engine._categorize_error(error, "connection timeout")

        assert category == "TIMEOUT"

    def test_get_thinking_stats(self, engine):
        stats = engine.get_thinking_stats()

        assert "total" in stats
        assert stats["total"] == 0

    @pytest.mark.asyncio
    async def test_get_thinking_stats_after_thinking(self, engine):
        await engine.think("Test 1", {})
        await engine.think("Test 2", {})

        stats = engine.get_thinking_stats()

        assert stats["total"] == 2
        assert "average_confidence" in stats

    def test_get_recent_thinking(self, engine):
        engine._thinking_history = [
            ThinkingResult(
                thinking_id=f"think_{i}",
                thinking_type=ThinkingType.ANALYTICAL,
                situation=f"Test {i}",
                context={},
                reasoning_chain=[],
                conclusions=[],
                recommendations=[],
                confidence=0.8,
                duration=0.5,
                llm_calls=0,
            )
            for i in range(5)
        ]

        recent = engine.get_recent_thinking(limit=3)

        assert len(recent) == 3
        assert recent[0].thinking_id == "think_2"

    def test_clear_history(self, engine):
        engine._thinking_history = [
            ThinkingResult(
                thinking_id="test",
                thinking_type=ThinkingType.ANALYTICAL,
                situation="test",
                context={},
                reasoning_chain=[],
                conclusions=[],
                recommendations=[],
                confidence=0.8,
                duration=0.5,
                llm_calls=0,
            )
        ]

        engine.clear_history()

        assert len(engine._thinking_history) == 0


class TestThinkingEngineIntegration:
    """Integration tests for ThinkingEngine."""

    @pytest.fixture
    def mock_llm_client(self):
        client = AsyncMock()
        client.generate = AsyncMock(return_value="""
THOUGHT: The error indicates a missing dependency
EVIDENCE:
- Cannot find symbol: ArrayList
- Package java.util does not exist
CONCLUSIONS:
- Missing import statement
- Need to add java.util.ArrayList import
CONFIDENCE: 0.85
""")
        return client

    @pytest.fixture
    def mock_prompt_builder(self):
        builder = MagicMock()
        builder.build_thinking_prompt = MagicMock(return_value="Test prompt")
        builder.build_error_thinking_prompt = MagicMock(return_value="Error prompt")
        return builder

    @pytest.mark.asyncio
    async def test_full_thinking_flow(self, mock_llm_client, mock_prompt_builder):
        engine = ThinkingEngine(
            llm_client=mock_llm_client,
            prompt_builder=mock_prompt_builder,
            enable_deep_thinking=True,
            enable_prediction=True,
        )

        result = await engine.think(
            situation="Analyzing compilation error",
            context={
                "error": "cannot find symbol",
                "test_code": "List<String> list = new ArrayList<>();",
            },
            thinking_type=ThinkingType.ANALYTICAL,
        )

        assert result is not None
        assert len(result.reasoning_chain) > 0
        assert result.confidence > 0

    @pytest.mark.asyncio
    async def test_error_recovery_thinking_flow(self, mock_llm_client, mock_prompt_builder):
        engine = ThinkingEngine(
            llm_client=mock_llm_client,
            prompt_builder=mock_prompt_builder,
        )

        error = Exception("Test failed: AssertionError")
        result = await engine.think_about_error(
            error=error,
            context={
                "test_code": "assertEquals(expected, actual);",
                "failures": [{"test_name": "testMethod", "error": "AssertionError"}],
            },
        )

        assert result is not None
        assert result.error_category is not None
        assert result.recovery_strategy is not None
        assert len(result.fix_suggestions) >= 0
