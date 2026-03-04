"""Unit tests for EnhancedFeedbackLoop module."""

import pytest
import tempfile
import os
from datetime import datetime
from unittest.mock import patch, MagicMock

from pyutagent.core.enhanced_feedback_loop import (
    EnhancedFeedbackLoop,
    FeedbackType,
    LearningCategory,
    FeedbackEvent,
    LearningInsight,
    AdaptiveAdjustment,
)


class TestEnhancedFeedbackLoop:
    """Tests for EnhancedFeedbackLoop class."""

    def test_init_default_db(self):
        """Test initialization with default database."""
        feedback = EnhancedFeedbackLoop()
        
        assert feedback.db_path is not None
        assert feedback._event_buffer == []
        assert isinstance(feedback._insight_cache, dict)

    def test_init_custom_db(self):
        """Test initialization with custom database."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test_feedback.db")
            feedback = EnhancedFeedbackLoop(db_path=db_path)
            
            assert feedback.db_path == db_path

    def test_record_feedback(self):
        """Test recording feedback event."""
        feedback = EnhancedFeedbackLoop()
        
        event_id = feedback.record_feedback(
            feedback_type=FeedbackType.TEST_PASS,
            context={"test_name": "test1"},
            outcome="success"
        )
        
        assert event_id is not None
        assert len(feedback._event_buffer) == 1

    def test_record_feedback_with_details(self):
        """Test recording feedback with details."""
        feedback = EnhancedFeedbackLoop()
        
        event_id = feedback.record_feedback(
            feedback_type=FeedbackType.COMPILATION_FAILURE,
            context={"file": "Test.java"},
            outcome="failed",
            details={"errors": ["error1", "error2"]},
            session_id="session-123"
        )
        
        assert event_id is not None

    def test_record_compilation_result_success(self):
        """Test recording successful compilation."""
        feedback = EnhancedFeedbackLoop()
        
        feedback.record_compilation_result(
            success=True,
            errors=[],
            context={"file": "Test.java"}
        )
        
        stats = feedback.get_learning_stats()
        assert stats["total_events"] >= 1

    def test_record_compilation_result_failure(self):
        """Test recording failed compilation."""
        feedback = EnhancedFeedbackLoop()
        
        feedback.record_compilation_result(
            success=False,
            errors=[{"type": "SyntaxError", "message": "Missing semicolon"}],
            context={"file": "Test.java"}
        )
        
        stats = feedback.get_learning_stats()
        assert stats["total_events"] >= 1

    def test_record_test_result_pass(self):
        """Test recording passed test."""
        feedback = EnhancedFeedbackLoop()
        
        feedback.record_test_result(
            test_name="testAdd",
            passed=True,
            context={"class": "CalculatorTest"}
        )
        
        stats = feedback.get_learning_stats()
        assert stats["total_events"] >= 1

    def test_record_test_result_failure(self):
        """Test recording failed test."""
        feedback = EnhancedFeedbackLoop()
        
        feedback.record_test_result(
            test_name="testDivide",
            passed=False,
            failure_reason="AssertionError: expected 5 but was 4",
            context={"class": "CalculatorTest"}
        )
        
        stats = feedback.get_learning_stats()
        assert stats["total_events"] >= 1

    def test_record_coverage_change_improvement(self):
        """Test recording coverage improvement."""
        feedback = EnhancedFeedbackLoop()
        
        feedback.record_coverage_change(
            old_coverage=0.5,
            new_coverage=0.7,
            context={"test_file": "Test.java"}
        )
        
        stats = feedback.get_learning_stats()
        event_counts = stats.get("event_distribution", {})
        assert "coverage_improvement" in event_counts or stats["total_events"] >= 1

    def test_record_coverage_change_regression(self):
        """Test recording coverage regression."""
        feedback = EnhancedFeedbackLoop()
        
        feedback.record_coverage_change(
            old_coverage=0.8,
            new_coverage=0.6,
            context={"test_file": "Test.java"}
        )
        
        stats = feedback.get_learning_stats()
        assert stats["total_events"] >= 1

    def test_get_adaptive_adjustments(self):
        """Test getting adaptive adjustments."""
        feedback = EnhancedFeedbackLoop()
        
        feedback.record_feedback(
            feedback_type=FeedbackType.TEST_FAILURE,
            context={"error_type": "NullPointerException"},
            outcome="failed"
        )
        
        adjustments = feedback.get_adaptive_adjustments({"error_type": "NullPointerException"})
        
        assert isinstance(adjustments, list)

    def test_get_learning_stats(self):
        """Test getting learning statistics."""
        feedback = EnhancedFeedbackLoop()
        
        feedback.record_feedback(FeedbackType.TEST_PASS, {}, "success")
        feedback.record_feedback(FeedbackType.TEST_FAILURE, {}, "failed")
        
        stats = feedback.get_learning_stats()
        
        assert "total_events" in stats
        assert "event_distribution" in stats
        assert stats["total_events"] >= 2

    def test_update_error_solution(self):
        """Test updating error solution."""
        feedback = EnhancedFeedbackLoop()
        
        feedback._learn_error_pattern(
            {"type": "NullPointerException", "message": "null reference"},
            {"class": "Service"}
        )
        
        feedback.update_error_solution(
            "NullPointerException",
            "Add null check before accessing object"
        )

    def test_clear_old_events(self):
        """Test clearing old events."""
        feedback = EnhancedFeedbackLoop()
        
        feedback.record_feedback(FeedbackType.TEST_PASS, {}, "success")
        
        feedback.clear_old_events(days=0)

    def test_classify_error_null_pointer(self):
        """Test classifying null pointer error."""
        feedback = EnhancedFeedbackLoop()
        
        error_type = feedback._classify_error("NullPointerException at line 10")
        
        assert error_type == "NullPointerException"

    def test_classify_error_illegal_argument(self):
        """Test classifying illegal argument error."""
        feedback = EnhancedFeedbackLoop()
        
        error_type = feedback._classify_error("IllegalArgumentException: invalid input")
        
        assert error_type == "IllegalArgumentException"

    def test_classify_error_compilation(self):
        """Test classifying compilation error."""
        feedback = EnhancedFeedbackLoop()
        
        error_type = feedback._classify_error("cannot find symbol: class User")
        
        assert error_type == "CompilationError"

    def test_classify_error_unknown(self):
        """Test classifying unknown error."""
        feedback = EnhancedFeedbackLoop()
        
        error_type = feedback._classify_error("Some random error message")
        
        assert error_type == "UnknownError"

    def test_buffer_processing(self):
        """Test that buffer is processed when full."""
        feedback = EnhancedFeedbackLoop()
        
        for i in range(15):
            feedback.record_feedback(FeedbackType.TEST_PASS, {"index": i}, "success")
        
        assert len(feedback._event_buffer) < 15


class TestFeedbackEvent:
    """Tests for FeedbackEvent dataclass."""

    def test_event_creation(self):
        """Test event creation."""
        event = FeedbackEvent(
            event_id="event-123",
            feedback_type=FeedbackType.TEST_PASS,
            context={"test": "test1"},
            outcome="success"
        )
        
        assert event.event_id == "event-123"
        assert event.feedback_type == FeedbackType.TEST_PASS
        assert event.outcome == "success"

    def test_event_to_dict(self):
        """Test event to dictionary conversion."""
        event = FeedbackEvent(
            event_id="event-123",
            feedback_type=FeedbackType.TEST_FAILURE,
            context={"error": "assertion failed"},
            outcome="failed",
            details={"line": 10}
        )
        
        d = event.to_dict()
        
        assert d["event_id"] == "event-123"
        assert d["feedback_type"] == "test_failure"
        assert d["outcome"] == "failed"


class TestLearningInsight:
    """Tests for LearningInsight dataclass."""

    def test_insight_creation(self):
        """Test insight creation."""
        insight = LearningInsight(
            insight_id="insight-123",
            category=LearningCategory.ERROR_PATTERN,
            pattern="NullPointerException in service layer",
            conditions=["service_class", "no_null_check"],
            recommendation="Add null check before accessing dependencies",
            confidence=0.85,
            occurrence_count=5,
            success_rate=0.8
        )
        
        assert insight.insight_id == "insight-123"
        assert insight.category == LearningCategory.ERROR_PATTERN
        assert insight.confidence == 0.85

    def test_insight_to_dict(self):
        """Test insight to dictionary conversion."""
        insight = LearningInsight(
            insight_id="insight-123",
            category=LearningCategory.SUCCESSFUL_PATTERN,
            pattern="Mock setup pattern",
            conditions=["has_dependencies"],
            recommendation="Use @Mock and @InjectMocks",
            confidence=0.9
        )
        
        d = insight.to_dict()
        
        assert d["insight_id"] == "insight-123"
        assert d["category"] == "successful_pattern"


class TestAdaptiveAdjustment:
    """Tests for AdaptiveAdjustment dataclass."""

    def test_adjustment_creation(self):
        """Test adjustment creation."""
        adjustment = AdaptiveAdjustment(
            adjustment_id="adj-123",
            target="error_prevention",
            action="Add null check for user parameter",
            reason="Pattern: NullPointerException in 3 similar cases",
            priority=1,
            confidence=0.9
        )
        
        assert adjustment.adjustment_id == "adj-123"
        assert adjustment.target == "error_prevention"
        assert adjustment.priority == 1
        assert adjustment.applied is False


class TestFeedbackType:
    """Tests for FeedbackType enum."""

    def test_feedback_type_values(self):
        """Test feedback type enum values."""
        assert FeedbackType.COMPILATION_SUCCESS.value == "compilation_success"
        assert FeedbackType.COMPILATION_FAILURE.value == "compilation_failure"
        assert FeedbackType.TEST_PASS.value == "test_pass"
        assert FeedbackType.TEST_FAILURE.value == "test_failure"
        assert FeedbackType.COVERAGE_IMPROVEMENT.value == "coverage_improvement"


class TestLearningCategory:
    """Tests for LearningCategory enum."""

    def test_learning_category_values(self):
        """Test learning category enum values."""
        assert LearningCategory.ERROR_PATTERN.value == "error_pattern"
        assert LearningCategory.SUCCESSFUL_PATTERN.value == "successful_pattern"
        assert LearningCategory.OPTIMAL_STRATEGY.value == "optimal_strategy"
        assert LearningCategory.AVOIDED_APPROACH.value == "avoided_approach"


class TestFeedbackLoopIntegration:
    """Integration tests for feedback loop."""

    def test_full_feedback_cycle(self):
        """Test full feedback cycle."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test_cycle.db")
            feedback = EnhancedFeedbackLoop(db_path=db_path)
            
            feedback.record_compilation_result(False, [{"type": "SyntaxError"}], {})
            feedback.record_test_result("test1", False, "Assertion failed", {})
            feedback.record_test_result("test2", True, None, {})
            feedback.record_coverage_change(0.5, 0.7, {})
            
            stats = feedback.get_learning_stats()
            
            assert stats["total_events"] >= 4

    def test_learning_from_failures(self):
        """Test learning from failures."""
        feedback = EnhancedFeedbackLoop()
        
        for _ in range(3):
            feedback.record_feedback(
                feedback_type=FeedbackType.TEST_FAILURE,
                context={"error": "NullPointerException", "class": "UserService"},
                outcome="failed"
            )
        
        adjustments = feedback.get_adaptive_adjustments(
            {"error": "NullPointerException"}
        )
        
        assert isinstance(adjustments, list)

    def test_learning_from_successes(self):
        """Test learning from successes."""
        feedback = EnhancedFeedbackLoop()
        
        for _ in range(3):
            feedback.record_feedback(
                feedback_type=FeedbackType.TEST_PASS,
                context={"strategy": "mock_test", "class": "UserService"},
                outcome="success"
            )
        
        stats = feedback.get_learning_stats()
        
        assert stats["total_events"] >= 3
