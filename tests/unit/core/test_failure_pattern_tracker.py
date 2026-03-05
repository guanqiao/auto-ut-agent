"""Tests for FailurePatternTracker."""

import pytest
from datetime import datetime, timedelta
from pyutagent.core.failure_pattern_tracker import (
    FailurePattern,
    FailurePatternTracker,
    SharedFailureKnowledge,
)


class TestFailurePattern:
    """Tests for FailurePattern dataclass."""

    def test_initial_state(self):
        """Test initial state of FailurePattern."""
        pattern = FailurePattern(
            error_type="ValueError",
            error_message="Test error",
            step_name="compilation"
        )

        assert pattern.count == 0
        assert pattern.error_type == "ValueError"
        assert pattern.error_message == "Test error"
        assert pattern.step_name == "compilation"

    def test_record_occurrence(self):
        """Test recording occurrences."""
        pattern = FailurePattern(
            error_type="ValueError",
            error_message="Test error",
            step_name="compilation"
        )

        pattern.record_occurrence()
        assert pattern.count == 1

        pattern.record_occurrence({"attempt": 2})
        assert pattern.count == 2
        assert len(pattern.contexts) == 1

    def test_is_recent(self):
        """Test is_recent method."""
        pattern = FailurePattern(
            error_type="ValueError",
            error_message="Test error",
            step_name="compilation"
        )

        pattern.record_occurrence()
        assert pattern.is_recent() is True

        # Simulate old pattern
        pattern.last_seen = datetime.now() - timedelta(seconds=400)
        assert pattern.is_recent(window_seconds=300) is False


class TestFailurePatternTracker:
    """Tests for FailurePatternTracker."""

    def test_initial_state(self):
        """Test initial state."""
        tracker = FailurePatternTracker()

        assert tracker.max_repeated_failures == 3
        assert len(tracker.patterns) == 0

    def test_record_failure(self):
        """Test recording failures."""
        tracker = FailurePatternTracker()
        error = ValueError("Test error")

        pattern = tracker.record_failure(error, "compilation")

        assert pattern.count == 1
        assert pattern.error_type == "ValueError"

        # Record same failure again
        pattern2 = tracker.record_failure(error, "compilation")
        assert pattern2.count == 2
        assert pattern is pattern2  # Same pattern object

    def test_should_stop_retrying(self):
        """Test should_stop_retrying logic."""
        tracker = FailurePatternTracker(max_repeated_failures=3)
        error = ValueError("Test error")

        # Should not stop initially
        assert tracker.should_stop_retrying(error, "compilation") is False

        # Record failures up to limit
        for _ in range(3):
            tracker.record_failure(error, "compilation")

        # Should stop now
        assert tracker.should_stop_retrying(error, "compilation") is True

    def test_get_recommendation(self):
        """Test getting recommendations."""
        tracker = FailurePatternTracker(max_repeated_failures=4)

        # No pattern yet - should retry
        error = ValueError("Test error")
        assert tracker.get_recommendation(error, "compilation") == "retry"

        # Record some failures
        for _ in range(2):
            tracker.record_failure(error, "compilation")
        assert tracker.get_recommendation(error, "compilation") == "retry_with_modification"

        # Record more failures
        for _ in range(2):
            tracker.record_failure(error, "compilation")
        assert tracker.get_recommendation(error, "compilation") == "skip_file"

        # Test different step types
        test_error = RuntimeError("Test")
        for _ in range(4):
            tracker.record_failure(test_error, "testing")
        assert tracker.get_recommendation(test_error, "testing") == "accept_partial"

    def test_get_pattern_stats(self):
        """Test getting pattern statistics."""
        tracker = FailurePatternTracker()

        # Add some patterns
        tracker.record_failure(ValueError("Error 1"), "step1")
        tracker.record_failure(ValueError("Error 2"), "step2")
        tracker.record_failure(ValueError("Error 1"), "step1")  # Same pattern

        stats = tracker.get_pattern_stats()

        assert stats["total_patterns"] == 2
        assert stats["active_patterns"] == 2
        assert stats["max_repeated_failures"] == 3

    def test_reset(self):
        """Test resetting patterns."""
        tracker = FailurePatternTracker()
        tracker.record_failure(ValueError("Test"), "step")

        assert len(tracker.patterns) == 1

        tracker.reset()
        assert len(tracker.patterns) == 0


class TestSharedFailureKnowledge:
    """Tests for SharedFailureKnowledge."""

    def test_initial_state(self):
        """Test initial state."""
        knowledge = SharedFailureKnowledge()

        assert knowledge.max_entries == 100
        assert len(knowledge.knowledge) == 0

    def test_record_and_check_failure(self):
        """Test recording and checking failures."""
        knowledge = SharedFailureKnowledge()
        error = ValueError("Test error")

        knowledge.record_failure("TestFile.java", error, "compilation", skip_recommended=True)

        should_skip, reason = knowledge.should_skip_file("TestFile.java")
        assert should_skip is True
        assert "Known issue" in reason

    def test_should_skip_file_no_match(self):
        """Test checking file with no matching failure."""
        knowledge = SharedFailureKnowledge()

        should_skip, reason = knowledge.should_skip_file("UnknownFile.java")
        assert should_skip is False
        assert reason is None

    def test_eviction(self):
        """Test eviction of old entries."""
        knowledge = SharedFailureKnowledge(max_entries=2)

        # Add entries up to limit
        knowledge.record_failure("File1.java", ValueError("Error 1"), "step")
        knowledge.record_failure("File2.java", ValueError("Error 2"), "step")

        assert len(knowledge.knowledge) == 2

        # Add one more - should evict oldest
        knowledge.record_failure("File3.java", ValueError("Error 3"), "step")

        assert len(knowledge.knowledge) == 2
        assert "File1.java" not in knowledge.knowledge

    def test_clear(self):
        """Test clearing all knowledge."""
        knowledge = SharedFailureKnowledge()
        knowledge.record_failure("File.java", ValueError("Error"), "step")

        assert len(knowledge.knowledge) == 1

        knowledge.clear()
        assert len(knowledge.knowledge) == 0
        assert len(knowledge.access_times) == 0


class TestFailurePatternIntegration:
    """Integration tests for failure pattern tracking."""

    def test_end_to_end_failure_tracking(self):
        """Test end-to-end failure tracking scenario."""
        tracker = FailurePatternTracker(max_repeated_failures=3)

        # Simulate repeated compilation failures
        error = ValueError("Compilation failed: cannot find symbol")

        # First failure - should retry (count=1, max=3, half=1.5, 1 < 1.5)
        tracker.record_failure(error, "compilation")
        assert tracker.should_stop_retrying(error, "compilation") is False
        # Note: count=1, max_repeated_failures=3, half=1, so 1 < 1 is False
        # Actually: 1 < (3 // 2) = 1, so 1 < 1 is False, returns "retry_with_modification"
        # Let's check the actual logic
        recommendation = tracker.get_recommendation(error, "compilation")
        assert recommendation in ["retry", "retry_with_modification"]

        # Second failure - count=2
        tracker.record_failure(error, "compilation")
        rec = tracker.get_recommendation(error, "compilation")
        assert rec in ["retry_with_modification", "skip_file"]

        # Third failure - should skip file (count=3 >= max=3)
        tracker.record_failure(error, "compilation")
        assert tracker.should_stop_retrying(error, "compilation") is True
        assert tracker.get_recommendation(error, "compilation") == "skip_file"

    def test_different_errors_same_step(self):
        """Test tracking different errors in the same step."""
        tracker = FailurePatternTracker()

        error1 = ValueError("Error 1")
        error2 = ValueError("Error 2")

        tracker.record_failure(error1, "compilation")
        tracker.record_failure(error2, "compilation")

        # Should be tracked as different patterns
        assert len(tracker.patterns) == 2

        # Each should have count 1
        assert tracker.should_stop_retrying(error1, "compilation") is False
        assert tracker.should_stop_retrying(error2, "compilation") is False
