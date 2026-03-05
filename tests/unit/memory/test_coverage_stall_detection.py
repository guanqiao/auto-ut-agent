"""Tests for coverage stall detection in WorkingMemory."""

import pytest
from pyutagent.memory.working_memory import WorkingMemory


class TestCoverageStallDetection:
    """Tests for coverage stall detection functionality."""

    def test_has_coverage_stalled_insufficient_history(self):
        """Test stall detection with insufficient history."""
        memory = WorkingMemory()

        # No history yet
        assert memory.has_coverage_stalled() is False

        # Only 2 entries
        memory.update_coverage(0.5)
        memory.update_coverage(0.5)
        assert memory.has_coverage_stalled() is False

    def test_has_coverage_stalled_stable_coverage(self):
        """Test detecting stalled coverage."""
        memory = WorkingMemory()

        # Add stable coverage history
        memory.update_coverage(0.5)
        memory.update_coverage(0.5)
        memory.update_coverage(0.5)

        assert memory.has_coverage_stalled() is True

    def test_has_coverage_stalled_with_improvement(self):
        """Test that improvement prevents stall detection."""
        memory = WorkingMemory()

        # Add improving coverage history
        memory.update_coverage(0.5)
        memory.update_coverage(0.6)
        memory.update_coverage(0.7)

        assert memory.has_coverage_stalled() is False

    def test_has_coverage_stalled_custom_threshold(self):
        """Test stall detection with custom threshold."""
        memory = WorkingMemory()

        # Small improvements - 0.50 -> 0.51 -> 0.52, diff = 0.02
        memory.update_coverage(0.50)
        memory.update_coverage(0.51)
        memory.update_coverage(0.52)

        # Default threshold (0.01) should NOT detect stall (0.52 - 0.50 = 0.02 >= 0.01)
        assert memory.has_coverage_stalled() is False

        # Higher threshold (0.03) - 0.02 < 0.03, should detect stall
        assert memory.has_coverage_stalled(threshold=0.03) is True

        # Test with truly stalled coverage
        memory2 = WorkingMemory()
        memory2.update_coverage(0.50)
        memory2.update_coverage(0.501)
        memory2.update_coverage(0.502)
        # diff = 0.002 < 0.01, should detect stall with default threshold
        assert memory2.has_coverage_stalled() is True

    def test_has_coverage_stalled_custom_window(self):
        """Test stall detection with custom window size."""
        memory = WorkingMemory()

        # Add more history
        for coverage in [0.5, 0.5, 0.5, 0.6, 0.7]:
            memory.update_coverage(coverage)

        # Window of 3 should not detect stall (0.5, 0.6, 0.7)
        assert memory.has_coverage_stalled(window_size=3) is False

        # Window of 5 should detect stall (includes earlier 0.5s)
        # Actually max(0.5, 0.6, 0.7) - min(0.5, 0.6, 0.7) = 0.2 > 0.01
        # So it should not detect stall
        assert memory.has_coverage_stalled(window_size=5) is False


class TestCoverageTrend:
    """Tests for coverage trend analysis."""

    def test_get_coverage_trend_insufficient_data(self):
        """Test trend with insufficient data."""
        memory = WorkingMemory()

        trend = memory.get_coverage_trend()
        assert trend["trend"] == "insufficient_data"

        # Only one entry
        memory.update_coverage(0.5)
        trend = memory.get_coverage_trend()
        assert trend["trend"] == "insufficient_data"

    def test_get_coverage_trend_improving(self):
        """Test detecting improving trend."""
        memory = WorkingMemory()

        memory.update_coverage(0.5)
        memory.update_coverage(0.6)

        trend = memory.get_coverage_trend()
        assert trend["trend"] == "improving"
        assert abs(trend["improvement"] - 0.1) < 0.001  # Use approximate comparison for float

    def test_get_coverage_trend_slight_improvement(self):
        """Test detecting slight improvement."""
        memory = WorkingMemory()

        memory.update_coverage(0.50)
        memory.update_coverage(0.52)

        trend = memory.get_coverage_trend()
        assert trend["trend"] == "slight_improvement"

    def test_get_coverage_trend_stable(self):
        """Test detecting stable trend."""
        memory = WorkingMemory()

        memory.update_coverage(0.50)
        memory.update_coverage(0.505)

        trend = memory.get_coverage_trend()
        assert trend["trend"] == "stable"

    def test_get_coverage_trend_declining(self):
        """Test detecting declining trend."""
        memory = WorkingMemory()

        memory.update_coverage(0.6)
        memory.update_coverage(0.5)

        trend = memory.get_coverage_trend()
        assert trend["trend"] == "declining"
        assert abs(trend["improvement"] - (-0.1)) < 0.001  # Use approximate comparison for float

    def test_get_coverage_trend_with_window(self):
        """Test trend with custom window size."""
        memory = WorkingMemory()

        # Add history
        for coverage in [0.3, 0.4, 0.5, 0.6, 0.7]:
            memory.update_coverage(coverage)

        # Full window - should be improving
        trend = memory.get_coverage_trend(window_size=5)
        assert trend["trend"] == "improving"
        assert trend["start_coverage"] == 0.3
        assert trend["current_coverage"] == 0.7

        # Smaller window - still improving
        trend = memory.get_coverage_trend(window_size=2)
        assert trend["trend"] == "improving"


class TestCoverageHistory:
    """Tests for coverage history tracking."""

    def test_update_coverage_adds_to_history(self):
        """Test that update_coverage adds entries to history."""
        memory = WorkingMemory()

        memory.update_coverage(0.5, "jacoco", 1.0)

        assert len(memory.coverage_history) == 1
        assert memory.coverage_history[0]["coverage"] == 0.5
        assert memory.coverage_history[0]["source"] == "jacoco"
        assert memory.coverage_history[0]["confidence"] == 1.0
        assert "timestamp" in memory.coverage_history[0]

    def test_current_coverage_updated(self):
        """Test that current_coverage is updated."""
        memory = WorkingMemory()

        memory.update_coverage(0.5)
        assert memory.current_coverage == 0.5

        memory.update_coverage(0.7)
        assert memory.current_coverage == 0.7

    def test_coverage_source_and_confidence(self):
        """Test coverage source and confidence tracking."""
        memory = WorkingMemory()

        memory.update_coverage(0.5, "jacoco", 1.0)
        assert memory.coverage_source == "jacoco"
        assert memory.coverage_confidence == 1.0

        memory.update_coverage(0.6, "llm_estimated", 0.8)
        assert memory.coverage_source == "llm_estimated"
        assert memory.coverage_confidence == 0.8
