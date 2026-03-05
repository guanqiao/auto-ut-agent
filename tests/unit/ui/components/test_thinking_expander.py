"""Tests for thinking expander component."""

import pytest
import time
from unittest.mock import MagicMock, patch

from pyutagent.ui.components.thinking_expander import (
    ThinkingStep,
    ThinkingStatus
)


class TestThinkingStatus:
    """Tests for ThinkingStatus enum."""
    
    def test_status_values(self):
        """Test status enum values."""
        assert ThinkingStatus.PENDING.value == "pending"
        assert ThinkingStatus.THINKING.value == "thinking"
        assert ThinkingStatus.COMPLETED.value == "completed"
        assert ThinkingStatus.ERROR.value == "error"


class TestThinkingStep:
    """Tests for ThinkingStep dataclass."""
    
    def test_step_creation(self):
        """Test creating a thinking step."""
        step = ThinkingStep(
            id="step1",
            title="Test Step",
            description="A test step",
            status=ThinkingStatus.PENDING
        )
        assert step.id == "step1"
        assert step.title == "Test Step"
        assert step.description == "A test step"
        assert step.status == ThinkingStatus.PENDING
        
    def test_step_start(self):
        """Test starting a step."""
        step = ThinkingStep(id="step1", title="Test")
        step.start()
        assert step.status == ThinkingStatus.THINKING
        assert step.start_time is not None
        
    def test_step_complete(self):
        """Test completing a step."""
        step = ThinkingStep(id="step1", title="Test")
        step.start()
        time.sleep(0.01)
        step.complete()
        assert step.status == ThinkingStatus.COMPLETED
        assert step.end_time is not None
        assert step.duration_ms > 0
        
    def test_step_fail(self):
        """Test failing a step."""
        step = ThinkingStep(id="step1", title="Test")
        step.fail()
        assert step.status == ThinkingStatus.ERROR
        assert step.end_time is not None
        
    def test_step_add_detail(self):
        """Test adding detail to a step."""
        step = ThinkingStep(id="step1", title="Test")
        step.add_detail("Detail 1")
        step.add_detail("Detail 2")
        assert len(step.details) == 2
        assert "Detail 1" in step.details
        assert "Detail 2" in step.details
        
    def test_duration_ms_not_started(self):
        """Test duration when step not started."""
        step = ThinkingStep(id="step1", title="Test")
        assert step.duration_ms == 0.0
        
    def test_duration_ms_running(self):
        """Test duration when step is running."""
        step = ThinkingStep(id="step1", title="Test")
        step.start()
        time.sleep(0.01)
        duration = step.duration_ms
        assert duration >= 10  # At least 10ms
