"""Tests for thinking expander component."""

import pytest
import time
from unittest.mock import MagicMock, patch

# Skip Qt tests if not available
pytest.importorskip("PyQt6")

from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import Qt

from pyutagent.ui.components.thinking_expander import (
    ThinkingExpander,
    ThinkingStep,
    ThinkingStepWidget,
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


@pytest.mark.gui
class TestThinkingStepWidget:
    """Tests for ThinkingStepWidget class."""
    
    @pytest.fixture(scope="class")
    def qapp(self):
        """Create QApplication for tests."""
        app = QApplication.instance()
        if app is None:
            app = QApplication([])
        yield app
        
    def test_widget_creation(self, qapp):
        """Test widget creation."""
        step = ThinkingStep(id="step1", title="Test Step")
        widget = ThinkingStepWidget(step)
        assert widget is not None
        assert widget.get_step() == step
        
    def test_widget_update_display(self, qapp):
        """Test updating display."""
        step = ThinkingStep(id="step1", title="Test Step")
        widget = ThinkingStepWidget(step)
        
        # Start and complete the step
        step.start()
        step.complete()
        widget.update_display()
        
        # Widget should reflect the completed status
        assert step.status == ThinkingStatus.COMPLETED
        
    def test_widget_click_expands(self, qapp, qtbot):
        """Test clicking widget expands details."""
        step = ThinkingStep(id="step1", title="Test")
        step.add_detail("Test detail")
        widget = ThinkingStepWidget(step)
        qtbot.addWidget(widget)
        
        # Initially not expanded
        assert not widget._expanded
        
        # Click to expand
        widget._toggle_expand()
        assert widget._expanded
        
        # Click to collapse
        widget._toggle_expand()
        assert not widget._expanded
        
    def test_widget_click_signal(self, qapp, qtbot):
        """Test click signal emission."""
        step = ThinkingStep(id="step1", title="Test")
        widget = ThinkingStepWidget(step)
        qtbot.addWidget(widget)
        
        with qtbot.waitSignal(widget.clicked, timeout=1000) as blocker:
            widget._toggle_expand()
            
        assert blocker.args[0] == "step1"


@pytest.mark.gui
class TestThinkingExpander:
    """Tests for ThinkingExpander class."""
    
    @pytest.fixture(scope="class")
    def qapp(self):
        """Create QApplication for tests."""
        app = QApplication.instance()
        if app is None:
            app = QApplication([])
        yield app
        
    def test_expander_creation(self, qapp):
        """Test expander creation."""
        expander = ThinkingExpander()
        assert expander is not None
        assert not expander.is_expanded()
        
    def test_start_thinking(self, qapp):
        """Test starting thinking process."""
        expander = ThinkingExpander()
        expander.start_thinking()
        assert len(expander.get_steps()) == 0
        assert expander.is_expanded()  # Auto-expands on start
        
    def test_add_step(self, qapp):
        """Test adding a step."""
        expander = ThinkingExpander()
        expander.start_thinking()
        
        step = ThinkingStep(id="step1", title="Test Step")
        expander.add_step(step)
        
        assert len(expander.get_steps()) == 1
        assert expander.get_steps()[0].id == "step1"
        
    def test_start_step(self, qapp):
        """Test starting a step."""
        expander = ThinkingExpander()
        expander.start_thinking()
        
        step = ThinkingStep(id="step1", title="Test Step")
        expander.add_step(step)
        expander.start_step("step1")
        
        assert step.status == ThinkingStatus.THINKING
        
    def test_complete_step(self, qapp):
        """Test completing a step."""
        expander = ThinkingExpander()
        expander.start_thinking()
        
        step = ThinkingStep(id="step1", title="Test Step")
        expander.add_step(step)
        expander.start_step("step1")
        expander.complete_step("step1")
        
        assert step.status == ThinkingStatus.COMPLETED
        
    def test_fail_step(self, qapp):
        """Test failing a step."""
        expander = ThinkingExpander()
        expander.start_thinking()
        
        step = ThinkingStep(id="step1", title="Test Step")
        expander.add_step(step)
        expander.fail_step("step1")
        
        assert step.status == ThinkingStatus.ERROR
        
    def test_add_step_detail(self, qapp):
        """Test adding detail to a step."""
        expander = ThinkingExpander()
        expander.start_thinking()
        
        step = ThinkingStep(id="step1", title="Test Step")
        expander.add_step(step)
        expander.add_step_detail("step1", "Test detail")
        
        assert "Test detail" in step.details
        
    def test_finish_thinking(self, qapp):
        """Test finishing thinking process."""
        expander = ThinkingExpander()
        expander.start_thinking()
        time.sleep(0.01)
        expander.finish_thinking()
        
        assert expander.get_duration_ms() > 0
        
    def test_toggle_expanded(self, qapp):
        """Test toggling expanded state."""
        expander = ThinkingExpander()
        
        assert not expander.is_expanded()
        
        expander._toggle_expanded()
        assert expander.is_expanded()
        
        expander._toggle_expanded()
        assert not expander.is_expanded()
        
    def test_set_expanded(self, qapp):
        """Test setting expanded state."""
        expander = ThinkingExpander()
        
        expander.set_expanded(True)
        assert expander.is_expanded()
        
        expander.set_expanded(False)
        assert not expander.is_expanded()
        
    def test_clear(self, qapp):
        """Test clearing expander."""
        expander = ThinkingExpander()
        expander.start_thinking()
        
        step = ThinkingStep(id="step1", title="Test")
        expander.add_step(step)
        
        expander.clear()
        
        assert len(expander.get_steps()) == 0
        assert expander.get_duration_ms() == 0.0
        
    def test_multiple_steps(self, qapp):
        """Test handling multiple steps."""
        expander = ThinkingExpander()
        expander.start_thinking()
        
        steps = [
            ThinkingStep(id="1", title="Step 1"),
            ThinkingStep(id="2", title="Step 2"),
            ThinkingStep(id="3", title="Step 3"),
        ]
        
        for step in steps:
            expander.add_step(step)
            
        assert len(expander.get_steps()) == 3
        
        # Complete all steps
        for step in steps:
            expander.start_step(step.id)
            expander.complete_step(step.id)
            
        for step in steps:
            assert step.status == ThinkingStatus.COMPLETED
            
    def test_expanded_changed_signal(self, qapp, qtbot):
        """Test expanded changed signal."""
        expander = ThinkingExpander()
        
        with qtbot.waitSignal(expander.expanded_changed, timeout=1000) as blocker:
            expander._toggle_expanded()
            
        assert blocker.args[0] is True
        
    def test_step_clicked_signal(self, qapp, qtbot):
        """Test step clicked signal."""
        expander = ThinkingExpander()
        expander.start_thinking()
        
        step = ThinkingStep(id="step1", title="Test")
        expander.add_step(step)
        
        with qtbot.waitSignal(expander.step_clicked, timeout=1000) as blocker:
            expander._step_widgets["step1"]._toggle_expand()
            
        assert blocker.args[0] == "step1"


class TestThinkingExpanderEdgeCases:
    """Tests for edge cases in thinking expander."""
    
    def test_complete_nonexistent_step(self):
        """Test completing a step that doesn't exist."""
        expander = ThinkingExpander()
        expander.start_thinking()
        
        # Should not raise error
        expander.complete_step("nonexistent")
        
    def test_fail_nonexistent_step(self):
        """Test failing a step that doesn't exist."""
        expander = ThinkingExpander()
        expander.start_thinking()
        
        # Should not raise error
        expander.fail_step("nonexistent")
        
    def test_add_detail_to_nonexistent_step(self):
        """Test adding detail to a step that doesn't exist."""
        expander = ThinkingExpander()
        expander.start_thinking()
        
        # Should not raise error
        expander.add_step_detail("nonexistent", "detail")
        
    def test_duration_before_start(self):
        """Test getting duration before starting."""
        expander = ThinkingExpander()
        assert expander.get_duration_ms() == 0.0
        
    def test_empty_step_list(self):
        """Test with empty step list."""
        expander = ThinkingExpander()
        expander.start_thinking()
        
        steps = expander.get_steps()
        assert len(steps) == 0
        assert isinstance(steps, list)
