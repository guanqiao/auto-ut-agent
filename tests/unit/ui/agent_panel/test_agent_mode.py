"""Tests for agent_mode module."""

import pytest
from unittest.mock import Mock, patch, MagicMock

from PyQt6.QtWidgets import QWidget
from PyQt6.QtCore import Qt

from pyutagent.ui.agent_panel.agent_mode import (
    AgentMode, ToolCallWidget
)
from pyutagent.ui.agent_panel.thinking_chain import ThinkingStep, StepStatus
from pyutagent.ui.agent_panel.agent_worker import (
    AgentWorker, AgentState, ToolCallInfo, ThinkingStepInfo, AgentProgress
)


class TestToolCallWidget:
    """Test ToolCallWidget class."""

    @pytest.fixture
    def tool_widget(self, qtbot):
        """Create ToolCallWidget instance."""
        widget = ToolCallWidget(
            tool_name="test_tool",
            params={"param1": "value1"},
            status="pending"
        )
        qtbot.addWidget(widget)
        return widget

    def test_tool_widget_creation(self, tool_widget):
        """Test ToolCallWidget can be created."""
        assert tool_widget is not None
        assert tool_widget._tool_name == "test_tool"
        assert tool_widget._status == "pending"

    def test_get_tool_icon(self, tool_widget):
        """Test _get_tool_icon method."""
        # Test file tool
        tool_widget._tool_name = "read_file"
        assert tool_widget._get_tool_icon() == "📄"

        # Test terminal tool
        tool_widget._tool_name = "bash_command"
        assert tool_widget._get_tool_icon() == "⌨️"

        # Test search tool
        tool_widget._tool_name = "search_code"
        assert tool_widget._get_tool_icon() == "🔍"

        # Test git tool
        tool_widget._tool_name = "git_status"
        assert tool_widget._get_tool_icon() == "🌿"

        # Test default
        tool_widget._tool_name = "unknown_tool"
        assert tool_widget._get_tool_icon() == "🔧"

    def test_update_status(self, tool_widget):
        """Test update_status method."""
        tool_widget.update_status("running")
        assert tool_widget._status == "running"

    def test_update_result(self, tool_widget):
        """Test update_result method."""
        result = {"status": "success", "data": "test"}
        tool_widget.update_result(result)
        assert tool_widget._result == result

    def test_update_duration(self, tool_widget):
        """Test update_duration method."""
        tool_widget.update_duration(1.5)
        assert tool_widget._duration == 1.5

    def test_set_expanded(self, tool_widget):
        """Test set_expanded method."""
        tool_widget.set_expanded(True)
        assert tool_widget._expanded is True
        assert tool_widget._details_container.isVisible() is True

    def test_toggle_expand(self, tool_widget):
        """Test _toggle_expand method."""
        tool_widget._expanded = False
        tool_widget._toggle_expand()
        assert tool_widget._expanded is True


class TestAgentMode:
    """Test AgentMode class."""

    @pytest.fixture
    def agent_mode(self, qtbot):
        """Create AgentMode instance."""
        mode = AgentMode()
        qtbot.addWidget(mode)
        return mode

    @pytest.fixture
    def mock_agent_worker(self):
        """Create mock AgentWorker."""
        worker = Mock(spec=AgentWorker)
        worker.signals = Mock()
        worker.signals.state_changed = Mock()
        worker.signals.progress_updated = Mock()
        worker.signals.thinking_step_added = Mock()
        worker.signals.thinking_step_updated = Mock()
        worker.signals.tool_call_started = Mock()
        worker.signals.tool_call_completed = Mock()
        worker.signals.tool_call_failed = Mock()
        worker.signals.error_occurred = Mock()
        worker.signals.raw_output = Mock()
        worker.signals.task_completed = Mock()
        worker.signals.task_failed = Mock()
        worker.signals.learning_recorded = Mock()
        return worker

    def test_agent_mode_creation(self, agent_mode):
        """Test AgentMode can be created."""
        assert agent_mode is not None
        assert agent_mode._current_task is None

    def test_set_agent_worker(self, agent_mode, mock_agent_worker):
        """Test set_agent_worker method."""
        agent_mode.set_agent_worker(mock_agent_worker)

        assert agent_mode._agent_worker == mock_agent_worker
        mock_agent_worker.signals.state_changed.connect.assert_called_once()
        mock_agent_worker.signals.progress_updated.connect.assert_called_once()

    def test_start_task(self, agent_mode):
        """Test start_task method."""
        agent_mode.start_task("Test Task")

        assert agent_mode._current_task == "Test Task"
        assert agent_mode._progress_tracker.isVisible() is True

    def test_add_thinking_step(self, agent_mode):
        """Test add_thinking_step method."""
        step = ThinkingStep(id="step_1", title="Test Step")
        agent_mode.add_thinking_step(step)

        assert agent_mode._thinking_chain.get_step_count() == 1

    def test_start_step(self, agent_mode):
        """Test start_step method."""
        step = ThinkingStep(id="step_1", title="Test Step")
        agent_mode.add_thinking_step(step)
        agent_mode.start_step("step_1")

        # Step should be in running state
        widget = agent_mode._thinking_chain._steps.get("step_1")
        assert widget is not None

    def test_complete_step(self, agent_mode):
        """Test complete_step method."""
        step = ThinkingStep(id="step_1", title="Test Step")
        agent_mode.add_thinking_step(step)
        agent_mode.start_step("step_1")
        agent_mode.complete_step("step_1")

        widget = agent_mode._thinking_chain._steps.get("step_1")
        assert widget._step.status == StepStatus.COMPLETED

    def test_fail_step(self, agent_mode):
        """Test fail_step method."""
        step = ThinkingStep(id="step_1", title="Test Step")
        agent_mode.add_thinking_step(step)
        agent_mode.fail_step("step_1")

        widget = agent_mode._thinking_chain._steps.get("step_1")
        assert widget._step.status == StepStatus.FAILED

    def test_add_step_detail(self, agent_mode):
        """Test add_step_detail method."""
        step = ThinkingStep(id="step_1", title="Test Step")
        agent_mode.add_thinking_step(step)
        agent_mode.add_step_detail("step_1", "Additional detail")

        widget = agent_mode._thinking_chain._steps.get("step_1")
        assert "Additional detail" in widget._step.details

    def test_add_tool_call(self, agent_mode):
        """Test add_tool_call method."""
        agent_mode.add_tool_call(
            tool_name="test_tool",
            params={"param1": "value1"},
            result={"status": "success"}
        )

        assert len(agent_mode._tool_calls) == 1

    def test_update_tool_call_status(self, agent_mode):
        """Test update_tool_call_status method."""
        agent_mode.add_tool_call("test_tool", {})
        tool_id = list(agent_mode._tool_calls.keys())[0]

        agent_mode.update_tool_call_status(tool_id, "running")
        assert agent_mode._tool_calls[tool_id]._status == "running"

    def test_update_tool_call_result(self, agent_mode):
        """Test update_tool_call_result method."""
        agent_mode.add_tool_call("test_tool", {})
        tool_id = list(agent_mode._tool_calls.keys())[0]

        result = {"data": "test"}
        agent_mode.update_tool_call_result(tool_id, result)
        assert agent_mode._tool_calls[tool_id]._result == result

    def test_append_raw_output(self, agent_mode):
        """Test append_raw_output method."""
        agent_mode.append_raw_output("Test output")
        assert "Test output" in agent_mode._raw_output.toPlainText()

    def test_set_progress(self, agent_mode):
        """Test set_progress method."""
        agent_mode.set_progress(5, 10, "Halfway")

        assert agent_mode._progress_tracker.isVisible() is True
        assert agent_mode._progress_tracker._progress_bar.value() == 50

    def test_add_error(self, agent_mode):
        """Test add_error method."""
        agent_mode.add_error(
            error_id="err_1",
            error_message="Test error",
            error_type="RuntimeError"
        )

        assert agent_mode._error_list.isVisible() is True
        assert agent_mode._error_list.get_error_count() == 1

    def test_clear(self, agent_mode):
        """Test clear method."""
        agent_mode.start_task("Test Task")
        agent_mode.add_tool_call("test_tool", {})
        agent_mode.add_error("err_1", "Test error")

        agent_mode.clear()

        assert agent_mode._current_task is None
        assert len(agent_mode._tool_calls) == 0
        assert agent_mode._error_list.get_error_count() == 0

    def test_set_status(self, agent_mode):
        """Test set_status method."""
        agent_mode.set_status("Processing", "info")
        assert "Processing" in agent_mode._status_label.text()

    def test_show_thinking_tab(self, agent_mode):
        """Test show_thinking_tab method."""
        agent_mode.show_thinking_tab()
        assert agent_mode._tabs.currentIndex() == 0

    def test_show_tools_tab(self, agent_mode):
        """Test show_tools_tab method."""
        agent_mode.show_tools_tab()
        assert agent_mode._tabs.currentIndex() == 1

    def test_show_output_tab(self, agent_mode):
        """Test show_output_tab method."""
        agent_mode.show_output_tab()
        assert agent_mode._tabs.currentIndex() == 2


class TestAgentModeSignals:
    """Test AgentMode signal handlers."""

    @pytest.fixture
    def agent_mode(self, qtbot):
        """Create AgentMode instance."""
        mode = AgentMode()
        qtbot.addWidget(mode)
        return mode

    def test_on_state_changed(self, agent_mode):
        """Test _on_state_changed handler."""
        agent_mode._on_state_changed(AgentState.ACTING, "Working...")
        assert "Working..." in agent_mode._status_label.text()

    def test_on_progress_updated(self, agent_mode):
        """Test _on_progress_updated handler."""
        progress = AgentProgress(
            current_step=5,
            total_steps=10,
            current_state=AgentState.ACTING,
            task_name="Test",
            progress_percent=50.0,
            message="Halfway"
        )
        agent_mode._on_progress_updated(progress)

        assert agent_mode._progress_tracker.isVisible() is True

    def test_on_thinking_step_added(self, agent_mode):
        """Test _on_thinking_step_added handler."""
        step_info = ThinkingStepInfo(
            id="step_1",
            title="Test Step",
            description="A test step"
        )
        agent_mode._on_thinking_step_added(step_info)

        assert agent_mode._thinking_chain.get_step_count() == 1

    def test_on_tool_call_started(self, agent_mode):
        """Test _on_tool_call_started handler."""
        tool_info = ToolCallInfo(
            id="tool_1",
            tool_name="test_tool",
            parameters={},
            status=Mock()  # ToolCallStatus.RUNNING
        )
        agent_mode._on_tool_call_started(tool_info)

        assert len(agent_mode._tool_calls) == 1
        assert "tool_1" in agent_mode._tool_calls

    def test_on_tool_call_completed(self, agent_mode):
        """Test _on_tool_call_completed handler."""
        # First add a tool call
        tool_info = ToolCallInfo(
            id="tool_1",
            tool_name="test_tool",
            parameters={}
        )
        agent_mode._on_tool_call_started(tool_info)

        # Complete it
        tool_info.result = {"status": "success"}
        tool_info.status = Mock()  # ToolCallStatus.SUCCESS
        tool_info.duration = 1.5
        agent_mode._on_tool_call_completed(tool_info)

        assert agent_mode._tool_calls["tool_1"]._status == "success"

    def test_on_task_completed(self, agent_mode):
        """Test _on_task_completed handler."""
        agent_mode._on_task_completed("Test Task", {"result": "success"})

        assert "Complete" in agent_mode._status_label.text()

    def test_on_task_failed(self, agent_mode):
        """Test _on_task_failed handler."""
        agent_mode._on_task_failed("Test Task", "Error message")

        assert "Failed" in agent_mode._status_label.text()

    def test_on_raw_output(self, agent_mode):
        """Test _on_raw_output handler."""
        agent_mode._on_raw_output("Test output\n")
        assert "Test output" in agent_mode._raw_output.toPlainText()

    def test_on_learning_recorded(self, agent_mode):
        """Test _on_learning_recorded handler."""
        agent_mode._on_learning_recorded("Learned something")
        assert "Learned something" in agent_mode._raw_output.toPlainText()
