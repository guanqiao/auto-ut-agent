"""Tests for agent_worker module."""

import pytest
import time
from unittest.mock import Mock, MagicMock, patch
from PyQt6.QtCore import QObject

from pyutagent.ui.agent_panel.agent_worker import (
    AgentWorker, AgentState, ToolCallStatus, ToolCallInfo,
    ThinkingStepInfo, AgentProgress, AgentError, AgentStateSignals,
    create_agent_worker
)


class TestAgentState:
    """Test AgentState enum."""

    def test_agent_state_values(self):
        """Test AgentState has all expected values."""
        assert AgentState.IDLE is not None
        assert AgentState.STARTING is not None
        assert AgentState.OBSERVING is not None
        assert AgentState.THINKING is not None
        assert AgentState.ACTING is not None
        assert AgentState.VERIFYING is not None
        assert AgentState.LEARNING is not None
        assert AgentState.COMPLETED is not None
        assert AgentState.FAILED is not None
        assert AgentState.PAUSED is not None


class TestToolCallStatus:
    """Test ToolCallStatus enum."""

    def test_tool_call_status_values(self):
        """Test ToolCallStatus has all expected values."""
        assert ToolCallStatus.PENDING is not None
        assert ToolCallStatus.RUNNING is not None
        assert ToolCallStatus.SUCCESS is not None
        assert ToolCallStatus.FAILED is not None
        assert ToolCallStatus.SKIPPED is not None


class TestToolCallInfo:
    """Test ToolCallInfo dataclass."""

    def test_tool_call_info_creation(self):
        """Test ToolCallInfo can be created."""
        info = ToolCallInfo(
            id="tool_1",
            tool_name="test_tool",
            parameters={"param1": "value1"},
            status=ToolCallStatus.RUNNING
        )
        assert info.id == "tool_1"
        assert info.tool_name == "test_tool"
        assert info.parameters == {"param1": "value1"}
        assert info.status == ToolCallStatus.RUNNING

    def test_tool_call_info_duration(self):
        """Test duration calculation."""
        start = time.time()
        end = start + 1.5

        info = ToolCallInfo(
            id="tool_1",
            tool_name="test_tool",
            start_time=start,
            end_time=end
        )
        assert info.duration == pytest.approx(1.5, 0.01)

    def test_tool_call_info_duration_none(self):
        """Test duration returns None when times not set."""
        info = ToolCallInfo(id="tool_1", tool_name="test_tool")
        assert info.duration is None


class TestThinkingStepInfo:
    """Test ThinkingStepInfo dataclass."""

    def test_thinking_step_info_creation(self):
        """Test ThinkingStepInfo can be created."""
        info = ThinkingStepInfo(
            id="step_1",
            title="Test Step",
            description="A test step",
            status="pending"
        )
        assert info.id == "step_1"
        assert info.title == "Test Step"
        assert info.description == "A test step"
        assert info.status == "pending"
        assert info.details == []


class TestAgentProgress:
    """Test AgentProgress dataclass."""

    def test_agent_progress_creation(self):
        """Test AgentProgress can be created."""
        progress = AgentProgress(
            current_step=5,
            total_steps=10,
            current_state=AgentState.ACTING,
            task_name="Test Task",
            progress_percent=50.0,
            message="Halfway done"
        )
        assert progress.current_step == 5
        assert progress.total_steps == 10
        assert progress.current_state == AgentState.ACTING
        assert progress.task_name == "Test Task"
        assert progress.progress_percent == 50.0
        assert progress.message == "Halfway done"


class TestAgentError:
    """Test AgentError dataclass."""

    def test_agent_error_creation(self):
        """Test AgentError can be created."""
        error = AgentError(
            step_id="step_1",
            error_message="Something went wrong",
            error_type="RuntimeError",
            retryable=True,
            context={"key": "value"}
        )
        assert error.step_id == "step_1"
        assert error.error_message == "Something went wrong"
        assert error.error_type == "RuntimeError"
        assert error.retryable is True
        assert error.context == {"key": "value"}


class TestAgentStateSignals:
    """Test AgentStateSignals."""

    def test_signals_exist(self):
        """Test all expected signals exist."""
        signals = AgentStateSignals()

        assert hasattr(signals, 'state_changed')
        assert hasattr(signals, 'progress_updated')
        assert hasattr(signals, 'thinking_step_added')
        assert hasattr(signals, 'thinking_step_updated')
        assert hasattr(signals, 'tool_call_started')
        assert hasattr(signals, 'tool_call_completed')
        assert hasattr(signals, 'tool_call_failed')
        assert hasattr(signals, 'error_occurred')
        assert hasattr(signals, 'raw_output')
        assert hasattr(signals, 'task_completed')
        assert hasattr(signals, 'task_failed')
        assert hasattr(signals, 'learning_recorded')


class TestAgentWorker:
    """Test AgentWorker class."""

    @pytest.fixture
    def mock_autonomous_loop(self):
        """Create mock autonomous loop."""
        return Mock()

    @pytest.fixture
    def agent_worker(self, mock_autonomous_loop, qtbot):
        """Create AgentWorker instance."""
        worker = AgentWorker(mock_autonomous_loop)
        return worker

    def test_agent_worker_creation(self, agent_worker):
        """Test AgentWorker can be created."""
        assert agent_worker is not None
        assert agent_worker.signals is not None

    def test_run_task_initialization(self, agent_worker):
        """Test run_task initializes state correctly."""
        agent_worker.run_task("Test Task", {"key": "value"})

        assert agent_worker._current_task == "Test Task"
        assert agent_worker._context == {"key": "value"}
        assert agent_worker._running is True
        assert agent_worker._paused is False

    def test_pause(self, agent_worker):
        """Test pause method."""
        agent_worker._paused = False
        agent_worker.pause()
        assert agent_worker._paused is True

    def test_resume(self, agent_worker):
        """Test resume method."""
        agent_worker._paused = True
        agent_worker.resume()
        assert agent_worker._paused is False

    def test_stop(self, agent_worker):
        """Test stop method."""
        agent_worker._running = True
        agent_worker.stop()
        assert agent_worker._running is False

    def test_skip_step(self, agent_worker):
        """Test skip_step method."""
        agent_worker._thinking_steps["step_1"] = ThinkingStepInfo(
            id="step_1", title="Test Step"
        )

        with patch.object(agent_worker.signals, 'thinking_step_updated') as mock_signal:
            agent_worker.skip_step("step_1")
            assert agent_worker._thinking_steps["step_1"].status == "skipped"
            mock_signal.emit.assert_called_once_with("step_1", "skipped", None)

    def test_add_thinking_step(self, agent_worker):
        """Test _add_thinking_step method."""
        with patch.object(agent_worker.signals, 'thinking_step_added') as mock_signal:
            step_id = agent_worker._add_thinking_step("Test Title", "Test Description")

            assert step_id == "step_1"
            assert step_id in agent_worker._thinking_steps
            assert agent_worker._thinking_steps[step_id].title == "Test Title"
            mock_signal.emit.assert_called_once()

    def test_start_tool_call(self, agent_worker):
        """Test _start_tool_call method."""
        with patch.object(agent_worker.signals, 'tool_call_started') as mock_signal:
            tool_id = agent_worker._start_tool_call("test_tool", {"param": "value"})

            assert tool_id == "tool_1"
            assert tool_id in agent_worker._tool_calls
            assert agent_worker._tool_calls[tool_id].tool_name == "test_tool"
            mock_signal.emit.assert_called_once()

    def test_complete_tool_call(self, agent_worker):
        """Test _complete_tool_call method."""
        agent_worker._tool_calls["tool_1"] = ToolCallInfo(
            id="tool_1",
            tool_name="test_tool",
            status=ToolCallStatus.RUNNING,
            start_time=time.time()
        )

        with patch.object(agent_worker.signals, 'tool_call_completed') as mock_signal:
            agent_worker._complete_tool_call("tool_1", {"result": "success"})

            assert agent_worker._tool_calls["tool_1"].status == ToolCallStatus.SUCCESS
            assert agent_worker._tool_calls["tool_1"].result == {"result": "success"}
            mock_signal.emit.assert_called_once()

    def test_fail_tool_call(self, agent_worker):
        """Test _fail_tool_call method."""
        agent_worker._tool_calls["tool_1"] = ToolCallInfo(
            id="tool_1",
            tool_name="test_tool",
            status=ToolCallStatus.RUNNING,
            start_time=time.time()
        )

        with patch.object(agent_worker.signals, 'tool_call_failed') as mock_signal:
            agent_worker._fail_tool_call("tool_1", "Error message")

            assert agent_worker._tool_calls["tool_1"].status == ToolCallStatus.FAILED
            assert agent_worker._tool_calls["tool_1"].error == "Error message"
            mock_signal.emit.assert_called_once()

    def test_handle_progress(self, agent_worker):
        """Test _handle_progress method."""
        agent_worker._current_task = "Test Task"

        with patch.object(agent_worker.signals, 'progress_updated') as mock_progress:
            with patch.object(agent_worker.signals, 'state_changed') as mock_state:
                with patch.object(agent_worker.signals, 'raw_output') as mock_output:
                    agent_worker._handle_progress({
                        "iteration": 5,
                        "max_iterations": 10,
                        "state": "ACTING"
                    })

                    mock_progress.emit.assert_called_once()
                    mock_state.emit.assert_called_once()
                    mock_output.emit.assert_called_once()


class TestCreateAgentWorker:
    """Test create_agent_worker function."""

    def test_create_agent_worker(self):
        """Test create_agent_worker creates correct instance."""
        mock_loop = Mock()
        worker = create_agent_worker(mock_loop)

        assert isinstance(worker, AgentWorker)
        assert worker.autonomous_loop == mock_loop
