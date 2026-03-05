"""Agent mode for Agent panel - shows thinking chain and tool calls."""

import logging
from typing import Optional, List, Dict, Any

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QTabWidget, QTextEdit, QFrame, QScrollArea
)
from PyQt6.QtCore import Qt, pyqtSignal

from .thinking_chain import ThinkingChainWidget, ThinkingStep, StepStatus
from ..components.progress_tracker import ProgressTracker
from ..components.error_display import ErrorListWidget
from .agent_worker import (
    AgentWorker, AgentState, ToolCallInfo, ThinkingStepInfo, AgentProgress
)

logger = logging.getLogger(__name__)


class ToolCallWidget(QFrame):
    """Widget for displaying a tool call with real-time status updates."""

    # 状态图标
    STATUS_ICONS = {
        "pending": "⏳",
        "running": "🔄",
        "success": "✓",
        "failed": "✗",
        "skipped": "⊘"
    }

    # 状态颜色
    STATUS_COLORS = {
        "pending": "#757575",
        "running": "#2196F3",
        "success": "#4CAF50",
        "failed": "#F44336",
        "skipped": "#9E9E9E"
    }

    def __init__(self, tool_name: str, params: Dict[str, Any],
                 result: Optional[Any] = None, status: str = "pending",
                 parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._tool_name = tool_name
        self._params = params
        self._result = result
        self._status = status
        self._expanded = False
        self._duration: Optional[float] = None

        self.setup_ui()
        self.update_display()

    def setup_ui(self):
        """Setup the tool call widget UI."""
        self.setFrameShape(QFrame.Shape.StyledPanel)
        self.setStyleSheet("""
            ToolCallWidget {
                background-color: #F5F5F5;
                border: 1px solid #E0E0E0;
                border-radius: 6px;
            }
        """)

        self._main_layout = QVBoxLayout(self)
        self._main_layout.setContentsMargins(12, 8, 12, 8)
        self._main_layout.setSpacing(8)

        # Header
        header = QHBoxLayout()

        # Status icon
        self._status_icon = QLabel()
        self._status_icon.setFixedWidth(24)
        header.addWidget(self._status_icon)

        # Tool icon and name
        tool_icon = self._get_tool_icon()
        self._name_label = QLabel()
        self._name_label.setStyleSheet("font-weight: bold; color: #333;")
        header.addWidget(self._name_label)

        # Duration
        self._duration_label = QLabel()
        self._duration_label.setStyleSheet("color: #999; font-size: 11px;")
        header.addWidget(self._duration_label)

        header.addStretch()

        # Expand button
        self._expand_btn = QPushButton("▶")
        self._expand_btn.setFixedSize(24, 24)
        self._expand_btn.setFlat(True)
        self._expand_btn.setStyleSheet("color: #666;")
        self._expand_btn.clicked.connect(self._toggle_expand)
        header.addWidget(self._expand_btn)

        self._main_layout.addLayout(header)

        # Details container (expandable)
        self._details_container = QWidget()
        details_layout = QVBoxLayout(self._details_container)
        details_layout.setContentsMargins(28, 0, 0, 0)
        details_layout.setSpacing(8)

        # Parameters section
        params_header = QLabel("📋 Parameters:")
        params_header.setStyleSheet("font-weight: bold; color: #666; font-size: 11px;")
        details_layout.addWidget(params_header)

        self._params_text = QTextEdit()
        self._params_text.setReadOnly(True)
        self._params_text.setMaximumHeight(80)
        self._params_text.setStyleSheet("""
            QTextEdit {
                background-color: #FFFFFF;
                border: 1px solid #E0E0E0;
                border-radius: 4px;
                color: #333;
                font-family: 'Consolas', 'Monaco', monospace;
                font-size: 11px;
                padding: 6px;
            }
        """)
        details_layout.addWidget(self._params_text)

        # Result section
        self._result_header = QLabel("📤 Result:")
        self._result_header.setStyleSheet("font-weight: bold; color: #666; font-size: 11px;")
        details_layout.addWidget(self._result_header)

        self._result_text = QTextEdit()
        self._result_text.setReadOnly(True)
        self._result_text.setMaximumHeight(100)
        self._result_text.setStyleSheet("""
            QTextEdit {
                background-color: #E8F5E9;
                border: 1px solid #C8E6C9;
                border-radius: 4px;
                color: #2E7D32;
                font-family: 'Consolas', 'Monaco', monospace;
                font-size: 11px;
                padding: 6px;
            }
        """)
        details_layout.addWidget(self._result_text)

        # Error section
        self._error_header = QLabel("❌ Error:")
        self._error_header.setStyleSheet("font-weight: bold; color: #C62828; font-size: 11px;")
        details_layout.addWidget(self._error_header)

        self._error_text = QTextEdit()
        self._error_text.setReadOnly(True)
        self._error_text.setMaximumHeight(80)
        self._error_text.setStyleSheet("""
            QTextEdit {
                background-color: #FFEBEE;
                border: 1px solid #FFCDD2;
                border-radius: 4px;
                color: #C62828;
                font-family: 'Consolas', 'Monaco', monospace;
                font-size: 11px;
                padding: 6px;
            }
        """)
        details_layout.addWidget(self._error_text)

        self._details_container.hide()
        self._main_layout.addWidget(self._details_container)

    def _get_tool_icon(self) -> str:
        """Get icon for tool name."""
        tool_lower = self._tool_name.lower()
        if "file" in tool_lower:
            return "📄"
        elif "terminal" in tool_lower or "command" in tool_lower or "bash" in tool_lower:
            return "⌨️"
        elif "search" in tool_lower:
            return "🔍"
        elif "edit" in tool_lower:
            return "✏️"
        elif "git" in tool_lower:
            return "🌿"
        elif "test" in tool_lower:
            return "🧪"
        elif "build" in tool_lower or "maven" in tool_lower:
            return "🔨"
        return "🔧"

    def update_display(self):
        """Update the display based on current state."""
        # Status icon
        icon = self.STATUS_ICONS.get(self._status, "•")
        color = self.STATUS_COLORS.get(self._status, "#757575")
        self._status_icon.setText(icon)
        self._status_icon.setStyleSheet(f"color: {color}; font-size: 16px;")

        # Tool name
        tool_icon = self._get_tool_icon()
        self._name_label.setText(f"{tool_icon} {self._tool_name}")

        # Duration
        if self._duration is not None:
            self._duration_label.setText(f"({self._duration:.2f}s)")
        else:
            self._duration_label.setText("")

        # Parameters
        params_str = "\n".join(f"  {k}: {v}" for k, v in self._params.items())
        self._params_text.setText(params_str)

        # Result
        if self._result is not None:
            result_str = str(self._result)
            if len(result_str) > 500:
                result_str = result_str[:500] + "\n... (truncated)"
            self._result_text.setText(result_str)
            self._result_header.show()
            self._result_text.show()
        else:
            self._result_header.hide()
            self._result_text.hide()

        # Error
        if self._status == "failed":
            error_str = str(self._result) if self._result else "Unknown error"
            self._error_text.setText(error_str)
            self._error_header.show()
            self._error_text.show()
            self._result_header.hide()
            self._result_text.hide()
        else:
            self._error_header.hide()
            self._error_text.hide()

        # Expand button
        has_details = bool(self._result) or self._status == "failed" or bool(self._params)
        self._expand_btn.setVisible(has_details)
        self._expand_btn.setText("▼" if self._expanded else "▶")

        # Border color based on status
        if self._status == "success":
            self.setStyleSheet("""
                ToolCallWidget {
                    background-color: #F1F8E9;
                    border: 1px solid #AED581;
                    border-radius: 6px;
                }
            """)
        elif self._status == "failed":
            self.setStyleSheet("""
                ToolCallWidget {
                    background-color: #FFEBEE;
                    border: 1px solid #EF9A9A;
                    border-radius: 6px;
                }
            """)
        elif self._status == "running":
            self.setStyleSheet("""
                ToolCallWidget {
                    background-color: #E3F2FD;
                    border: 1px solid #90CAF9;
                    border-radius: 6px;
                }
            """)

    def _toggle_expand(self):
        """Toggle expand/collapse."""
        self._expanded = not self._expanded
        self._details_container.setVisible(self._expanded)
        self.update_display()

    def update_status(self, status: str):
        """Update tool call status."""
        self._status = status
        self.update_display()

    def update_result(self, result: Any):
        """Update tool call result."""
        self._result = result
        self.update_display()

    def update_duration(self, duration: float):
        """Update execution duration."""
        self._duration = duration
        self.update_display()

    def set_expanded(self, expanded: bool):
        """Set expanded state."""
        self._expanded = expanded
        self._details_container.setVisible(expanded)
        self.update_display()


class AgentMode(QWidget):
    """Agent mode widget for Agent panel.

    Shows:
    - Thinking chain (reasoning process)
    - Tool calls
    - Execution progress
    - Error handling
    """

    step_clicked = pyqtSignal(str)
    tool_call_clicked = pyqtSignal(str)
    retry_requested = pyqtSignal(str)
    skip_requested = pyqtSignal(str)

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._current_task: Optional[str] = None
        self._tool_calls: Dict[str, ToolCallWidget] = {}
        self._agent_worker: Optional[AgentWorker] = None

        self.setup_ui()

    def setup_ui(self):
        """Setup the agent mode UI."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Header
        header = QFrame()
        header.setStyleSheet("""
            QFrame {
                background-color: #E3F2FD;
                border-bottom: 1px solid #BBDEFB;
            }
        """)
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(12, 8, 12, 8)

        self._status_label = QLabel("🤖 Agent Ready")
        self._status_label.setStyleSheet("font-weight: bold; color: #1976D2;")
        header_layout.addWidget(self._status_label)

        header_layout.addStretch()

        self._clear_btn = QPushButton("🗑️ Clear")
        self._clear_btn.clicked.connect(self.clear)
        header_layout.addWidget(self._clear_btn)

        layout.addWidget(header)

        # Progress tracker
        self._progress_tracker = ProgressTracker()
        self._progress_tracker.setVisible(False)
        layout.addWidget(self._progress_tracker)

        # Error list
        self._error_list = ErrorListWidget()
        self._error_list.setVisible(False)
        self._error_list.retry_requested.connect(self._on_retry_requested)
        self._error_list.skip_requested.connect(self._on_skip_requested)
        layout.addWidget(self._error_list)

        # Tab widget
        self._tabs = QTabWidget()

        # Thinking chain tab
        self._thinking_chain = ThinkingChainWidget()
        self._thinking_chain.step_clicked.connect(self.step_clicked.emit)
        self._tabs.addTab(self._thinking_chain, "💭 Thinking")

        # Tool calls tab
        self._tool_calls_scroll = QScrollArea()
        self._tool_calls_scroll.setWidgetResizable(True)
        self._tool_calls_container = QWidget()
        self._tool_calls_layout = QVBoxLayout(self._tool_calls_container)
        self._tool_calls_layout.setContentsMargins(8, 8, 8, 8)
        self._tool_calls_layout.setSpacing(8)
        self._tool_calls_layout.addStretch()
        self._tool_calls_scroll.setWidget(self._tool_calls_container)
        self._tabs.addTab(self._tool_calls_scroll, "🔧 Tools")

        # Raw output tab
        self._raw_output = QTextEdit()
        self._raw_output.setReadOnly(True)
        self._raw_output.setStyleSheet("""
            QTextEdit {
                background-color: #263238;
                color: #EEFFFF;
                font-family: 'Consolas', 'Monaco', monospace;
                font-size: 12px;
            }
        """)
        self._tabs.addTab(self._raw_output, "📄 Output")

        layout.addWidget(self._tabs, stretch=1)

    def set_agent_worker(self, worker: AgentWorker):
        """Set the agent worker and connect signals.

        Args:
            worker: AgentWorker instance
        """
        self._agent_worker = worker

        # Connect signals
        worker.signals.state_changed.connect(self._on_state_changed)
        worker.signals.progress_updated.connect(self._on_progress_updated)
        worker.signals.thinking_step_added.connect(self._on_thinking_step_added)
        worker.signals.thinking_step_updated.connect(self._on_thinking_step_updated)
        worker.signals.tool_call_started.connect(self._on_tool_call_started)
        worker.signals.tool_call_completed.connect(self._on_tool_call_completed)
        worker.signals.tool_call_failed.connect(self._on_tool_call_failed)
        worker.signals.error_occurred.connect(self._on_error_occurred)
        worker.signals.raw_output.connect(self._on_raw_output)
        worker.signals.task_completed.connect(self._on_task_completed)
        worker.signals.task_failed.connect(self._on_task_failed)
        worker.signals.learning_recorded.connect(self._on_learning_recorded)

    def _on_state_changed(self, state: AgentState, message: str):
        """Handle state change."""
        state_icons = {
            AgentState.IDLE: "🤖",
            AgentState.STARTING: "🚀",
            AgentState.OBSERVING: "👁️",
            AgentState.THINKING: "💭",
            AgentState.ACTING: "🔧",
            AgentState.VERIFYING: "✓",
            AgentState.LEARNING: "📚",
            AgentState.COMPLETED: "✅",
            AgentState.FAILED: "❌",
            AgentState.PAUSED: "⏸️"
        }

        icon = state_icons.get(state, "🤖")
        self._status_label.setText(f"{icon} {message}")

        # Update status color
        if state == AgentState.COMPLETED:
            self._status_label.setStyleSheet("font-weight: bold; color: #4CAF50;")
        elif state == AgentState.FAILED:
            self._status_label.setStyleSheet("font-weight: bold; color: #F44336;")
        elif state == AgentState.ACTING:
            self._status_label.setStyleSheet("font-weight: bold; color: #FF9800;")
        else:
            self._status_label.setStyleSheet("font-weight: bold; color: #1976D2;")

    def _on_progress_updated(self, progress: AgentProgress):
        """Handle progress update."""
        self._progress_tracker.setVisible(True)
        self._progress_tracker.set_progress(
            progress.current_step,
            progress.total_steps,
            progress.message
        )

    def _on_thinking_step_added(self, step_info: ThinkingStepInfo):
        """Handle thinking step added."""
        step = ThinkingStep(
            id=step_info.id,
            title=step_info.title,
            description=step_info.description,
            status=StepStatus.PENDING
        )
        self._thinking_chain.add_step(step, step_info.parent_id)

    def _on_thinking_step_updated(self, step_id: str, status: str, details: Any):
        """Handle thinking step updated."""
        if status == "running":
            self._thinking_chain.start_step(step_id)
        elif status == "completed":
            self._thinking_chain.complete_step(step_id)
        elif status == "failed":
            self._thinking_chain.fail_step(step_id)
        elif status == "skipped":
            self._thinking_chain.update_step(step_id, status=StepStatus.SKIPPED)

        if details:
            self._thinking_chain.add_detail(step_id, str(details))

    def _on_tool_call_started(self, tool_info: ToolCallInfo):
        """Handle tool call started."""
        widget = ToolCallWidget(
            tool_name=tool_info.tool_name,
            params=tool_info.parameters,
            status="running"
        )
        self._tool_calls[tool_info.id] = widget

        # Insert before stretch
        index = self._tool_calls_layout.count() - 1
        self._tool_calls_layout.insertWidget(index, widget)

        # Switch to tools tab
        self._tabs.setCurrentIndex(1)

    def _on_tool_call_completed(self, tool_info: ToolCallInfo):
        """Handle tool call completed."""
        if tool_info.id in self._tool_calls:
            widget = self._tool_calls[tool_info.id]
            widget.update_status("success")
            widget.update_result(tool_info.result)
            if tool_info.duration:
                widget.update_duration(tool_info.duration)

    def _on_tool_call_failed(self, tool_info: ToolCallInfo, error: str):
        """Handle tool call failed."""
        if tool_info.id in self._tool_calls:
            widget = self._tool_calls[tool_info.id]
            widget.update_status("failed")
            widget.update_result(error)
            if tool_info.duration:
                widget.update_duration(tool_info.duration)

    def _on_error_occurred(self, error: Any):
        """Handle error occurred."""
        self._error_list.setVisible(True)
        self._error_list.add_error(
            error_id=error.step_id,
            error_message=error.error_message,
            error_type=error.error_type,
            context=error.context,
            retryable=error.retryable
        )

    def _on_raw_output(self, text: str):
        """Handle raw output."""
        self._raw_output.append(text)

    def _on_task_completed(self, task_name: str, result: Any):
        """Handle task completed."""
        self._status_label.setText(f"✅ {task_name} Complete")
        self._status_label.setStyleSheet("font-weight: bold; color: #4CAF50;")
        self._progress_tracker.set_status("Completed", "success")

    def _on_task_failed(self, task_name: str, error: str):
        """Handle task failed."""
        self._status_label.setText(f"❌ {task_name} Failed")
        self._status_label.setStyleSheet("font-weight: bold; color: #F44336;")
        self._progress_tracker.set_status("Failed", "error")

    def _on_learning_recorded(self, learning: str):
        """Handle learning recorded."""
        self._raw_output.append(f"[Learning] {learning}\n")

    def _on_retry_requested(self, error_id: str):
        """Handle retry requested."""
        self.retry_requested.emit(error_id)
        if self._agent_worker:
            self._agent_worker.retry_step(error_id)

    def _on_skip_requested(self, error_id: str):
        """Handle skip requested."""
        self.skip_requested.emit(error_id)
        if self._agent_worker:
            self._agent_worker.skip_step(error_id)
        self._error_list.remove_error(error_id)

    def start_task(self, task_name: str):
        """Start a new task.

        Args:
            task_name: Name/description of the task
        """
        self._current_task = task_name
        self._status_label.setText(f"🤖 {task_name}")
        self._thinking_chain.clear()
        self._raw_output.clear()
        self._progress_tracker.clear()
        self._progress_tracker.setVisible(True)
        self._error_list.clear_errors()
        self._error_list.setVisible(False)

        # Clear tool calls
        while self._tool_calls_layout.count() > 1:
            item = self._tool_calls_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        self._tool_calls.clear()

        logger.info(f"Agent task started: {task_name}")

    def add_thinking_step(self, step: ThinkingStep, parent_id: Optional[str] = None):
        """Add a thinking step.

        Args:
            step: The thinking step
            parent_id: Optional parent step ID
        """
        self._thinking_chain.add_step(step, parent_id)

    def start_step(self, step_id: str):
        """Mark a step as running."""
        self._thinking_chain.start_step(step_id)

    def complete_step(self, step_id: str):
        """Mark a step as completed."""
        self._thinking_chain.complete_step(step_id)

    def fail_step(self, step_id: str):
        """Mark a step as failed."""
        self._thinking_chain.fail_step(step_id)

    def add_step_detail(self, step_id: str, detail: str):
        """Add a detail to a step."""
        self._thinking_chain.add_detail(step_id, detail)

    def add_tool_call(self, tool_name: str, params: Dict[str, Any],
                     result: Optional[Any] = None):
        """Add a tool call.

        Args:
            tool_name: Name of the tool
            params: Tool parameters
            result: Optional result
        """
        widget = ToolCallWidget(tool_name, params, result)
        tool_id = f"manual_{len(self._tool_calls)}"
        self._tool_calls[tool_id] = widget

        # Insert before stretch
        index = self._tool_calls_layout.count() - 1
        self._tool_calls_layout.insertWidget(index, widget)

        # Switch to tools tab
        self._tabs.setCurrentIndex(1)

        logger.debug(f"Tool call: {tool_name}")

    def update_tool_call_status(self, tool_id: str, status: str):
        """Update tool call status.

        Args:
            tool_id: Tool call ID
            status: New status
        """
        if tool_id in self._tool_calls:
            self._tool_calls[tool_id].update_status(status)

    def update_tool_call_result(self, tool_id: str, result: Any):
        """Update tool call result.

        Args:
            tool_id: Tool call ID
            result: Tool result
        """
        if tool_id in self._tool_calls:
            self._tool_calls[tool_id].update_result(result)

    def append_raw_output(self, text: str):
        """Append text to raw output."""
        self._raw_output.append(text)

    def set_progress(self, current: int, total: int, message: str = ""):
        """Set progress.

        Args:
            current: Current step
            total: Total steps
            message: Progress message
        """
        self._progress_tracker.setVisible(True)
        self._progress_tracker.set_progress(current, total, message)

    def add_error(self, error_id: str, error_message: str,
                  error_type: str = "Execution Error",
                  context: Optional[Dict[str, Any]] = None,
                  retryable: bool = True):
        """Add an error.

        Args:
            error_id: Error ID
            error_message: Error message
            error_type: Error type
            context: Error context
            retryable: Whether error is retryable
        """
        self._error_list.setVisible(True)
        self._error_list.add_error(error_id, error_message, error_type, context, retryable)

    def clear(self):
        """Clear all content."""
        self._current_task = None
        self._status_label.setText("🤖 Agent Ready")
        self._status_label.setStyleSheet("font-weight: bold; color: #1976D2;")

        self._thinking_chain.clear()
        self._raw_output.clear()
        self._progress_tracker.clear()
        self._progress_tracker.setVisible(False)
        self._error_list.clear_errors()
        self._error_list.setVisible(False)

        while self._tool_calls_layout.count() > 1:
            item = self._tool_calls_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        self._tool_calls.clear()

    def set_status(self, status: str, status_type: str = "info"):
        """Set agent status.

        Args:
            status: Status text
            status_type: One of "info", "success", "warning", "error"
        """
        self._status_label.setText(f"🤖 {status}")

        colors = {
            "info": "#1976D2",
            "success": "#4CAF50",
            "warning": "#FF9800",
            "error": "#F44336"
        }
        color = colors.get(status_type, "#1976D2")
        self._status_label.setStyleSheet(f"font-weight: bold; color: {color};")

    def show_thinking_tab(self):
        """Switch to thinking chain tab."""
        self._tabs.setCurrentIndex(0)

    def show_tools_tab(self):
        """Switch to tools tab."""
        self._tabs.setCurrentIndex(1)

    def show_output_tab(self):
        """Switch to raw output tab."""
        self._tabs.setCurrentIndex(2)

    def show_progress_tracker(self):
        """Show progress tracker."""
        self._progress_tracker.setVisible(True)

    def hide_progress_tracker(self):
        """Hide progress tracker."""
        self._progress_tracker.setVisible(False)

    def show_error_list(self):
        """Show error list."""
        self._error_list.setVisible(True)

    def hide_error_list(self):
        """Hide error list."""
        self._error_list.setVisible(False)
