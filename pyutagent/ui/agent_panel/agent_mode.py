"""Agent mode for Agent panel - shows thinking chain and tool calls."""

import logging
from typing import Optional, List, Dict, Any

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QTabWidget, QTextEdit, QFrame, QScrollArea
)
from PyQt6.QtCore import Qt, pyqtSignal

from .thinking_chain import ThinkingChainWidget, ThinkingStep, StepStatus

logger = logging.getLogger(__name__)


class ToolCallWidget(QFrame):
    """Widget for displaying a tool call."""
    
    def __init__(self, tool_name: str, params: Dict[str, Any], 
                 result: Optional[Any] = None, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._tool_name = tool_name
        self._params = params
        self._result = result
        
        self.setup_ui()
        
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
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 8, 12, 8)
        layout.setSpacing(8)
        
        # Header
        header = QHBoxLayout()
        
        tool_icon = "🔧"
        if "file" in self._tool_name.lower():
            tool_icon = "📄"
        elif "terminal" in self._tool_name.lower() or "command" in self._tool_name.lower():
            tool_icon = "⌨️"
        elif "search" in self._tool_name.lower():
            tool_icon = "🔍"
        elif "edit" in self._tool_name.lower():
            tool_icon = "✏️"
        
        name_label = QLabel(f"{tool_icon} {self._tool_name}")
        name_label.setStyleSheet("font-weight: bold; color: #333;")
        header.addWidget(name_label)
        
        header.addStretch()
        
        layout.addLayout(header)
        
        # Parameters
        params_text = "\n".join(f"  {k}: {v}" for k, v in self._params.items())
        params_label = QLabel(f"Parameters:\n{params_text}")
        params_label.setStyleSheet("color: #666; font-family: monospace; font-size: 12px;")
        layout.addWidget(params_label)
        
        # Result (if available)
        if self._result is not None:
            result_text = str(self._result)
            if len(result_text) > 200:
                result_text = result_text[:200] + "..."
            
            result_label = QLabel(f"Result:\n  {result_text}")
            result_label.setStyleSheet("""
                color: #4CAF50;
                font-family: monospace;
                font-size: 12px;
                background-color: #E8F5E9;
                padding: 8px;
                border-radius: 4px;
            """)
            layout.addWidget(result_label)


class AgentMode(QWidget):
    """Agent mode widget for Agent panel.
    
    Shows:
    - Thinking chain (reasoning process)
    - Tool calls
    - Execution progress
    """
    
    step_clicked = pyqtSignal(str)  # step_id
    tool_call_clicked = pyqtSignal(str)  # tool_name
    
    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._current_task: Optional[str] = None
        self._tool_calls: List[ToolCallWidget] = []
        
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
        
        # Progress bar
        self._progress_frame = QFrame()
        self._progress_frame.setStyleSheet("""
            QFrame {
                background-color: #FAFAFA;
                border-top: 1px solid #E0E0E0;
            }
        """)
        progress_layout = QHBoxLayout(self._progress_frame)
        progress_layout.setContentsMargins(12, 8, 12, 8)
        
        self._progress_label = QLabel("No task running")
        progress_layout.addWidget(self._progress_label)
        
        progress_layout.addStretch()
        
        self._step_count_label = QLabel("0/0 steps")
        progress_layout.addWidget(self._step_count_label)
        
        layout.addWidget(self._progress_frame)
        
    def start_task(self, task_name: str):
        """Start a new task.
        
        Args:
            task_name: Name/description of the task
        """
        self._current_task = task_name
        self._status_label.setText(f"🤖 {task_name}")
        self._progress_label.setText("Starting...")
        self._thinking_chain.clear()
        self._raw_output.clear()
        
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
        self._update_progress()
        
    def start_step(self, step_id: str):
        """Mark a step as running."""
        self._thinking_chain.start_step(step_id)
        self._update_progress()
        
    def complete_step(self, step_id: str):
        """Mark a step as completed."""
        self._thinking_chain.complete_step(step_id)
        self._update_progress()
        
    def fail_step(self, step_id: str):
        """Mark a step as failed."""
        self._thinking_chain.fail_step(step_id)
        self._update_progress()
        
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
        self._tool_calls.append(widget)
        
        # Insert before stretch
        index = self._tool_calls_layout.count() - 1
        self._tool_calls_layout.insertWidget(index, widget)
        
        # Switch to tools tab
        self._tabs.setCurrentIndex(1)
        
        logger.debug(f"Tool call: {tool_name}")
        
    def append_raw_output(self, text: str):
        """Append text to raw output."""
        self._raw_output.append(text)
        
    def _update_progress(self):
        """Update progress display."""
        total = self._thinking_chain.get_step_count()
        completed = self._thinking_chain.get_completed_count()
        
        self._step_count_label.setText(f"{completed}/{total} steps")
        
        if total > 0:
            progress = int((completed / total) * 100)
            self._progress_label.setText(f"Progress: {progress}%")
        
        if completed == total and total > 0:
            self._status_label.setText(f"✅ {self._current_task or 'Task'} Complete")
            self._status_label.setStyleSheet("font-weight: bold; color: #4CAF50;")
        
    def clear(self):
        """Clear all content."""
        self._current_task = None
        self._status_label.setText("🤖 Agent Ready")
        self._status_label.setStyleSheet("font-weight: bold; color: #1976D2;")
        self._progress_label.setText("No task running")
        self._step_count_label.setText("0/0 steps")
        
        self._thinking_chain.clear()
        self._raw_output.clear()
        
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
