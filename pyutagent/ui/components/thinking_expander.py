"""Thinking process expander widget for displaying AI reasoning."""

import logging
import time
from typing import Optional, List
from dataclasses import dataclass, field
from enum import Enum

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QTextEdit, QFrame, QScrollArea, QSizePolicy, QGraphicsOpacityEffect
)
from PyQt6.QtCore import Qt, pyqtSignal, QTimer, QPropertyAnimation, QEasingCurve
from PyQt6.QtGui import QFont, QColor

logger = logging.getLogger(__name__)


class ThinkingStatus(Enum):
    """Status of thinking process."""
    PENDING = "pending"
    THINKING = "thinking"
    COMPLETED = "completed"
    ERROR = "error"


@dataclass
class ThinkingStep:
    """A single step in the thinking process."""
    id: str
    title: str
    description: str = ""
    status: ThinkingStatus = ThinkingStatus.PENDING
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    details: List[str] = field(default_factory=list)
    
    @property
    def duration_ms(self) -> float:
        """Get step duration in milliseconds."""
        if self.start_time:
            end = self.end_time or time.time()
            return (end - self.start_time) * 1000
        return 0.0
    
    def start(self):
        """Mark step as started."""
        self.status = ThinkingStatus.THINKING
        self.start_time = time.time()
        
    def complete(self):
        """Mark step as completed."""
        self.status = ThinkingStatus.COMPLETED
        self.end_time = time.time()
        
    def fail(self):
        """Mark step as failed."""
        self.status = ThinkingStatus.ERROR
        self.end_time = time.time()
        
    def add_detail(self, detail: str):
        """Add a detail to the step."""
        self.details.append(detail)


class ThinkingStepWidget(QFrame):
    """Widget for displaying a single thinking step."""
    
    clicked = pyqtSignal(str)  # step_id
    
    STATUS_ICONS = {
        ThinkingStatus.PENDING: "⏳",
        ThinkingStatus.THINKING: "💭",
        ThinkingStatus.COMPLETED: "✅",
        ThinkingStatus.ERROR: "❌"
    }
    
    STATUS_COLORS = {
        ThinkingStatus.PENDING: "#9E9E9E",
        ThinkingStatus.THINKING: "#2196F3",
        ThinkingStatus.COMPLETED: "#4CAF50",
        ThinkingStatus.ERROR: "#F44336"
    }
    
    def __init__(self, step: ThinkingStep, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._step = step
        self._expanded = False
        
        self.setup_ui()
        self.update_display()
        
    def setup_ui(self):
        """Setup the step widget UI."""
        self.setFrameShape(QFrame.Shape.StyledPanel)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setStyleSheet("""
            ThinkingStepWidget {
                background-color: #FAFAFA;
                border: 1px solid #E0E0E0;
                border-radius: 6px;
            }
            ThinkingStepWidget:hover {
                background-color: #F5F5F5;
                border-color: #BDBDBD;
            }
        """)
        
        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(12, 8, 12, 8)
        self._layout.setSpacing(4)
        
        # Header
        header = QHBoxLayout()
        
        self._icon_label = QLabel()
        self._icon_label.setFont(QFont("Segoe UI", 14))
        header.addWidget(self._icon_label)
        
        self._title_label = QLabel(self._step.title)
        self._title_label.setFont(QFont("Segoe UI", 11, QFont.Weight.Bold))
        header.addWidget(self._title_label)
        
        header.addStretch()
        
        self._time_label = QLabel()
        self._time_label.setStyleSheet("color: #999; font-size: 11px;")
        header.addWidget(self._time_label)
        
        self._expand_btn = QPushButton("▼")
        self._expand_btn.setFixedSize(24, 24)
        self._expand_btn.setStyleSheet("""
            QPushButton {
                background-color: transparent;
                border: none;
                color: #666;
            }
            QPushButton:hover {
                color: #333;
            }
        """)
        self._expand_btn.clicked.connect(self._toggle_expand)
        header.addWidget(self._expand_btn)
        
        self._layout.addLayout(header)
        
        # Description
        if self._step.description:
            self._desc_label = QLabel(self._step.description)
            self._desc_label.setStyleSheet("color: #666; font-size: 12px;")
            self._desc_label.setWordWrap(True)
            self._layout.addWidget(self._desc_label)
        
        # Details (expandable)
        self._details_widget = QWidget()
        details_layout = QVBoxLayout(self._details_widget)
        details_layout.setContentsMargins(0, 0, 0, 0)
        details_layout.setSpacing(4)
        
        self._details_text = QTextEdit()
        self._details_text.setReadOnly(True)
        self._details_text.setFrameStyle(QFrame.Shape.NoFrame)
        self._details_text.setStyleSheet("""
            QTextEdit {
                background-color: #263238;
                color: #EEFFFF;
                font-family: 'Consolas', 'Monaco', monospace;
                font-size: 11px;
                padding: 8px;
                border-radius: 4px;
            }
        """)
        self._details_text.setMaximumHeight(0)
        details_layout.addWidget(self._details_text)
        
        self._layout.addWidget(self._details_widget)
        self._details_widget.setVisible(False)
        
        # Click to expand
        self.mousePressEvent = lambda e: self._toggle_expand()
        
    def update_display(self):
        """Update the display based on current step state."""
        # Update icon
        icon = self.STATUS_ICONS.get(self._step.status, "⏳")
        self._icon_label.setText(icon)
        
        # Update color
        color = self.STATUS_COLORS.get(self._step.status, "#9E9E9E")
        self._title_label.setStyleSheet(f"color: {color};")
        
        # Update time
        if self._step.duration_ms > 0:
            duration_str = self._format_duration(self._step.duration_ms)
            self._time_label.setText(duration_str)
        
        # Update details
        if self._step.details:
            details_text = "\n".join(f"• {d}" for d in self._step.details)
            self._details_text.setText(details_text)
            
    def _format_duration(self, ms: float) -> str:
        """Format duration in human readable form."""
        if ms < 1000:
            return f"{ms:.0f}ms"
        elif ms < 60000:
            return f"{ms/1000:.1f}s"
        else:
            return f"{ms/60000:.1f}m"
            
    def _toggle_expand(self):
        """Toggle details expansion."""
        self._expanded = not self._expanded
        
        if self._expanded:
            self._details_widget.setVisible(True)
            self._details_text.setMaximumHeight(200)
            self._expand_btn.setText("▲")
        else:
            self._details_text.setMaximumHeight(0)
            self._expand_btn.setText("▼")
            QTimer.singleShot(200, lambda: self._details_widget.setVisible(False))
            
        self.clicked.emit(self._step.id)
        
    def get_step(self) -> ThinkingStep:
        """Get the associated thinking step."""
        return self._step
        
    def refresh(self):
        """Refresh the display."""
        self.update_display()


class ThinkingExpander(QFrame):
    """Expandable widget for displaying AI thinking process.
    
    Features:
    - Collapsible/expandable interface
    - Shows thinking steps with status
    - Displays thinking duration
    - Real-time updates
    """
    
    step_clicked = pyqtSignal(str)
    expanded_changed = pyqtSignal(bool)
    
    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._steps: List[ThinkingStep] = []
        self._step_widgets: dict = {}  # id -> widget
        self._expanded = False
        self._start_time: Optional[float] = None
        self._end_time: Optional[float] = None
        
        self.setup_ui()
        
    def setup_ui(self):
        """Setup the thinking expander UI."""
        self.setFrameShape(QFrame.Shape.StyledPanel)
        self.setStyleSheet("""
            ThinkingExpander {
                background-color: #E3F2FD;
                border: 1px solid #BBDEFB;
                border-radius: 8px;
            }
        """)
        
        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(0, 0, 0, 0)
        self._layout.setSpacing(0)
        
        # Header
        header = QFrame()
        header.setStyleSheet("""
            QFrame {
                background-color: #1976D2;
                border-top-left-radius: 8px;
                border-top-right-radius: 8px;
            }
        """)
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(12, 8, 12, 8)
        
        self._header_label = QLabel("💭 Thinking Process")
        self._header_label.setStyleSheet("""
            color: white;
            font-weight: bold;
            font-size: 13px;
        """)
        header_layout.addWidget(self._header_label)
        
        header_layout.addStretch()
        
        self._time_label = QLabel()
        self._time_label.setStyleSheet("color: rgba(255,255,255,0.8); font-size: 11px;")
        header_layout.addWidget(self._time_label)
        
        self._toggle_btn = QPushButton("▼")
        self._toggle_btn.setFixedSize(28, 28)
        self._toggle_btn.setStyleSheet("""
            QPushButton {
                background-color: rgba(255,255,255,0.2);
                color: white;
                border: none;
                border-radius: 4px;
                font-size: 12px;
            }
            QPushButton:hover {
                background-color: rgba(255,255,255,0.3);
            }
        """)
        self._toggle_btn.clicked.connect(self._toggle_expanded)
        header_layout.addWidget(self._toggle_btn)
        
        self._layout.addWidget(header)
        
        # Content area (collapsible)
        self._content = QWidget()
        content_layout = QVBoxLayout(self._content)
        content_layout.setContentsMargins(12, 12, 12, 12)
        content_layout.setSpacing(8)
        
        # Summary label
        self._summary_label = QLabel("Analyzing your request...")
        self._summary_label.setStyleSheet("color: #666; font-style: italic;")
        content_layout.addWidget(self._summary_label)
        
        # Steps container
        self._steps_container = QWidget()
        self._steps_layout = QVBoxLayout(self._steps_container)
        self._steps_layout.setContentsMargins(0, 0, 0, 0)
        self._steps_layout.setSpacing(8)
        self._steps_layout.addStretch()
        
        content_layout.addWidget(self._steps_container)
        
        self._layout.addWidget(self._content)
        self._content.setVisible(False)
        
        # Update timer
        self._update_timer = QTimer(self)
        self._update_timer.timeout.connect(self._update_time_display)
        
    def _toggle_expanded(self):
        """Toggle expanded state."""
        self._expanded = not self._expanded
        
        if self._expanded:
            self._content.setVisible(True)
            self._toggle_btn.setText("▲")
        else:
            self._toggle_btn.setText("▼")
            self._content.setVisible(False)
            
        self.expanded_changed.emit(self._expanded)
        
    def start_thinking(self):
        """Start a new thinking process."""
        self._start_time = time.time()
        self._end_time = None
        self._steps.clear()
        
        # Clear existing widgets
        while self._steps_layout.count() > 1:
            item = self._steps_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        self._step_widgets.clear()
        
        self._summary_label.setText("Analyzing your request...")
        self._update_timer.start(100)  # Update every 100ms
        
        # Auto-expand on start
        if not self._expanded:
            self._toggle_expanded()
            
    def add_step(self, step: ThinkingStep):
        """Add a thinking step.
        
        Args:
            step: The thinking step to add
        """
        self._steps.append(step)
        
        widget = ThinkingStepWidget(step)
        widget.clicked.connect(self.step_clicked.emit)
        self._step_widgets[step.id] = widget
        
        # Insert before stretch
        index = self._steps_layout.count() - 1
        self._steps_layout.insertWidget(index, widget)
        
        self._update_summary()
        
    def start_step(self, step_id: str):
        """Mark a step as started."""
        for step in self._steps:
            if step.id == step_id:
                step.start()
                if step_id in self._step_widgets:
                    self._step_widgets[step_id].refresh()
                break
                
    def complete_step(self, step_id: str):
        """Mark a step as completed."""
        for step in self._steps:
            if step.id == step_id:
                step.complete()
                if step_id in self._step_widgets:
                    self._step_widgets[step_id].refresh()
                break
        self._update_summary()
        
    def fail_step(self, step_id: str):
        """Mark a step as failed."""
        for step in self._steps:
            if step.id == step_id:
                step.fail()
                if step_id in self._step_widgets:
                    self._step_widgets[step_id].refresh()
                break
        self._update_summary()
        
    def add_step_detail(self, step_id: str, detail: str):
        """Add detail to a step."""
        for step in self._steps:
            if step.id == step_id:
                step.add_detail(detail)
                if step_id in self._step_widgets:
                    self._step_widgets[step_id].refresh()
                break
                
    def finish_thinking(self):
        """Finish the thinking process."""
        self._end_time = time.time()
        self._update_timer.stop()
        self._update_time_display()
        self._update_summary()
        
        # Update header
        self._header_label.setText("✅ Thinking Complete")
        
    def _update_summary(self):
        """Update the summary text."""
        total = len(self._steps)
        completed = sum(1 for s in self._steps if s.status == ThinkingStatus.COMPLETED)
        
        if self._end_time:
            self._summary_label.setText(f"Completed {completed}/{total} steps")
        else:
            self._summary_label.setText(f"Progress: {completed}/{total} steps")
            
    def _update_time_display(self):
        """Update the time display."""
        if self._start_time:
            duration = (self._end_time or time.time()) - self._start_time
            if duration < 60:
                time_str = f"{duration:.1f}s"
            else:
                time_str = f"{duration/60:.1f}m"
            self._time_label.setText(time_str)
            
        # Refresh step widgets
        for widget in self._step_widgets.values():
            widget.refresh()
            
    def get_duration_ms(self) -> float:
        """Get total thinking duration in milliseconds."""
        if self._start_time:
            end = self._end_time or time.time()
            return (end - self._start_time) * 1000
        return 0.0
        
    def get_steps(self) -> List[ThinkingStep]:
        """Get all thinking steps."""
        return self._steps.copy()
        
    def is_expanded(self) -> bool:
        """Check if expander is expanded."""
        return self._expanded
        
    def set_expanded(self, expanded: bool):
        """Set expanded state."""
        if expanded != self._expanded:
            self._toggle_expanded()
            
    def clear(self):
        """Clear all thinking data."""
        self._steps.clear()
        self._start_time = None
        self._end_time = None
        
        while self._steps_layout.count() > 1:
            item = self._steps_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        self._step_widgets.clear()
        
        self._header_label.setText("💭 Thinking Process")
        self._summary_label.setText("Ready")
        self._time_label.setText("")
