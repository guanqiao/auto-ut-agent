"""Thinking chain visualization for Agent reasoning process."""

import logging
from typing import Optional, List, Dict, Any
from enum import Enum
from dataclasses import dataclass, field

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QFrame,
    QScrollArea, QSizePolicy
)
from PyQt6.QtCore import Qt, pyqtSignal, QTimer
from PyQt6.QtGui import QFont, QColor

logger = logging.getLogger(__name__)


class StepStatus(Enum):
    """Status of a thinking step."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class ThinkingStep:
    """A single step in the thinking chain."""
    id: str
    title: str
    description: str = ""
    status: StepStatus = StepStatus.PENDING
    details: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    
    @property
    def duration(self) -> Optional[float]:
        """Calculate step duration."""
        if self.start_time and self.end_time:
            return self.end_time - self.start_time
        return None


class ThinkingStepWidget(QFrame):
    """Widget for displaying a single thinking step."""
    
    clicked = pyqtSignal(str)  # step_id
    
    # Status icons
    STATUS_ICONS = {
        StepStatus.PENDING: "⏳",
        StepStatus.RUNNING: "🔄",
        StepStatus.COMPLETED: "✓",
        StepStatus.FAILED: "✗",
        StepStatus.SKIPPED: "⊘"
    }
    
    # Status colors
    STATUS_COLORS = {
        StepStatus.PENDING: "#757575",
        StepStatus.RUNNING: "#2196F3",
        StepStatus.COMPLETED: "#4CAF50",
        StepStatus.FAILED: "#F44336",
        StepStatus.SKIPPED: "#9E9E9E"
    }
    
    def __init__(self, step: ThinkingStep, level: int = 0, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._step = step
        self._level = level
        self._expanded = False
        
        self.setup_ui()
        self.update_display()
        
    def setup_ui(self):
        """Setup the step widget UI."""
        self.setFrameShape(QFrame.Shape.NoFrame)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        
        self._main_layout = QVBoxLayout(self)
        self._main_layout.setContentsMargins(20 * self._level, 4, 8, 4)
        self._main_layout.setSpacing(4)
        
        # Header row
        self._header = QWidget()
        header_layout = QHBoxLayout(self._header)
        header_layout.setContentsMargins(0, 0, 0, 0)
        header_layout.setSpacing(8)
        
        # Status icon
        self._icon_label = QLabel()
        self._icon_label.setFixedWidth(20)
        header_layout.addWidget(self._icon_label)
        
        # Title
        self._title_label = QLabel()
        self._title_label.setFont(QFont("Segoe UI", 10, QFont.Weight.Bold))
        header_layout.addWidget(self._title_label)
        
        # Duration (if completed)
        self._duration_label = QLabel()
        self._duration_label.setStyleSheet("color: #999; font-size: 11px;")
        header_layout.addWidget(self._duration_label)
        
        header_layout.addStretch()
        
        # Expand indicator (if has details)
        self._expand_label = QLabel()
        self._expand_label.setFixedWidth(16)
        header_layout.addWidget(self._expand_label)
        
        self._main_layout.addWidget(self._header)
        
        # Details container (hidden by default)
        self._details_container = QWidget()
        details_layout = QVBoxLayout(self._details_container)
        details_layout.setContentsMargins(28, 0, 0, 0)
        details_layout.setSpacing(2)
        
        self._details_labels: List[QLabel] = []
        for detail in self._step.details:
            label = QLabel(f"  └─ {detail}")
            label.setStyleSheet("color: #666; font-size: 12px;")
            label.setWordWrap(True)
            details_layout.addWidget(label)
            self._details_labels.append(label)
        
        self._details_container.hide()
        self._main_layout.addWidget(self._details_container)
        
        # Connect click
        self._header.mousePressEvent = self._on_header_clicked
        
    def update_display(self):
        """Update the display based on current step state."""
        # Icon
        self._icon_label.setText(self.STATUS_ICONS.get(self._step.status, "•"))
        self._icon_label.setStyleSheet(f"color: {self.STATUS_COLORS.get(self._step.status, '#757575')}")
        
        # Title
        self._title_label.setText(self._step.title)
        color = self.STATUS_COLORS.get(self._step.status, "#757575")
        self._title_label.setStyleSheet(f"color: {color};")
        
        # Duration
        if self._step.duration:
            self._duration_label.setText(f"({self._step.duration:.1f}s)")
        else:
            self._duration_label.setText("")
        
        # Expand indicator
        if self._step.details:
            self._expand_label.setText("▼" if self._expanded else "▶")
        else:
            self._expand_label.setText("")
        
        # Update details
        self._update_details()
        
    def _update_details(self):
        """Update details display."""
        # Clear existing
        for label in self._details_labels:
            label.deleteLater()
        self._details_labels.clear()
        
        # Add new details
        layout = self._details_container.layout()
        for detail in self._step.details:
            label = QLabel(f"  └─ {detail}")
            label.setStyleSheet("color: #666; font-size: 12px;")
            label.setWordWrap(True)
            layout.addWidget(label)
            self._details_labels.append(label)
        
        # Show/hide container
        self._details_container.setVisible(self._expanded and bool(self._step.details))
        
    def _on_header_clicked(self, event):
        """Handle header click."""
        if self._step.details:
            self._expanded = not self._expanded
            self.update_display()
        self.clicked.emit(self._step.id)
        
    def update_step(self, step: ThinkingStep):
        """Update the step data."""
        self._step = step
        self.update_display()
        
    @property
    def step_id(self) -> str:
        """Get step ID."""
        return self._step.id


class ThinkingChainWidget(QScrollArea):
    """Widget for visualizing the Agent's thinking chain.
    
    Displays a tree-like structure of the Agent's reasoning process,
    including planning, execution, and results.
    """
    
    step_clicked = pyqtSignal(str)  # step_id
    
    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._steps: Dict[str, ThinkingStepWidget] = {}
        self._step_order: List[str] = []
        
        self.setup_ui()
        
    def setup_ui(self):
        """Setup the thinking chain UI."""
        self.setWidgetResizable(True)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        
        # Container widget
        self._container = QWidget()
        self._layout = QVBoxLayout(self._container)
        self._layout.setContentsMargins(8, 8, 8, 8)
        self._layout.setSpacing(4)
        self._layout.addStretch()
        
        self.setWidget(self._container)
        
        # Header
        self._header = QLabel("🤔 Agent Thinking Process")
        self._header.setStyleSheet("font-size: 14px; font-weight: bold; padding-bottom: 8px;")
        self._layout.insertWidget(0, self._header)
        
    def add_step(self, step: ThinkingStep, parent_id: Optional[str] = None) -> ThinkingStepWidget:
        """Add a thinking step.
        
        Args:
            step: The thinking step to add
            parent_id: Optional parent step ID for nesting
            
        Returns:
            The created step widget
        """
        # Calculate level based on parent
        level = 0
        if parent_id and parent_id in self._steps:
            # Find parent's level
            for i, sid in enumerate(self._step_order):
                if sid == parent_id:
                    # Count indentation level
                    level = 1  # Direct child
                    break
        
        widget = ThinkingStepWidget(step, level=level)
        widget.clicked.connect(self.step_clicked.emit)
        
        # Insert before stretch
        insert_index = self._layout.count() - 1
        self._layout.insertWidget(insert_index, widget)
        
        self._steps[step.id] = widget
        self._step_order.append(step.id)
        
        # Auto-scroll to new step
        QTimer.singleShot(50, self._scroll_to_bottom)
        
        return widget
        
    def update_step(self, step_id: str, **kwargs):
        """Update a step's properties.
        
        Args:
            step_id: Step ID to update
            **kwargs: Properties to update (status, description, details, etc.)
        """
        if step_id not in self._steps:
            logger.warning(f"Step not found: {step_id}")
            return
        
        widget = self._steps[step_id]
        step = widget._step
        
        # Update properties
        if 'status' in kwargs:
            step.status = kwargs['status']
        if 'description' in kwargs:
            step.description = kwargs['description']
        if 'details' in kwargs:
            step.details = kwargs['details']
        if 'end_time' in kwargs:
            step.end_time = kwargs['end_time']
        
        widget.update_display()
        
    def start_step(self, step_id: str):
        """Mark a step as running."""
        import time
        self.update_step(step_id, status=StepStatus.RUNNING, start_time=time.time())
        
    def complete_step(self, step_id: str):
        """Mark a step as completed."""
        import time
        self.update_step(step_id, status=StepStatus.COMPLETED, end_time=time.time())
        
    def fail_step(self, step_id: str):
        """Mark a step as failed."""
        import time
        self.update_step(step_id, status=StepStatus.FAILED, end_time=time.time())
        
    def add_detail(self, step_id: str, detail: str):
        """Add a detail to a step."""
        if step_id not in self._steps:
            return
        
        widget = self._steps[step_id]
        step = widget._step
        step.details.append(detail)
        widget.update_display()
        
    def clear(self):
        """Clear all steps."""
        for widget in self._steps.values():
            widget.deleteLater()
        self._steps.clear()
        self._step_order.clear()
        
    def _scroll_to_bottom(self):
        """Scroll to the bottom of the chain."""
        scrollbar = self.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())
        
    def get_step_count(self) -> int:
        """Get total number of steps."""
        return len(self._steps)
        
    def get_completed_count(self) -> int:
        """Get number of completed steps."""
        return sum(1 for w in self._steps.values() if w._step.status == StepStatus.COMPLETED)
