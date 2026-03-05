"""Progress widget for displaying generation progress and logs."""

import logging
from datetime import datetime
from typing import Optional

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QTextEdit,
    QLabel, QProgressBar, QPushButton, QSplitter, QFrame
)
from PyQt6.QtCore import Qt

from ..styles import get_style_manager

logger = logging.getLogger(__name__)


class ProgressWidget(QWidget):
    """Widget displaying generation progress and logs."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._style_manager = get_style_manager()
        self.setup_ui()

    def setup_ui(self):
        """Setup the UI with vertical splitter for progress and logs."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        splitter = QSplitter(Qt.Orientation.Vertical)

        progress_container = QWidget()
        progress_layout = QVBoxLayout(progress_container)
        progress_layout.setContentsMargins(10, 10, 10, 10)

        header = QLabel("Progress")
        header.setStyleSheet("font-size: 16px; font-weight: bold; padding: 5px;")
        progress_layout.addWidget(header)

        self.state_label = QLabel("Status: Ready")
        self.state_label.setStyleSheet("font-weight: bold; color: #666;")
        progress_layout.addWidget(self.state_label)

        self.status_label = QLabel("Waiting to start...")
        progress_layout.addWidget(self.status_label)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        progress_layout.addWidget(self.progress_bar)

        self.coverage_label = QLabel("Coverage: -")
        progress_layout.addWidget(self.coverage_label)

        self.iteration_label = QLabel("Iteration: -")
        progress_layout.addWidget(self.iteration_label)

        self.details_label = QLabel("")
        self.details_label.setWordWrap(True)
        progress_layout.addWidget(self.details_label)

        progress_layout.addStretch()
        splitter.addWidget(progress_container)

        log_container = QWidget()
        log_layout = QVBoxLayout(log_container)
        log_layout.setContentsMargins(10, 10, 10, 10)

        log_header_layout = QHBoxLayout()
        log_header = QLabel("Logs")
        log_header.setStyleSheet("font-size: 14px; font-weight: bold; padding: 5px;")
        log_header_layout.addWidget(log_header)
        log_header_layout.addStretch()

        self.clear_log_btn = QPushButton("Clear")
        self.clear_log_btn.setMaximumWidth(60)
        self.clear_log_btn.clicked.connect(self.clear_log)
        log_header_layout.addWidget(self.clear_log_btn)
        log_layout.addLayout(log_header_layout)

        self.log_area = QTextEdit()
        self.log_area.setReadOnly(True)
        self.log_area.setStyleSheet("""
            QTextEdit {
                background-color: #1e1e1e;
                color: #d4d4d4;
                font-family: 'Consolas', 'Monaco', monospace;
                font-size: 12px;
                border: 1px solid #3c3c3c;
                border-radius: 4px;
                padding: 5px;
            }
        """)
        log_layout.addWidget(self.log_area)

        splitter.addWidget(log_container)

        splitter.setSizes([200, 300])
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)

        layout.addWidget(splitter)

    def update_progress(self, value: int, status: str = ""):
        """Update progress bar."""
        self.progress_bar.setValue(value)
        if status:
            self.status_label.setText(status)

    def update_state(self, state: str, message: str = ""):
        """Update state indicator."""
        state_colors = {
            "IDLE": "#666",
            "PARSING": "#2196F3",
            "GENERATING": "#9C27B0",
            "COMPILING": "#FF9800",
            "TESTING": "#00BCD4",
            "ANALYZING": "#3F51B5",
            "FIXING": "#F44336",
            "OPTIMIZING": "#4CAF50",
            "COMPLETED": "#4CAF50",
            "FAILED": "#F44336",
            "PAUSED": "#FF9800",
        }
        color = state_colors.get(state, "#666")
        self.state_label.setText(f"Status: {state}")
        self.state_label.setStyleSheet(f"font-weight: bold; color: {color};")

        if message:
            self.status_label.setText(message)

    def update_coverage(self, coverage: float, target: float, source: str = "jacoco", confidence: float = 1.0):
        """Update coverage display with source information.
        
        Args:
            coverage: Current coverage percentage
            target: Target coverage percentage
            source: Coverage source ("jacoco" or "llm_estimated")
            confidence: Confidence level for LLM estimation
        """
        if source == "llm_estimated":
            text = f"Coverage: {coverage:.1%} (LLM估算, 置信度: {confidence:.0%}) / Target: {target:.1%}"
        else:
            text = f"Coverage: {coverage:.1%} / Target: {target:.1%}"
        
        self.coverage_label.setText(text)

        if coverage >= target:
            self.coverage_label.setStyleSheet("color: #4CAF50; font-weight: bold;")
        elif coverage >= target * 0.8:
            self.coverage_label.setStyleSheet("color: #FF9800; font-weight: bold;")
        else:
            self.coverage_label.setStyleSheet("color: #F44336; font-weight: bold;")

    def update_iteration(self, current: int, max_iter: int):
        """Update iteration display."""
        self.iteration_label.setText(f"Iteration: {current} / {max_iter}")

    def update_details(self, details: str):
        """Update details text."""
        self.details_label.setText(details)

    def add_log(self, message: str, level: str = "INFO"):
        """Add log message with color coding.

        Args:
            message: Log message
            level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        """
        timestamp = datetime.now().strftime("%H:%M:%S")

        colors = {
            "DEBUG": "#808080",
            "INFO": "#4FC1FF",
            "WARNING": "#FFCC00",
            "ERROR": "#FF6B6B",
            "CRITICAL": "#FF0000",
        }
        color = colors.get(level.upper(), "#d4d4d4")

        formatted_message = f"[{timestamp}] [{level.upper()}] {message}"

        self.log_area.append(f'<span style="color: {color};">{formatted_message}</span>')

        scrollbar = self.log_area.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

    def clear_log(self):
        """Clear log area."""
        self.log_area.clear()
