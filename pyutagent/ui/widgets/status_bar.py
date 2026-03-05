"""Enhanced status bar widget."""

from typing import Optional

from PyQt6.QtWidgets import QWidget, QHBoxLayout, QLabel, QProgressBar, QFrame
from PyQt6.QtCore import Qt


class StatusBar(QFrame):
    """Enhanced status bar with multiple sections.
    
    Sections:
    - Left: Project info
    - Center: Agent status
    - Right: Progress and LLM status
    """
    
    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setObjectName("statusBar")
        self.setFixedHeight(28)
        
        self.setup_ui()
        
    def setup_ui(self):
        """Setup the status bar UI."""
        layout = QHBoxLayout(self)
        layout.setContentsMargins(10, 2, 10, 2)
        layout.setSpacing(20)
        
        # Left section: Project info
        self._project_label = QLabel("📁 No project")
        self._project_label.setStyleSheet("color: #757575;")
        layout.addWidget(self._project_label)
        
        # Center section: Agent status
        self._agent_label = QLabel("🤖 Ready")
        self._agent_label.setStyleSheet("color: #2196F3;")
        layout.addWidget(self._agent_label)
        
        layout.addStretch()
        
        # Right section: Progress bar (hidden by default)
        self._progress_bar = QProgressBar()
        self._progress_bar.setFixedWidth(150)
        self._progress_bar.setMaximumHeight(16)
        self._progress_bar.setRange(0, 100)
        self._progress_bar.setValue(0)
        self._progress_bar.setTextVisible(True)
        self._progress_bar.hide()
        layout.addWidget(self._progress_bar)
        
        # Progress text
        self._progress_label = QLabel("")
        self._progress_label.setStyleSheet("color: #757575;")
        layout.addWidget(self._progress_label)
        
        # LLM connection status
        self._llm_label = QLabel("🔴 LLM: Not connected")
        self._llm_label.setStyleSheet("color: #F44336;")
        layout.addWidget(self._llm_label)
        
    def set_project(self, text: str, connected: bool = True):
        """Set project status.
        
        Args:
            text: Project text to display
            connected: Whether project is connected/loaded
        """
        self._project_label.setText(text)
        if connected:
            self._project_label.setStyleSheet("color: #4CAF50;")
        else:
            self._project_label.setStyleSheet("color: #757575;")
            
    def set_agent_status(self, text: str, status_type: str = "info"):
        """Set agent status.
        
        Args:
            text: Status text
            status_type: One of "info", "success", "warning", "error", "busy"
        """
        self._agent_label.setText(text)
        
        colors = {
            "info": "#2196F3",
            "success": "#4CAF50",
            "warning": "#FF9800",
            "error": "#F44336",
            "busy": "#9C27B0"
        }
        color = colors.get(status_type, "#2196F3")
        self._agent_label.setStyleSheet(f"color: {color};")
        
    def set_progress(self, value: int, text: str = ""):
        """Set progress value.
        
        Args:
            value: Progress percentage (0-100)
            text: Optional progress text
        """
        if value > 0 and value < 100:
            self._progress_bar.show()
            self._progress_bar.setValue(value)
            if text:
                self._progress_label.setText(text)
                self._progress_label.show()
        else:
            self._progress_bar.hide()
            self._progress_label.hide()
            self._progress_bar.setValue(0)
            self._progress_label.setText("")
            
    def set_llm_status(self, text: str, connected: bool = False):
        """Set LLM connection status.
        
        Args:
            text: Status text
            connected: Whether LLM is connected
        """
        self._llm_label.setText(text)
        if connected:
            self._llm_label.setStyleSheet("color: #4CAF50;")
        else:
            self._llm_label.setStyleSheet("color: #F44336;")
