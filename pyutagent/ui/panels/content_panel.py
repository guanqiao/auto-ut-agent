"""Content panel for displaying code editor, diff viewer, and other content."""

import logging
from typing import Optional

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QStackedWidget, QFrame, QTabWidget, QTabBar
)
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QIcon

from ..editor import CodeEditorWidget, DiffViewer

logger = logging.getLogger(__name__)


class ContentPanel(QWidget):
    """Content panel for displaying editors and viewers.
    
    Features:
    - Tabbed interface for multiple files
    - Code editor with syntax highlighting
    - Diff viewer
    - Welcome screen
    """
    
    # Signals
    file_opened = pyqtSignal(str)  # file_path
    file_closed = pyqtSignal(str)  # file_path
    file_modified = pyqtSignal(str, bool)  # file_path, is_modified
    
    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._open_files: dict = {}  # file_path -> tab_index
        self._current_file: Optional[str] = None
        
        self.setup_ui()
        
    def setup_ui(self):
        """Setup the content panel UI."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        
        # Tab widget for multiple files
        self._tabs = QTabWidget()
        self._tabs.setTabsClosable(True)
        self._tabs.setMovable(True)
        self._tabs.tabCloseRequested.connect(self._on_tab_close)
        self._tabs.currentChanged.connect(self._on_tab_changed)
        
        # Welcome tab
        self._welcome_widget = self._create_welcome_widget()
        self._tabs.addTab(self._welcome_widget, "🏠 Welcome")
        
        layout.addWidget(self._tabs, stretch=1)
        
    def _create_welcome_widget(self) -> QWidget:
        """Create the welcome screen widget."""
        widget = QWidget()
        widget.setStyleSheet("""
            QWidget {
                background-color: #1E1E1E;
                color: #D4D4D4;
            }
        """)
        
        layout = QVBoxLayout(widget)
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        # Logo/Title
        title = QLabel("🚀 PyUT Agent")
        title.setStyleSheet("font-size: 32px; font-weight: bold; color: #4CAF50;")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title)
        
        # Subtitle
        subtitle = QLabel("AI-Powered Coding Assistant")
        subtitle.setStyleSheet("font-size: 16px; color: #858585;")
        subtitle.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(subtitle)
        
        layout.addSpacing(40)
        
        # Quick actions
        actions_label = QLabel("Quick Actions:")
        actions_label.setStyleSheet("font-size: 14px; color: #CCCCCC;")
        actions_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(actions_label)
        
        layout.addSpacing(20)
        
        # Action buttons
        actions_layout = QHBoxLayout()
        actions_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        btn_open = QPushButton("📁 Open Project")
        btn_open.setStyleSheet("""
            QPushButton {
                background-color: #0E639C;
                color: white;
                border: none;
                padding: 10px 20px;
                border-radius: 4px;
                font-size: 14px;
            }
            QPushButton:hover {
                background-color: #1177BB;
            }
        """)
        btn_open.clicked.connect(self._on_open_project)
        actions_layout.addWidget(btn_open)
        
        layout.addLayout(actions_layout)
        
        layout.addSpacing(40)
        
        # Tips
        tips_label = QLabel(
            "💡 Tips:\n"
            "• Select a file from the sidebar to start editing\n"
            "• Use the Agent panel on the right to chat with AI\n"
            "• Right-click on code for AI actions\n"
            "• Use keyboard shortcuts for quick navigation"
        )
        tips_label.setStyleSheet("font-size: 12px; color: #858585;")
        tips_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(tips_label)
        
        layout.addStretch()
        
        return widget
        
    def open_file(self, file_path: str) -> bool:
        """Open a file in the editor.
        
        Args:
            file_path: Path to the file
            
        Returns:
            True if opened successfully
        """
        # Check if already open
        if file_path in self._open_files:
            index = self._open_files[file_path]
            self._tabs.setCurrentIndex(index)
            return True
        
        # Create editor
        editor = CodeEditorWidget()
        
        if not editor.load_file(file_path):
            return False
        
        # Add tab
        from pathlib import Path
        file_name = Path(file_path).name
        index = self._tabs.addTab(editor, f"📄 {file_name}")
        self._open_files[file_path] = index
        
        # Connect signals
        editor.file_loaded.connect(lambda: self.file_opened.emit(file_path))
        
        # Switch to the new tab
        self._tabs.setCurrentIndex(index)
        self._current_file = file_path
        
        logger.info(f"Opened file: {file_path}")
        return True
        
    def show_diff(self, old_content: str, new_content: str, title: str = ""):
        """Show a diff view.
        
        Args:
            old_content: Original content
            new_content: Modified content
            title: Optional title
        """
        diff_viewer = DiffViewer()
        diff_viewer.set_diff(old_content, new_content, title)
        diff_viewer.change_accepted.connect(self._on_diff_accepted)
        diff_viewer.change_rejected.connect(self._on_diff_rejected)
        
        diff_title = title or "Diff"
        index = self._tabs.addTab(diff_viewer, f"🔍 {diff_title}")
        self._tabs.setCurrentIndex(index)
        
    def _on_tab_close(self, index: int):
        """Handle tab close request."""
        # Don't close welcome tab
        if index == 0:
            return
        
        widget = self._tabs.widget(index)
        
        # Find file path
        file_path = None
        for path, idx in list(self._open_files.items()):
            if idx == index:
                file_path = path
                break
        
        # Remove from tracking
        if file_path:
            del self._open_files[file_path]
            self.file_closed.emit(file_path)
        
        # Remove tab
        self._tabs.removeTab(index)
        
        # Update indices
        self._update_tab_indices()
        
    def _on_tab_changed(self, index: int):
        """Handle tab change."""
        if index == 0:
            self._current_file = None
            return
        
        # Find file path for current index
        for path, idx in self._open_files.items():
            if idx == index:
                self._current_file = path
                break
                
    def _update_tab_indices(self):
        """Update tab indices after tab removal."""
        new_mapping = {}
        for i in range(1, self._tabs.count()):  # Skip welcome tab
            widget = self._tabs.widget(i)
            # Find the file path for this widget
            for path, old_index in self._open_files.items():
                # This is a simplification - in practice you'd need to track widget references
                pass
        
        # Rebuild mapping
        self._open_files.clear()
        for i in range(1, self._tabs.count()):
            widget = self._tabs.widget(i)
            if isinstance(widget, CodeEditorWidget):
                file_path = widget.get_editor().get_file_path()
                if file_path:
                    self._open_files[file_path] = i
                    
    def _on_open_project(self):
        """Handle open project button click."""
        logger.info("Open project requested from welcome screen")
        # TODO: Emit signal to main window
        
    def _on_diff_accepted(self):
        """Handle diff accepted."""
        logger.info("Diff changes accepted")
        # TODO: Apply changes to file
        
    def _on_diff_rejected(self):
        """Handle diff rejected."""
        logger.info("Diff changes rejected")
        # Close the diff tab
        current_index = self._tabs.currentIndex()
        if current_index > 0:  # Don't close welcome
            self._tabs.removeTab(current_index)
            
    def get_current_editor(self) -> Optional[CodeEditorWidget]:
        """Get the current editor widget."""
        current = self._tabs.currentWidget()
        if isinstance(current, CodeEditorWidget):
            return current
        return None
        
    def get_current_file(self) -> Optional[str]:
        """Get the current file path."""
        return self._current_file
        
    def close_all_files(self):
        """Close all open files."""
        # Remove all tabs except welcome
        while self._tabs.count() > 1:
            self._tabs.removeTab(1)
        
        self._open_files.clear()
        self._current_file = None
        
    def has_unsaved_changes(self) -> bool:
        """Check if any open files have unsaved changes."""
        # TODO: Implement modification tracking
        return False
