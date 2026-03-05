"""Sidebar panel containing file tree and project navigation."""

import logging
from typing import Optional

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QTabWidget, QFrame, QSizePolicy
)
from PyQt6.QtCore import Qt, pyqtSignal

from ..widgets import FileTree, SearchBox

logger = logging.getLogger(__name__)


class SidebarPanel(QWidget):
    """Sidebar panel with file tree and project tools.
    
    Features:
    - File tree with multi-language support
    - Project search
    - Quick actions
    """
    
    # Signals
    file_selected = pyqtSignal(str)  # file_path
    file_activated = pyqtSignal(str)  # file_path (double-click)
    search_requested = pyqtSignal(str)  # search_text
    
    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        
        self.setup_ui()
        
    def setup_ui(self):
        """Setup the sidebar panel UI."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        
        # Header
        header = QFrame()
        header.setStyleSheet("""
            QFrame {
                background-color: #252526;
                border-bottom: 1px solid #3C3C3C;
            }
        """)
        header.setFixedHeight(40)
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(12, 8, 12, 8)
        
        title = QLabel("📁 Explorer")
        title.setStyleSheet("color: #CCCCCC; font-weight: bold;")
        header_layout.addWidget(title)
        
        header_layout.addStretch()
        
        # Refresh button
        self._btn_refresh = QPushButton("🔄")
        self._btn_refresh.setFixedSize(24, 24)
        self._btn_refresh.setToolTip("Refresh")
        self._btn_refresh.clicked.connect(self._on_refresh)
        header_layout.addWidget(self._btn_refresh)
        
        # Collapse/Expand button
        self._btn_collapse = QPushButton("📂")
        self._btn_collapse.setFixedSize(24, 24)
        self._btn_collapse.setToolTip("Collapse All")
        self._btn_collapse.clicked.connect(self._on_collapse)
        header_layout.addWidget(self._btn_collapse)
        
        layout.addWidget(header)
        
        # Search box
        self._search_box = SearchBox(placeholder="Search files...")
        self._search_box.search_requested.connect(self._on_search)
        layout.addWidget(self._search_box)
        
        # File tree
        self._file_tree = FileTree()
        self._file_tree.file_selected.connect(self.file_selected.emit)
        self._file_tree.file_activated.connect(self.file_activated.emit)
        layout.addWidget(self._file_tree, stretch=1)
        
        # Quick actions footer
        footer = QFrame()
        footer.setStyleSheet("""
            QFrame {
                background-color: #252526;
                border-top: 1px solid #3C3C3C;
            }
        """)
        footer.setFixedHeight(40)
        footer_layout = QHBoxLayout(footer)
        footer_layout.setContentsMargins(8, 4, 8, 4)
        
        # New file button
        self._btn_new_file = QPushButton("📄 New")
        self._btn_new_file.setToolTip("New File")
        self._btn_new_file.clicked.connect(self._on_new_file)
        footer_layout.addWidget(self._btn_new_file)
        
        # New folder button
        self._btn_new_folder = QPushButton("📁 Folder")
        self._btn_new_folder.setToolTip("New Folder")
        self._btn_new_folder.clicked.connect(self._on_new_folder)
        footer_layout.addWidget(self._btn_new_folder)
        
        footer_layout.addStretch()
        
        layout.addWidget(footer)
        
    def load_project(self, project_path: str):
        """Load a project into the file tree.
        
        Args:
            project_path: Path to the project directory
        """
        self._file_tree.load_project(project_path)
        logger.info(f"Sidebar loaded project: {project_path}")
        
    def _on_refresh(self):
        """Handle refresh button click."""
        self._file_tree.refresh()
        logger.debug("File tree refreshed")
        
    def _on_collapse(self):
        """Handle collapse button click."""
        self._file_tree.collapse_all()
        logger.debug("File tree collapsed")
        
    def _on_search(self, text: str):
        """Handle search request."""
        self.search_requested.emit(text)
        logger.debug(f"Search requested: {text}")
        
    def _on_new_file(self):
        """Handle new file button click."""
        logger.info("New file requested")
        # TODO: Implement new file creation
        
    def _on_new_folder(self):
        """Handle new folder button click."""
        logger.info("New folder requested")
        # TODO: Implement new folder creation
        
    def get_file_tree(self) -> FileTree:
        """Get the file tree widget."""
        return self._file_tree
        
    def select_file(self, file_path: str):
        """Select a file in the tree."""
        self._file_tree.select_path(file_path)
