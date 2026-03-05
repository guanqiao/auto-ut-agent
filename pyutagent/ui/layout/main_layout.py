"""Main three-panel layout for PyUT Agent UI.

Layout structure:
┌─────────────────────────────────────────────────────────────┐
│  Toolbar (Quick Actions + Context Selection)               │
├──────────┬──────────────────────────────┬───────────────────┤
│          │                              │                   │
│  Sidebar │      Main Content Area       │   Agent Panel     │
│  (Files) │      (Editor/Preview)        │   (Chat/Flow)     │
│          │                              │                   │
├──────────┴──────────────────────────────┴───────────────────┤
│  Status Bar (Project Info + Agent Status + Progress)       │
└─────────────────────────────────────────────────────────────┘
"""

import logging
from typing import Optional, Dict, Any

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QMainWindow,
    QToolBar, QLabel, QPushButton, QFrame, QSizePolicy
)
from PyQt6.QtCore import Qt, pyqtSignal, QSize
from PyQt6.QtGui import QAction, QIcon

from .collapsible_splitter import CollapsibleSplitter

logger = logging.getLogger(__name__)


class MainLayout(QWidget):
    """Main three-panel layout widget.
    
    Provides a standardized layout with:
    - Left sidebar (file tree, project navigation)
    - Center content area (code editor, diff viewer, terminal)
    - Right agent panel (chat, agent flow, context management)
    - Top toolbar (quick actions, context selection)
    - Bottom status bar (project info, agent status)
    """
    
    # Signals
    panel_collapsed = pyqtSignal(str, bool)  # panel_name, is_collapsed
    layout_changed = pyqtSignal(str)  # layout_mode
    
    # Panel names
    SIDEBAR_PANEL = "sidebar"
    CONTENT_PANEL = "content"
    AGENT_PANEL = "agent"
    
    # Layout modes
    MODE_DEFAULT = "default"      # All panels visible
    MODE_FOCUS_EDITOR = "focus_editor"  # Content panel expanded
    MODE_FOCUS_AGENT = "focus_agent"    # Agent panel expanded
    MODE_COMPACT = "compact"      # Sidebar collapsed
    
    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._current_mode = self.MODE_DEFAULT
        self._panels: Dict[str, QWidget] = {}
        self._panel_sizes: Dict[str, int] = {
            self.SIDEBAR_PANEL: 250,
            self.CONTENT_PANEL: 700,
            self.AGENT_PANEL: 400
        }
        
        self.setup_ui()
        
    def setup_ui(self):
        """Setup the main layout UI."""
        self.setObjectName("mainLayout")
        
        # Main vertical layout
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        
        # Top toolbar
        self._toolbar = self._create_toolbar()
        main_layout.addWidget(self._toolbar)
        
        # Main horizontal splitter (three panels)
        self._main_splitter = CollapsibleSplitter(Qt.Orientation.Horizontal, self)
        self._main_splitter.setObjectName("mainSplitter")
        self._main_splitter.setHandleWidth(6)
        
        # Create panels
        self._sidebar_container = self._create_panel_container(self.SIDEBAR_PANEL)
        self._content_container = self._create_panel_container(self.CONTENT_PANEL)
        self._agent_container = self._create_panel_container(self.AGENT_PANEL)
        
        # Add panels to splitter
        self._main_splitter.addWidget(
            self._sidebar_container, 
            collapsible=True, 
            min_size=150, 
            collapsed_size=40
        )
        self._main_splitter.addWidget(
            self._content_container, 
            collapsible=False, 
            min_size=300
        )
        self._main_splitter.addWidget(
            self._agent_container, 
            collapsible=True, 
            min_size=250, 
            collapsed_size=40
        )
        
        # Set initial sizes
        self._main_splitter.setSizes([
            self._panel_sizes[self.SIDEBAR_PANEL],
            self._panel_sizes[self.CONTENT_PANEL],
            self._panel_sizes[self.AGENT_PANEL]
        ])
        
        # Connect signals
        self._main_splitter.panel_collapsed.connect(self._on_panel_collapsed)
        self._main_splitter.splitterMoved.connect(self._on_splitter_moved)
        
        main_layout.addWidget(self._main_splitter, stretch=1)
        
        # Status bar container
        self._status_container = self._create_status_container()
        main_layout.addWidget(self._status_container)
        
    def _create_toolbar(self) -> QToolBar:
        """Create the top toolbar."""
        toolbar = QToolBar("Main Toolbar", self)
        toolbar.setMovable(False)
        toolbar.setFloatable(False)
        toolbar.setIconSize(QSize(20, 20))
        
        # Layout mode buttons
        toolbar.addWidget(QLabel("Layout: "))
        
        self._btn_default = QPushButton("Default")
        self._btn_default.setCheckable(True)
        self._btn_default.setChecked(True)
        self._btn_default.clicked.connect(lambda: self.set_layout_mode(self.MODE_DEFAULT))
        toolbar.addWidget(self._btn_default)
        
        self._btn_focus_editor = QPushButton("Focus Editor")
        self._btn_focus_editor.setCheckable(True)
        self._btn_focus_editor.clicked.connect(lambda: self.set_layout_mode(self.MODE_FOCUS_EDITOR))
        toolbar.addWidget(self._btn_focus_editor)
        
        self._btn_focus_agent = QPushButton("Focus Agent")
        self._btn_focus_agent.setCheckable(True)
        self._btn_focus_agent.clicked.connect(lambda: self.set_layout_mode(self.MODE_FOCUS_AGENT))
        toolbar.addWidget(self._btn_focus_agent)
        
        toolbar.addSeparator()
        
        # Toggle panel buttons
        self._btn_toggle_sidebar = QPushButton("Toggle Sidebar")
        self._btn_toggle_sidebar.clicked.connect(lambda: self.toggle_panel(self.SIDEBAR_PANEL))
        toolbar.addWidget(self._btn_toggle_sidebar)
        
        self._btn_toggle_agent = QPushButton("Toggle Agent")
        self._btn_toggle_agent.clicked.connect(lambda: self.toggle_panel(self.AGENT_PANEL))
        toolbar.addWidget(self._btn_toggle_agent)
        
        toolbar.addSeparator()
        
        # Context label
        self._context_label = QLabel("No project loaded")
        toolbar.addWidget(self._context_label)
        
        toolbar.addStretch()
        
        return toolbar
        
    def _create_panel_container(self, panel_name: str) -> QFrame:
        """Create a container for a panel.
        
        Args:
            panel_name: Name of the panel (sidebar, content, agent)
            
        Returns:
            QFrame container widget
        """
        container = QFrame(self)
        container.setObjectName(f"{panel_name}Container")
        container.setFrameShape(QFrame.Shape.NoFrame)
        
        layout = QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        
        # Header
        header = QFrame(container)
        header.setObjectName(f"{panel_name}Header")
        header.setFixedHeight(32)
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(8, 4, 8, 4)
        
        title_label = QLabel(panel_name.capitalize())
        title_label.setObjectName(f"{panel_name}Title")
        header_layout.addWidget(title_label)
        header_layout.addStretch()
        
        layout.addWidget(header)
        
        # Content area (placeholder)
        content = QFrame(container)
        content.setObjectName(f"{panel_name}Content")
        layout.addWidget(content, stretch=1)
        
        # Store reference
        self._panels[panel_name] = content
        
        return container
        
    def _create_status_container(self) -> QFrame:
        """Create the status bar container."""
        container = QFrame(self)
        container.setObjectName("statusContainer")
        container.setFixedHeight(28)
        
        layout = QHBoxLayout(container)
        layout.setContentsMargins(10, 2, 10, 2)
        layout.setSpacing(20)
        
        # Left: Project info
        self._status_project = QLabel("📁 No project")
        layout.addWidget(self._status_project)
        
        # Center: Agent status
        self._status_agent = QLabel("🤖 Ready")
        layout.addWidget(self._status_agent)
        
        layout.addStretch()
        
        # Right: Progress
        self._status_progress = QLabel("")
        layout.addWidget(self._status_progress)
        
        # LLM connection status
        self._status_llm = QLabel("🔴 LLM: Not connected")
        layout.addWidget(self._status_llm)
        
        return container
        
    def set_panel_widget(self, panel_name: str, widget: QWidget):
        """Set the widget for a panel.
        
        Args:
            panel_name: Name of the panel (sidebar, content, agent)
            widget: Widget to set
        """
        if panel_name not in self._panels:
            logger.warning(f"Unknown panel: {panel_name}")
            return
            
        content = self._panels[panel_name]
        layout = content.layout()
        
        # Clear existing widgets
        if layout is None:
            layout = QVBoxLayout(content)
            layout.setContentsMargins(0, 0, 0, 0)
            layout.setSpacing(0)
        else:
            while layout.count():
                item = layout.takeAt(0)
                if item.widget():
                    item.widget().setParent(None)
                    
        layout.addWidget(widget, stretch=1)
        logger.debug(f"Set widget for panel: {panel_name}")
        
    def get_panel_widget(self, panel_name: str) -> Optional[QWidget]:
        """Get the widget for a panel.
        
        Args:
            panel_name: Name of the panel
            
        Returns:
            The widget or None
        """
        return self._panels.get(panel_name)
        
    def set_layout_mode(self, mode: str):
        """Set the layout mode.
        
        Args:
            mode: One of MODE_DEFAULT, MODE_FOCUS_EDITOR, MODE_FOCUS_AGENT, MODE_COMPACT
        """
        if mode not in [self.MODE_DEFAULT, self.MODE_FOCUS_EDITOR, 
                       self.MODE_FOCUS_AGENT, self.MODE_COMPACT]:
            logger.warning(f"Unknown layout mode: {mode}")
            return
            
        self._current_mode = mode
        
        # Update button states
        self._btn_default.setChecked(mode == self.MODE_DEFAULT)
        self._btn_focus_editor.setChecked(mode == self.MODE_FOCUS_EDITOR)
        self._btn_focus_agent.setChecked(mode == self.MODE_FOCUS_AGENT)
        
        # Apply layout
        total_width = self._main_splitter.width()
        
        if mode == self.MODE_DEFAULT:
            sizes = [
                int(total_width * 0.2),   # Sidebar: 20%
                int(total_width * 0.5),   # Content: 50%
                int(total_width * 0.3)    # Agent: 30%
            ]
        elif mode == self.MODE_FOCUS_EDITOR:
            sizes = [
                50,   # Sidebar: collapsed
                int(total_width * 0.7),   # Content: 70%
                int(total_width * 0.25)   # Agent: 25%
            ]
        elif mode == self.MODE_FOCUS_AGENT:
            sizes = [
                50,   # Sidebar: collapsed
                int(total_width * 0.4),   # Content: 40%
                int(total_width * 0.55)   # Agent: 55%
            ]
        elif mode == self.MODE_COMPACT:
            sizes = [
                50,   # Sidebar: collapsed
                int(total_width * 0.6),   # Content: 60%
                int(total_width * 0.35)   # Agent: 35%
            ]
            
        self._main_splitter.setSizes(sizes)
        self.layout_changed.emit(mode)
        logger.debug(f"Layout mode changed to: {mode}")
        
    def toggle_panel(self, panel_name: str):
        """Toggle a panel's visibility.
        
        Args:
            panel_name: Name of the panel
        """
        panel_map = {
            self.SIDEBAR_PANEL: 0,
            self.CONTENT_PANEL: 1,
            self.AGENT_PANEL: 2
        }
        
        if panel_name not in panel_map:
            return
            
        index = panel_map[panel_name]
        self._main_splitter.toggle_panel(index)
        
    def collapse_panel(self, panel_name: str):
        """Collapse a panel.
        
        Args:
            panel_name: Name of the panel
        """
        panel_map = {
            self.SIDEBAR_PANEL: 0,
            self.AGENT_PANEL: 2
        }
        
        if panel_name in panel_map:
            self._main_splitter.collapse_panel(panel_map[panel_name])
            
    def expand_panel(self, panel_name: str):
        """Expand a panel.
        
        Args:
            panel_name: Name of the panel
        """
        panel_map = {
            self.SIDEBAR_PANEL: 0,
            self.AGENT_PANEL: 2
        }
        
        if panel_name in panel_map:
            self._main_splitter.expand_panel(panel_map[panel_name])
            
    def is_panel_collapsed(self, panel_name: str) -> bool:
        """Check if a panel is collapsed.
        
        Args:
            panel_name: Name of the panel
            
        Returns:
            True if collapsed
        """
        panel_map = {
            self.SIDEBAR_PANEL: 0,
            self.AGENT_PANEL: 2
        }
        
        if panel_name in panel_map:
            return self._main_splitter.is_panel_collapsed(panel_map[panel_name])
        return False
        
    def _on_panel_collapsed(self, index: int, is_collapsed: bool):
        """Handle panel collapse/expand."""
        panel_names = [self.SIDEBAR_PANEL, self.CONTENT_PANEL, self.AGENT_PANEL]
        if index < len(panel_names):
            self.panel_collapsed.emit(panel_names[index], is_collapsed)
            
    def _on_splitter_moved(self, pos: int, index: int):
        """Handle splitter movement."""
        # Save current sizes
        sizes = self._main_splitter.sizes()
        panel_names = [self.SIDEBAR_PANEL, self.CONTENT_PANEL, self.AGENT_PANEL]
        for i, name in enumerate(panel_names):
            if i < len(sizes):
                self._panel_sizes[name] = sizes[i]
                
    def set_context_label(self, text: str):
        """Set the context label text."""
        self._context_label.setText(text)
        
    def set_status_project(self, text: str):
        """Set the project status."""
        self._status_project.setText(text)
        
    def set_status_agent(self, text: str):
        """Set the agent status."""
        self._status_agent.setText(text)
        
    def set_status_progress(self, text: str):
        """Set the progress status."""
        self._status_progress.setText(text)
        
    def set_status_llm(self, text: str, connected: bool = False):
        """Set the LLM connection status."""
        self._status_llm.setText(text)
        if connected:
            self._status_llm.setStyleSheet("color: #4CAF50;")
        else:
            self._status_llm.setStyleSheet("color: #F44336;")
            
    def save_state(self) -> Dict[str, Any]:
        """Save the layout state."""
        return {
            'mode': self._current_mode,
            'splitter_state': self._main_splitter.save_state(),
            'panel_sizes': self._panel_sizes.copy()
        }
        
    def restore_state(self, state: Dict[str, Any]):
        """Restore the layout state."""
        if 'mode' in state:
            self.set_layout_mode(state['mode'])
        if 'splitter_state' in state:
            self._main_splitter.restore_state(state['splitter_state'])
        if 'panel_sizes' in state:
            self._panel_sizes.update(state['panel_sizes'])
