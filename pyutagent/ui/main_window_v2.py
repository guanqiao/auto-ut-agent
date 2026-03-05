"""Main window v2 with new three-panel layout.

This is the refactored main window using the new layout system.
It maintains backward compatibility with the existing agent and config systems.
"""

import logging
from pathlib import Path
from typing import Optional

from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QMessageBox, QFileDialog, QApplication
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QAction, QKeySequence

from .layout import MainLayout
from .widgets import FileTree, StatusBar
from .chat_widget import ChatWidget
from .styles import get_style_manager
from .components import get_notification_manager

# Import existing components (backward compatibility)
from ..core.config import (
    LLMConfigCollection, AiderConfig, AppState,
    load_llm_config, save_llm_config,
    load_aider_config, save_aider_config,
    load_app_state, save_app_state, get_settings
)
from ..llm.client import LLMClient

logger = logging.getLogger(__name__)


class MainWindowV2(QMainWindow):
    """Refactored main window with three-panel layout.
    
    Features:
    - Three-panel layout (sidebar, content, agent panel)
    - Collapsible panels
    - Multiple layout modes
    - Multi-language project support
    """
    
    project_opened = pyqtSignal(str)
    generate_requested = pyqtSignal(str)
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("PyUT Agent - AI Coding Assistant")
        self.setGeometry(100, 100, 1600, 900)
        
        # Initialize state
        self.current_project: str = ""
        self.config_collection: LLMConfigCollection = load_llm_config()
        self.aider_config: AiderConfig = load_aider_config()
        self.app_state: AppState = load_app_state()
        self.llm_client: Optional[LLMClient] = None
        
        # Initialize UI components
        self._style_manager = get_style_manager()
        self._notification_manager = get_notification_manager()
        
        self.setup_ui()
        self.setup_menu()
        self.setup_shortcuts()
        self.apply_styles()
        self.setup_llm_client()
        
    def setup_ui(self):
        """Setup the main UI."""
        # Create central widget with new layout
        self._main_layout = MainLayout(self)
        self.setCentralWidget(self._main_layout)
        
        # Setup sidebar (file tree)
        self._file_tree = FileTree()
        self._file_tree.file_selected.connect(self.on_file_selected)
        self._file_tree.file_activated.connect(self.on_file_activated)
        self._main_layout.set_panel_widget(MainLayout.SIDEBAR_PANEL, self._file_tree)
        
        # Setup agent panel (chat)
        self._chat_widget = ChatWidget()
        self._chat_widget.message_sent.connect(self.on_message_sent)
        self._chat_widget.generate_clicked.connect(self.on_generate_tests)
        self._main_layout.set_panel_widget(MainLayout.AGENT_PANEL, self._chat_widget)
        
        # Connect layout signals
        self._main_layout.panel_collapsed.connect(self.on_panel_collapsed)
        
    def setup_menu(self):
        """Setup menu bar."""
        menubar = self.menuBar()
        
        # File menu
        file_menu = menubar.addMenu("&File")
        
        open_action = QAction("&Open Project...", self)
        open_action.setShortcut(QKeySequence.StandardKey.Open)
        open_action.triggered.connect(self.on_open_project)
        file_menu.addAction(open_action)
        
        file_menu.addSeparator()
        
        exit_action = QAction("E&xit", self)
        exit_action.setShortcut(QKeySequence.StandardKey.Quit)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)
        
        # View menu
        view_menu = menubar.addMenu("&View")
        
        toggle_sidebar_action = QAction("Toggle &Sidebar", self)
        toggle_sidebar_action.setShortcut("Ctrl+B")
        toggle_sidebar_action.triggered.connect(lambda: self._main_layout.toggle_panel(MainLayout.SIDEBAR_PANEL))
        view_menu.addAction(toggle_sidebar_action)
        
        toggle_agent_action = QAction("Toggle &Agent Panel", self)
        toggle_agent_action.setShortcut("Ctrl+J")
        toggle_agent_action.triggered.connect(lambda: self._main_layout.toggle_panel(MainLayout.AGENT_PANEL))
        view_menu.addAction(toggle_agent_action)
        
        view_menu.addSeparator()
        
        # Layout modes
        default_layout_action = QAction("&Default Layout", self)
        default_layout_action.triggered.connect(lambda: self._main_layout.set_layout_mode(MainLayout.MODE_DEFAULT))
        view_menu.addAction(default_layout_action)
        
        focus_editor_action = QAction("Focus &Editor", self)
        focus_editor_action.triggered.connect(lambda: self._main_layout.set_layout_mode(MainLayout.MODE_FOCUS_EDITOR))
        view_menu.addAction(focus_editor_action)
        
        focus_agent_action = QAction("Focus A&gent", self)
        focus_agent_action.triggered.connect(lambda: self._main_layout.set_layout_mode(MainLayout.MODE_FOCUS_AGENT))
        view_menu.addAction(focus_agent_action)
        
        # Settings menu
        settings_menu = menubar.addMenu("&Settings")
        
        llm_config_action = QAction("&LLM Configuration...", self)
        llm_config_action.triggered.connect(self.on_llm_config)
        settings_menu.addAction(llm_config_action)
        
        # Help menu
        help_menu = menubar.addMenu("&Help")
        
        about_action = QAction("&About", self)
        about_action.triggered.connect(self.on_about)
        help_menu.addAction(about_action)
        
    def setup_shortcuts(self):
        """Setup keyboard shortcuts."""
        # Ctrl+Shift+P for command palette (placeholder)
        command_palette_shortcut = QAction(self)
        command_palette_shortcut.setShortcut("Ctrl+Shift+P")
        command_palette_shortcut.triggered.connect(self.on_command_palette)
        self.addAction(command_palette_shortcut)
        
    def apply_styles(self):
        """Apply theme styles."""
        self._style_manager.apply_stylesheet(self, "main_window")
        
    def setup_llm_client(self):
        """Setup LLM client."""
        try:
            default_config = self.config_collection.get_default_config()
            if default_config:
                self.llm_client = LLMClient.from_config(default_config)
                self._main_layout.set_status_llm(
                    f"🟢 LLM: {default_config.get_display_name()}",
                    connected=True
                )
                self._notification_manager.show_success(
                    f"LLM client initialized: {default_config.get_display_name()}",
                    duration=3000
                )
            else:
                self._main_layout.set_status_llm("🔴 LLM: Not configured")
        except Exception as e:
            logger.exception("Failed to initialize LLM client")
            self._main_layout.set_status_llm("🔴 LLM: Error")
            
    def on_open_project(self):
        """Handle open project action."""
        dir_path = QFileDialog.getExistingDirectory(
            self,
            "Select Project",
            "",
            QFileDialog.Option.ShowDirsOnly
        )
        
        if dir_path:
            self._open_project(dir_path)
            
    def _open_project(self, dir_path: str) -> bool:
        """Open a project by path."""
        try:
            self.current_project = dir_path
            self._file_tree.load_project(dir_path)
            self.project_opened.emit(dir_path)
            
            # Update status
            project_name = Path(dir_path).name
            self._main_layout.set_status_project(f"📁 {project_name}", connected=True)
            self._main_layout.set_context_label(f"Project: {project_name}")
            
            # Save to app state
            self.app_state.add_project(dir_path)
            save_app_state(self.app_state)
            
            # Show notification
            self._notification_manager.show_success(
                f"Project '{project_name}' opened",
                duration=3000
            )
            
            self._chat_widget.add_agent_message(
                f"Project opened: {project_name}\n"
                "Select a file to start working with the AI assistant."
            )
            
            return True
        except Exception as e:
            logger.exception(f"Failed to open project: {dir_path}")
            self._notification_manager.show_error(f"Failed to open project: {e}")
            return False
            
    def on_file_selected(self, file_path: str):
        """Handle file selection."""
        logger.info(f"File selected: {file_path}")
        # TODO: Show file in editor
        
    def on_file_activated(self, file_path: str):
        """Handle file activation (double-click)."""
        logger.info(f"File activated: {file_path}")
        # TODO: Open file in editor
        
    def on_message_sent(self, message: str):
        """Handle user message."""
        logger.info(f"User message: {message}")
        # TODO: Process message with agent
        self._chat_widget.add_agent_message(
            "I'm your AI coding assistant. I can help you with:\n"
            "- Generating tests\n"
            "- Explaining code\n"
            "- Refactoring\n"
            "- And more..."
        )
        
    def on_generate_tests(self):
        """Handle generate tests request."""
        selected_file = self._file_tree.get_selected_path()
        if not selected_file:
            self._chat_widget.add_agent_message(
                "Please select a file first."
            )
            return
            
        logger.info(f"Generate tests for: {selected_file}")
        # TODO: Start test generation
        
    def on_panel_collapsed(self, panel_name: str, is_collapsed: bool):
        """Handle panel collapse/expand."""
        logger.debug(f"Panel {panel_name} collapsed: {is_collapsed}")
        
    def on_llm_config(self):
        """Handle LLM config action."""
        # Import here to avoid circular imports
        from .dialogs.llm_config_dialog import LLMConfigDialog
        from PyQt6.QtWidgets import QDialog
        
        try:
            dialog = LLMConfigDialog(self.config_collection, self.aider_config, self)
            if dialog.exec() == QDialog.DialogCode.Accepted:
                self.config_collection = dialog.get_config_collection()
                self.aider_config = dialog.aider_config
                save_llm_config(self.config_collection)
                self.setup_llm_client()
        except Exception as e:
            logger.exception("Failed to open LLM config dialog")
            
    def on_command_palette(self):
        """Show command palette."""
        # TODO: Implement command palette
        logger.info("Command palette requested")
        
    def on_about(self):
        """Show about dialog."""
        QMessageBox.about(
            self,
            "About PyUT Agent",
            "<h2>PyUT Agent</h2>"
            "<p>AI-powered Coding Assistant</p>"
            "<p>Version: 2.0.0</p>"
        )
        
    def closeEvent(self, event):
        """Handle window close."""
        # Save layout state
        # TODO: Save to app state
        event.accept()
