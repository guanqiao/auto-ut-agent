"""Command palette for quick access to all features."""

import logging
from typing import List, Callable, Optional
from dataclasses import dataclass

from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit,
    QListWidget, QListWidgetItem, QAbstractItemView
)
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QKeyEvent

from .styles import get_style_manager

logger = logging.getLogger(__name__)


@dataclass
class Command:
    """A command in the command palette."""
    id: str
    name: str
    description: str
    shortcut: str
    category: str
    callback: Callable
    icon: str = ""


class CommandPalette(QDialog):
    """Command palette dialog for quick access to features."""
    
    command_executed = pyqtSignal(str)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Command Palette")
        self.setMinimumSize(600, 400)
        
        self._style_manager = get_style_manager()
        self._commands: List[Command] = []
        self._filtered_commands: List[Command] = []
        
        self.setup_ui()
        self.setup_commands()
        self.apply_styles()
    
    def setup_ui(self):
        """Setup the UI."""
        self.setWindowFlags(
            Qt.WindowType.Dialog |
            Qt.WindowType.FramelessWindowHint |
            Qt.WindowType.WindowStaysOnTopHint
        )
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(10)
        
        # Search box
        self.search_box = QLineEdit()
        self.search_box.setPlaceholderText("Type a command...")
        self.search_box.textChanged.connect(self.on_search_changed)
        layout.addWidget(self.search_box)
        
        # Results list
        self.results_list = QListWidget()
        self.results_list.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.results_list.itemActivated.connect(self.on_item_activated)
        self.results_list.itemClicked.connect(self.on_item_activated)
        layout.addWidget(self.results_list)
        
        # Hint label
        hint = QLabel("↑↓ to navigate, ↵ to execute, Esc to close")
        hint.setStyleSheet("color: #999; font-size: 11px;")
        hint.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(hint)
        
        # Set focus to search box
        self.search_box.setFocus()
    
    def setup_commands(self):
        """Setup available commands."""
        self._commands = [
            Command(
                "file.open",
                "Open Project",
                "Open a Maven project directory",
                "Ctrl+O",
                "File",
                lambda: self._execute("open_project")
            ),
            Command(
                "file.recent",
                "Open Recent Project",
                "Open a recently used project",
                "",
                "File",
                lambda: self._execute("open_recent")
            ),
            Command(
                "generate.single",
                "Generate Tests",
                "Generate tests for selected file",
                "Ctrl+G",
                "Generate",
                lambda: self._execute("generate_tests")
            ),
            Command(
                "generate.batch",
                "Batch Generate",
                "Generate tests for multiple files",
                "Ctrl+B",
                "Generate",
                lambda: self._execute("batch_generate")
            ),
            Command(
                "view.history",
                "Test History",
                "View test generation history",
                "Ctrl+H",
                "View",
                lambda: self._execute("test_history")
            ),
            Command(
                "view.config",
                "Configuration",
                "Open configuration dialog",
                "Ctrl+,",
                "View",
                lambda: self._execute("config")
            ),
            Command(
                "tools.stats",
                "Project Statistics",
                "Show project statistics",
                "Ctrl+Shift+S",
                "Tools",
                lambda: self._execute("project_stats")
            ),
            Command(
                "settings.llm",
                "LLM Settings",
                "Configure LLM settings",
                "",
                "Settings",
                lambda: self._execute("llm_config")
            ),
            Command(
                "settings.coverage",
                "Coverage Settings",
                "Configure coverage settings",
                "",
                "Settings",
                lambda: self._execute("coverage_config")
            ),
            Command(
                "settings.jdk",
                "JDK Settings",
                "Configure JDK path",
                "",
                "Settings",
                lambda: self._execute("jdk_config")
            ),
            Command(
                "settings.maven",
                "Maven Settings",
                "Configure Maven path",
                "",
                "Settings",
                lambda: self._execute("maven_config")
            ),
            Command(
                "help.shortcuts",
                "Keyboard Shortcuts",
                "Show keyboard shortcuts",
                "Ctrl+/",
                "Help",
                lambda: self._execute("shortcuts")
            ),
            Command(
                "help.about",
                "About",
                "Show about dialog",
                "",
                "Help",
                lambda: self._execute("about")
            ),
            Command(
                "theme.light",
                "Light Theme",
                "Switch to light theme",
                "",
                "Theme",
                lambda: self._execute("theme_light")
            ),
            Command(
                "theme.dark",
                "Dark Theme",
                "Switch to dark theme",
                "",
                "Theme",
                lambda: self._execute("theme_dark")
            ),
        ]
        
        self._filtered_commands = self._commands.copy()
        self.update_results()
    
    def apply_styles(self):
        """Apply theme styles."""
        is_dark = self._style_manager.current_theme == "dark"
        
        bg_color = "#2D2D2D" if is_dark else "#FFFFFF"
        text_color = "#E0E0E0" if is_dark else "#212121"
        border_color = "#3C3C3C" if is_dark else "#E0E0E0"
        
        self.setStyleSheet(f"""
            CommandPalette {{
                background-color: {bg_color};
                border: 1px solid {border_color};
                border-radius: 8px;
            }}
            QLineEdit {{
                background-color: {"#1E1E1E" if is_dark else "#F5F5F5"};
                color: {text_color};
                border: 1px solid {border_color};
                border-radius: 4px;
                padding: 8px;
                font-size: 14px;
            }}
            QListWidget {{
                background-color: {bg_color};
                color: {text_color};
                border: none;
                outline: none;
            }}
            QListWidget::item {{
                padding: 8px;
                border-radius: 4px;
            }}
            QListWidget::item:selected {{
                background-color: {"#094771" if is_dark else "#E3F2FD"};
            }}
            QListWidget::item:hover {{
                background-color: {"#3C3C3C" if is_dark else "#F5F5F5"};
            }}
        """)
    
    def on_search_changed(self, text: str):
        """Handle search text change."""
        text = text.lower().strip()
        
        if not text:
            self._filtered_commands = self._commands.copy()
        else:
            self._filtered_commands = [
                cmd for cmd in self._commands
                if text in cmd.name.lower() or
                   text in cmd.description.lower() or
                   text in cmd.category.lower()
            ]
        
        self.update_results()
    
    def update_results(self):
        """Update the results list."""
        self.results_list.clear()
        
        for cmd in self._filtered_commands[:20]:  # Limit to 20 results
            item = QListWidgetItem()
            item.setData(Qt.ItemDataRole.UserRole, cmd)
            
            # Format display text
            shortcut_text = f" ({cmd.shortcut})" if cmd.shortcut else ""
            display_text = f"{cmd.name}{shortcut_text}\n{cmd.description}"
            item.setText(display_text)
            
            self.results_list.addItem(item)
        
        # Select first item
        if self.results_list.count() > 0:
            self.results_list.setCurrentRow(0)
    
    def on_item_activated(self, item: QListWidgetItem):
        """Handle item activation."""
        cmd = item.data(Qt.ItemDataRole.UserRole)
        if cmd:
            self.execute_command(cmd)
    
    def execute_command(self, cmd: Command):
        """Execute a command."""
        try:
            cmd.callback()
            self.command_executed.emit(cmd.id)
            self.accept()
        except Exception as e:
            logger.error(f"Failed to execute command {cmd.id}: {e}")
    
    def _execute(self, action: str):
        """Execute an action on the parent window."""
        parent = self.parent()
        if not parent:
            return
        
        try:
            if action == "open_project":
                parent.on_open_project()
            elif action == "generate_tests":
                parent.on_generate_tests()
            elif action == "batch_generate":
                parent.on_generate_all_tests()
            elif action == "test_history":
                parent.on_test_history()
            elif action == "config":
                parent.on_llm_config()
            elif action == "project_stats":
                parent.on_project_stats()
            elif action == "llm_config":
                parent.on_llm_config()
            elif action == "coverage_config":
                parent.on_coverage_config()
            elif action == "jdk_config":
                parent.on_jdk_config()
            elif action == "maven_config":
                parent.on_maven_config()
            elif action == "shortcuts":
                parent.on_shortcuts()
            elif action == "about":
                parent.on_about()
            elif action == "theme_light":
                parent.on_theme_changed("light")
            elif action == "theme_dark":
                parent.on_theme_changed("dark")
        except Exception as e:
            logger.error(f"Failed to execute action {action}: {e}")
    
    def keyPressEvent(self, event: QKeyEvent):
        """Handle key press events."""
        if event.key() == Qt.Key.Key_Escape:
            self.reject()
        elif event.key() == Qt.Key.Key_Return or event.key() == Qt.Key.Key_Enter:
            current_item = self.results_list.currentItem()
            if current_item:
                self.on_item_activated(current_item)
        elif event.key() == Qt.Key.Key_Down:
            current_row = self.results_list.currentRow()
            if current_row < self.results_list.count() - 1:
                self.results_list.setCurrentRow(current_row + 1)
        elif event.key() == Qt.Key.Key_Up:
            current_row = self.results_list.currentRow()
            if current_row > 0:
                self.results_list.setCurrentRow(current_row - 1)
        else:
            super().keyPressEvent(event)


def show_command_palette(parent=None) -> Optional[str]:
    """Show the command palette.
    
    Args:
        parent: Parent widget
        
    Returns:
        The executed command ID or None if cancelled
    """
    palette = CommandPalette(parent)
    
    # Center on parent
    if parent:
        rect = parent.geometry()
        palette.move(
            rect.center().x() - palette.width() // 2,
            rect.center().y() - palette.height() // 2
        )
    
    result = palette.exec()
    
    if result == QDialog.DialogCode.Accepted:
        # Get the last executed command
        return getattr(palette, '_last_command', None)
    
    return None
